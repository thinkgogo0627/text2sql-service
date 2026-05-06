"""
Text2SQL 평가 스크립트.

실행:
    python mlflow_experiments/text2sql_eval/eval/run_eval.py \
      --run_name brain_fewshot_v1 --prompt_version v1
"""
import argparse
import os
import re
import sys
import json
import time
import asyncio
import csv
from pathlib import Path

import mlflow
import yaml
from sqlalchemy import text as sa_text
from dotenv import load_dotenv

load_dotenv("infra/.env")

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agent.brain import generate_sql, extract_company_keywords, build_company_search_sql
from rag.embedder import search_schema
from pipeline.load.postgres_loader import get_engine

EVAL_DIR = Path(__file__).parent
PROMPTS_DIR = EVAL_DIR.parent / "prompts"
TEST_CASES_PATH = EVAL_DIR / "test_cases.json"
RESULTS_PATH = EVAL_DIR / "results_v3.csv"


def load_test_cases() -> list:
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_config() -> dict:
    with open(PROMPTS_DIR / "brain_v1.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_valid_sql(sql: str) -> bool:
    """생성된 SQL이 SELECT로 시작하는 유효한 쿼리인지 확인."""
    return bool(sql and sql.strip().upper().startswith(("SELECT", "WITH")))


def execute_sql_on_db(sql: str) -> tuple[list, str]:
    """
    SQL을 실제 PostgreSQL에 실행하고 결과 반환.
    Returns: (result_rows, error_msg)
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(sa_text(sql))
            rows = [dict(row._mapping) for row in result]
            return rows, ""
    except Exception as e:
        return [], str(e)


def value_match(result_rows: list, expected_values: list) -> bool:
    """핵심 수치가 결과에 포함되는지 확인."""
    if not expected_values:
        return True  # expected_values 비어있으면 SQL 실행 성공 여부만 체크
    result_values = []
    for row in result_rows:
        for v in row.values():
            if isinstance(v, (int, float)):
                result_values.append(v)
    return all(v in result_values for v in expected_values)


def resolve_company_names(user_query: str) -> str:
    """유저 쿼리에서 기업명을 추출하고 company_dim에서 실제 이름 조회."""
    keywords = extract_company_keywords(user_query)
    if not keywords:
        return ""
    search_sql = build_company_search_sql(keywords)
    rows, err = execute_sql_on_db(search_sql)
    if err or not rows:
        return "매칭 없음"
    return ", ".join(row["corp_name"] for row in rows)


async def run_single_case(tc: dict, schema_context: str) -> dict:
    start = time.time()

    company_matches = resolve_company_names(tc["user_query"])

    sql = await generate_sql(
        user_query=tc["user_query"],
        schema_context=schema_context,
        chat_history=[],
        error_feedback="",
        company_matches=company_matches,
    )
    latency_ms = int((time.time() - start) * 1000)

    valid = is_valid_sql(sql)

    # DB 실제 실행
    result_rows, error_msg = execute_sql_on_db(sql) if valid else ([], "Invalid SQL")
    exec_success = len(result_rows) > 0 and not error_msg
    exec_accuracy = value_match(result_rows, tc.get("expected_values", []))

    return {
        "id": tc["id"],
        "difficulty": tc.get("difficulty", ""),
        "category": tc.get("category", ""),
        "user_query": tc["user_query"],
        "generated_sql": sql,
        "valid_sql": valid,
        "exec_success": exec_success,
        "exec_accuracy": exec_accuracy,
        "error_msg": error_msg,
        "latency_ms": latency_ms,
    }


async def run_evaluation(run_name: str, prompt_version: str):
    test_cases = load_test_cases()
    prompt_config = load_prompt_config()

    # 스키마 컨텍스트 사전 조회
    schema_context = search_schema("재무 영업이익 기업", top_k=3)

    results = []
    for tc in test_cases:
        print(f"Evaluating: {tc['id']} - {tc['user_query'][:40]}...")
        result = await run_single_case(tc, schema_context)
        results.append(result)
        print(f"  valid={result['valid_sql']}, exec_success={result['exec_success']}, exec_accuracy={result['exec_accuracy']}, latency={result['latency_ms']}ms")

    # 메트릭 계산
    valid_sql_rate = sum(r["valid_sql"] for r in results) / len(results)
    exec_success_rate = sum(r["exec_success"] for r in results) / len(results)
    execution_accuracy = sum(r["exec_accuracy"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    # CSV 저장
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # MLflow 기록
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("text2sql_eval")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("num_examples", prompt_config.get("num_examples", 2))
        mlflow.log_param("model", prompt_config.get("model"))
        mlflow.log_metric("execution_accuracy", execution_accuracy)
        mlflow.log_metric("valid_sql_rate", valid_sql_rate)
        mlflow.log_metric("exec_success_rate", exec_success_rate)
        mlflow.log_metric("avg_latency_ms", avg_latency)
        mlflow.log_artifact(str(PROMPTS_DIR / "brain_v1.yaml"))
        mlflow.log_artifact(str(RESULTS_PATH))

    print(f"\n=== 평가 결과 ===")
    print(f"Valid SQL Rate:       {valid_sql_rate:.2%}")
    print(f"Exec Success Rate:    {exec_success_rate:.2%}")
    print(f"Execution Accuracy:   {execution_accuracy:.2%}")
    print(f"Avg Latency:          {avg_latency:.0f}ms")
    print(f"Results saved to:     {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text2SQL 평가")
    parser.add_argument("--run_name", default="brain_fewshot_v1", help="MLflow run name")
    parser.add_argument("--prompt_version", default="v1", help="프롬프트 버전")
    args = parser.parse_args()
    asyncio.run(run_evaluation(run_name=args.run_name, prompt_version=args.prompt_version))
