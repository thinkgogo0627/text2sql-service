"""
Text2SQL 평가 스크립트.

실행:
    python mlflow_experiments/text2sql_eval/eval/run_eval.py
"""
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agent.brain import generate_sql
from rag.embedder import search_schema

EVAL_DIR = Path(__file__).parent
PROMPTS_DIR = EVAL_DIR.parent / "prompts"
TEST_CASES_PATH = EVAL_DIR / "test_cases.json"
RESULTS_PATH = EVAL_DIR / "results_v1.csv"


def load_test_cases() -> list:
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_config() -> dict:
    with open(PROMPTS_DIR / "brain_v1.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_valid_sql(sql: str) -> bool:
    """생성된 SQL이 SELECT로 시작하는 유효한 쿼리인지 확인."""
    return bool(sql and sql.strip().upper().startswith("SELECT"))


def check_keywords(sql: str, keywords: list) -> bool:
    """SQL에 필수 키워드가 모두 포함되는지 확인."""
    sql_upper = sql.upper()
    return all(kw.upper() in sql_upper for kw in keywords)


async def run_single_case(tc: dict, schema_context: str) -> dict:
    start = time.time()
    sql = await generate_sql(
        user_query=tc["user_query"],
        schema_context=schema_context,
        chat_history=[],
        error_feedback="",
    )
    latency_ms = int((time.time() - start) * 1000)

    valid = is_valid_sql(sql)
    keyword_match = check_keywords(sql, tc.get("expected_keywords", []))

    return {
        "id": tc["id"],
        "user_query": tc["user_query"],
        "generated_sql": sql,
        "valid_sql": valid,
        "keyword_match": keyword_match,
        "latency_ms": latency_ms,
    }


async def run_evaluation():
    test_cases = load_test_cases()
    prompt_config = load_prompt_config()

    # 스키마 컨텍스트 사전 조회
    schema_context = search_schema("재무 영업이익 기업", top_k=3)

    results = []
    for tc in test_cases:
        print(f"Evaluating: {tc['id']} - {tc['user_query'][:40]}...")
        result = await run_single_case(tc, schema_context)
        results.append(result)
        print(f"  valid={result['valid_sql']}, keyword_match={result['keyword_match']}, latency={result['latency_ms']}ms")

    # 메트릭 계산
    valid_sql_rate = sum(r["valid_sql"] for r in results) / len(results)
    keyword_match_rate = sum(r["keyword_match"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    # execution_accuracy: valid + keyword_match 모두 통과 비율
    execution_accuracy = sum(r["valid_sql"] and r["keyword_match"] for r in results) / len(results)
    retry_rate = 0.0  # 이 스크립트에서는 재시도 없음

    # CSV 저장
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # MLflow 기록
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("text2sql_eval")

    with mlflow.start_run(run_name="brain_fewshot_v1"):
        mlflow.log_param("prompt_version", prompt_config.get("version", "v1"))
        mlflow.log_param("num_examples", prompt_config.get("num_examples", 2))
        mlflow.log_param("model", prompt_config.get("model"))
        mlflow.log_metric("execution_accuracy", execution_accuracy)
        mlflow.log_metric("valid_sql_rate", valid_sql_rate)
        mlflow.log_metric("retry_rate", retry_rate)
        mlflow.log_metric("avg_latency_ms", avg_latency)
        mlflow.log_metric("keyword_match_rate", keyword_match_rate)
        mlflow.log_artifact(str(PROMPTS_DIR / "brain_v1.yaml"))
        mlflow.log_artifact(str(RESULTS_PATH))

    print(f"\n=== 평가 결과 ===")
    print(f"Valid SQL Rate:      {valid_sql_rate:.2%}")
    print(f"Keyword Match Rate:  {keyword_match_rate:.2%}")
    print(f"Execution Accuracy:  {execution_accuracy:.2%}")
    print(f"Avg Latency:         {avg_latency:.0f}ms")
    print(f"Results saved to:    {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
