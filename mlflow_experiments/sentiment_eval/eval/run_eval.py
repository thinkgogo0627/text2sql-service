"""
Sentiment 분석 평가 스크립트 — Worker(Llama 8B) vs Judge(Qwen 72B)

실행:
    python mlflow_experiments/sentiment_eval/eval/run_eval.py \
      --run_name worker_sentiment_runA_baseline \
      --prompt_version v1 \
      --num_samples 50
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sqlalchemy import text

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent.worker import analyze_sentiment
from pipeline.load.postgres_loader import get_engine

EVAL_DIR = Path(__file__).parent
PROMPTS_DIR = EVAL_DIR.parent / "prompts"


# ── 환경변수 로드 ────────────────────────────────────────────────────────────

def load_env():
    env_path = ROOT / "infra" / ".env"
    load_dotenv(env_path)


# ── YAML 프롬프트 로드 ───────────────────────────────────────────────────────

def load_prompt_template(yaml_path: Path) -> str:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["prompt"]


# ── Judge (Qwen 72B) ─────────────────────────────────────────────────────────

def _get_brain_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )


async def judge_sentiment(text_content: str, prompt_template: str) -> float:
    """Qwen 72B로 감성 점수를 생성한다 (정답 레이블 역할)."""
    client = _get_brain_client()
    model = os.getenv("MODEL_BRAIN", "Qwen/Qwen2.5-72B-Instruct-Turbo")
    prompt = prompt_template.replace("{text}", text_content[:3000])

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=128,
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    result = json.loads(raw)
    score = float(result.get("sentiment_score", 0.0))
    return max(-1.0, min(1.0, score))


# ── DB 샘플 추출 ─────────────────────────────────────────────────────────────

def fetch_samples(num_samples: int) -> list:
    engine = get_engine()
    query = text("""
        SELECT id, corp_code, report_nm, news_content, sentiment_score
        FROM event_logs_fact
        WHERE news_content IS NOT NULL AND LENGTH(news_content) > 10
        ORDER BY RANDOM()
        LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": num_samples}).fetchall()
    return [dict(row._mapping) for row in rows]


# ── 메인 평가 루프 ────────────────────────────────────────────────────────────

async def run_eval(args: argparse.Namespace):
    load_env()

    judge_prompt_template = load_prompt_template(PROMPTS_DIR / "judge_prompt.yaml")
    worker_prompt_file = PROMPTS_DIR / f"worker_{args.prompt_version}.yaml"
    worker_prompt_template = load_prompt_template(worker_prompt_file)

    print(f"Worker 프롬프트: {worker_prompt_file.name}")
    print(f"DB에서 최대 {args.num_samples}개 샘플 추출 중...")
    samples = fetch_samples(args.num_samples)
    print(f"추출 완료: {len(samples)}개\n")

    judge_scores = []
    worker_scores = []
    latencies = []
    records = []

    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        corp_code = sample["corp_code"]
        text_content = sample["news_content"]
        text_preview = text_content[:100].replace("\n", " ")

        print(f"[{i + 1}/{len(samples)}] id={sample_id}, corp_code={corp_code}")

        # Judge score
        try:
            judge_score = await judge_sentiment(text_content, judge_prompt_template)
        except Exception as e:
            print(f"  Judge 실패, skip: {e}")
            continue

        # Worker score + latency
        try:
            t0 = time.perf_counter()
            worker_result = await analyze_sentiment(text_content, worker_prompt_template)
            latency_ms = (time.perf_counter() - t0) * 1000
            worker_score = worker_result["sentiment_score"]
        except Exception as e:
            print(f"  Worker 실패, skip: {e}")
            continue

        abs_error = abs(judge_score - worker_score)
        direction_match = int(np.sign(judge_score) == np.sign(worker_score))

        judge_scores.append(judge_score)
        worker_scores.append(worker_score)
        latencies.append(latency_ms)
        records.append({
            "id": sample_id,
            "corp_code": corp_code,
            "text_preview": text_preview,
            "judge_score": round(judge_score, 4),
            "worker_score": round(worker_score, 4),
            "abs_error": round(abs_error, 4),
            "direction_match": direction_match,
            "latency_ms": round(latency_ms, 1),
        })

        print(
            f"  judge={judge_score:.3f}  worker={worker_score:.3f}  "
            f"abs_err={abs_error:.3f}  latency={latency_ms:.0f}ms"
        )

    if len(judge_scores) < 2:
        print("\n유효 샘플이 2개 미만이라 평가를 중단합니다.")
        return

    judge_arr = np.array(judge_scores)
    worker_arr = np.array(worker_scores)

    mae = mean_absolute_error(judge_arr, worker_arr)
    direction_match_rate = float((np.sign(judge_arr) == np.sign(worker_arr)).mean())
    pearson_r, _ = pearsonr(judge_arr, worker_arr)
    avg_latency = float(np.mean(latencies))

    print(f"\n{'=' * 40}")
    print(f"유효 샘플   : {len(judge_scores)}/{len(samples)}")
    print(f"MAE         : {mae:.4f}")
    print(f"Dir. Match  : {direction_match_rate:.4f}")
    print(f"Pearson R   : {pearson_r:.4f}")
    print(f"Avg Latency : {avg_latency:.0f}ms")
    print(f"{'=' * 40}\n")

    # results.csv 저장
    results_path = EVAL_DIR / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "corp_code", "text_preview",
                "judge_score", "worker_score", "abs_error",
                "direction_match", "latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"결과 저장: {results_path}")

    # MLflow 기록
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("sentiment_eval")

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("model", os.getenv("MODEL_WORKER"))
        mlflow.log_param("judge_model", os.getenv("MODEL_BRAIN"))
        mlflow.log_param("prompt_version", args.prompt_version)
        mlflow.log_param("num_samples", len(samples))
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("direction_match_rate", direction_match_rate)
        mlflow.log_metric("pearson_r", pearson_r)
        mlflow.log_metric("avg_latency_ms", avg_latency)
        mlflow.log_artifact(str(worker_prompt_file))
        mlflow.log_artifact(str(results_path))

    print("MLflow 기록 완료.")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sentiment 분석 평가 (Worker vs Judge)")
    parser.add_argument("--run_name", type=str, default="worker_sentiment_baseline",
                        help="MLflow run 이름")
    parser.add_argument("--prompt_version", type=str, default="v1",
                        help="프롬프트 버전 태그")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="평가에 사용할 샘플 수")
    args = parser.parse_args()

    try:
        loop = asyncio.get_running_loop()
        # Jupyter 등 이미 event loop가 실행 중인 환경
        import nest_asyncio
        nest_asyncio.apply()
        loop.run_until_complete(run_eval(args))
    except RuntimeError:
        # 일반 스크립트 실행 환경
        asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
