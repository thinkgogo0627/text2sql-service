"""
Sentiment 분석 평가 스크립트.

실행:
    python mlflow_experiments/sentiment_eval/eval/run_eval.py
"""
import os
import sys
import asyncio
import json
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agent.worker import analyze_sentiment

TEST_CASES = [
    {
        "id": "s_001",
        "text": "삼성전자, 2023년 4분기 영업이익 2.8조원으로 전년 대비 흑자 전환",
        "expected_polarity": "positive",
    },
    {
        "id": "s_002",
        "text": "SK하이닉스, 반도체 업황 부진으로 2023년 적자 기록. 대규모 구조조정 불가피",
        "expected_polarity": "negative",
    },
    {
        "id": "s_003",
        "text": "현대자동차 1분기 실적 발표. 전분기 대비 소폭 상승",
        "expected_polarity": "neutral",
    },
]

POLARITY_MAP = {"positive": 0.3, "neutral": 0.0, "negative": -0.3}


async def run_evaluation():
    results = []
    for tc in TEST_CASES:
        result = await analyze_sentiment(tc["text"])
        score = result.get("sentiment_score", 0.0)
        expected_threshold = POLARITY_MAP[tc["expected_polarity"]]
        correct = (
            (tc["expected_polarity"] == "positive" and score > 0.2) or
            (tc["expected_polarity"] == "negative" and score < -0.2) or
            (tc["expected_polarity"] == "neutral" and -0.2 <= score <= 0.2)
        )
        results.append({
            "id": tc["id"],
            "text": tc["text"][:50],
            "expected": tc["expected_polarity"],
            "score": score,
            "correct": correct,
            "summary": result.get("summary", "")[:100],
        })
        print(f"{tc['id']}: score={score:.2f}, correct={correct}")

    accuracy = sum(r["correct"] for r in results) / len(results)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("sentiment_eval")

    with mlflow.start_run(run_name="worker_llama_v1"):
        mlflow.log_param("model", os.getenv("MODEL_WORKER", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"))
        mlflow.log_metric("sentiment_accuracy", accuracy)

    print(f"\nSentiment Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
