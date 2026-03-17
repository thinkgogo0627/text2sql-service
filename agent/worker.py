import os
from openai import AsyncOpenAI

SENTIMENT_PROMPT = """
아래 텍스트를 분석하여 감성 점수와 3줄 요약을 반환해.

규칙:
- sentiment_score: -1.0(매우 부정) ~ 1.0(매우 긍정) 사이의 실수
- summary: 핵심 내용 3줄 요약 (각 줄은 한 문장)
- JSON 형식으로만 반환. 설명 없이.

형식:
{{"sentiment_score": 0.5, "summary": "첫 번째 요약 문장.\\n두 번째 요약 문장.\\n세 번째 요약 문장."}}

[텍스트]
{text}
"""


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )


async def analyze_sentiment(text: str) -> dict:
    """
    Llama 8B를 사용하여 텍스트의 감성 분석 및 3줄 요약을 수행한다.

    Args:
        text: 분석할 텍스트 (공시 제목, 뉴스 본문 등)

    Returns:
        {"sentiment_score": float, "summary": str}
    """
    client = _get_client()
    model = os.getenv("MODEL_WORKER", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

    prompt = SENTIMENT_PROMPT.format(text=text[:2000])  # 최대 2000자

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()

    try:
        import json
        # 마크다운 코드블록 제거
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()
        result = json.loads(raw)
        score = float(result.get("sentiment_score", 0.0))
        score = max(-1.0, min(1.0, score))  # -1 ~ 1 범위 클리핑
        return {
            "sentiment_score": score,
            "summary": result.get("summary", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "sentiment_score": 0.0,
            "summary": raw[:300],
        }
