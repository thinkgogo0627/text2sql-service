import os
from openai import AsyncOpenAI

BRAIN_SQL_PROMPT = """
너는 DART 재무 데이터 전문 Text2SQL 모델이야.
아래 스키마와 예시를 참고해서 PostgreSQL 쿼리를 생성해.

[스키마]
{schema_context}

[예시 1]
질문: 삼성전자 2023년 영업이익 알려줘
SQL: SELECT c.corp_name, f.bsns_year, f.amount
     FROM financial_fact f
     JOIN company_dim c ON f.corp_code = c.corp_code
     WHERE c.corp_name = '삼성전자'
     AND f.account_nm = '영업이익'
     AND f.bsns_year = 2023
     AND f.fs_div = 'CFS'

[예시 2]
질문: 현대오토에버 최근 3년 영업이익률 추이
SQL: SELECT f.bsns_year,
     MAX(CASE WHEN f.account_nm = '영업이익' THEN f.amount END) as 영업이익,
     MAX(CASE WHEN f.account_nm = '매출액' THEN f.amount END) as 매출액,
     ROUND(MAX(CASE WHEN f.account_nm = '영업이익' THEN f.amount END) * 100.0
           / NULLIF(MAX(CASE WHEN f.account_nm = '매출액' THEN f.amount END), 0), 2) as 영업이익률
     FROM financial_fact f
     JOIN company_dim c ON f.corp_code = c.corp_code
     WHERE c.corp_name = '현대오토에버'
     AND f.bsns_year >= 2021
     AND f.fs_div = 'CFS'
     GROUP BY f.bsns_year
     ORDER BY f.bsns_year

[이전 대화]
{chat_history}

[에러 피드백] (재시도 시)
{error_feedback}

[질문]
{user_query}

SQL만 반환할 것. 설명 없이.
"""

REPORT_PROMPT = """
너는 기업 재무 분석 전문가야.
아래 데이터를 바탕으로 사용자 질문에 대한 자연어 인사이트 리포트를 작성해.

규칙:
- 금액은 억 단위로 변환하여 표시 (예: 1,234,000,000원 → 약 12.3억원)
- 핵심 수치를 먼저 제시하고, 해석/분석을 이어서 작성
- 마크다운 형식 사용 (제목, 목록, 굵은 글씨 활용)
- 3~5개 문단 이내로 간결하게 작성

[사용자 질문]
{user_query}

[조회된 데이터]
{raw_data}

[이전 대화]
{chat_history}
"""


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )


async def generate_sql(
    user_query: str,
    schema_context: str,
    chat_history: list,
    error_feedback: str = "",
) -> str:
    """
    Qwen 72B를 사용하여 SQL을 생성한다.

    Returns:
        생성된 SQL 문자열
    """
    client = _get_client()
    model = os.getenv("MODEL_BRAIN", "Qwen/Qwen2.5-72B-Instruct-Turbo")

    chat_str = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in chat_history[-6:]  # 최근 6턴만
    )

    prompt = BRAIN_SQL_PROMPT.format(
        schema_context=schema_context,
        chat_history=chat_str or "없음",
        error_feedback=error_feedback or "없음",
        user_query=user_query,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    sql = response.choices[0].message.content.strip()

    # 마크다운 코드블록 제거
    if sql.startswith("```"):
        lines = sql.splitlines()
        sql = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    return sql


async def write_report(
    user_query: str,
    raw_data: list,
    chat_history: list,
) -> str:
    """
    Qwen 72B를 사용하여 자연어 리포트를 생성한다.

    Returns:
        마크다운 형식의 리포트 문자열
    """
    client = _get_client()
    model = os.getenv("MODEL_BRAIN", "Qwen/Qwen2.5-72B-Instruct-Turbo")

    chat_str = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}"
        for msg in chat_history[-6:]
    )

    prompt = REPORT_PROMPT.format(
        user_query=user_query,
        raw_data=str(raw_data[:50]),  # 최대 50행
        chat_history=chat_str or "없음",
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2048,
    )

    return response.choices[0].message.content.strip()
