import os
import re

from openai import AsyncOpenAI

# 영문/약칭 → 한글 정식명 매핑 (LIKE 검색용)
COMPANY_ALIASES = {
    "SK하이닉스": "에스케이하이닉스",
    "SK이노베이션": "에스케이이노베이션",
    "SK텔레콤": "에스케이텔레콤",
    "POSCO홀딩스": "포스코홀딩스",
    "POSCO": "포스코",
    "NAVER": "네이버",
    "LIG넥스원": "엘아이지넥스원",
    "신한지주": "신한금융지주",
}

_STOP_WORDS = re.compile(
    r"(\d+[년~]?|최근|알려줘|비교해줘|추이|변화|얼마|어떻게|상위|하위|가장|높은|낮은|기업|"
    r"영업이익률?|당기순이익|매출액|부채총계|자산총계|자본총계|부채비율|유동비율|유동자산|유동부채|"
    r"법인세비용|매출총이익률?|매출원가|영업비용|현금및현금성자산|무형자산|유형자산|"
    r"ROE|자기자본비율|자본금|순이익|매출|이익|비용|자산|부채|자본|비율|증가율)"
)


def extract_company_keywords(user_query: str) -> list[str]:
    """유저 쿼리에서 기업명 후보 키워드를 추출하고 별칭을 추가 반환."""
    cleaned = _STOP_WORDS.sub("", user_query)
    cleaned = re.sub(r"[~와과의를]", " ", cleaned)
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) >= 2]

    results = []
    for token in tokens:
        results.append(token)
        if token in COMPANY_ALIASES:
            results.append(COMPANY_ALIASES[token])
    return results


def build_company_search_sql(keywords: list[str]) -> str:
    """기업명 LIKE 검색 SQL 생성."""
    if not keywords:
        return ""
    safe = [k.replace("'", "''") for k in keywords]
    conds = " OR ".join(f"corp_name LIKE '%{k}%'" for k in safe)
    return f"SELECT DISTINCT corp_name FROM company_dim WHERE {conds} LIMIT 10"


BRAIN_SQL_PROMPT = """
너는 DART 재무 데이터 전문 Text2SQL 모델이야.
아래 스키마와 예시를 참고해서 PostgreSQL 쿼리를 생성해.

[스키마]
{schema_context}

[기업명 검색 규칙]
1. DB에 기업명이 '삼성전자(주)', 'SK하이닉스(주)' 형태로 저장됨
2. 기업명 검색 시 반드시 LIKE '%기업명%' 패턴 사용. = 연산자 사용 금지.
3. 단, 그룹명(삼성, SK, LG, 현대)은 절대 단독으로 LIKE 검색 금지
   → '삼성전자'라고 하면 LIKE '%삼성전자%'
   → '삼성바이오'라고 하면 LIKE '%삼성바이오%'
   → '삼성'만 입력 시 corp_name도 SELECT에 포함 + LIMIT 5로 반환
4. Few-shot 예시 WHERE절도 전부 LIKE로 수정:
   WHERE c.corp_name LIKE '%삼성전자%'
   WHERE c.corp_name LIKE '%현대오토에버%'

[fs_div / sj_div 규칙]
1. fs_div: 재무제표 구분. CFS(연결) / OFS(별도). 기본값은 항상 'CFS'.
2. sj_div: 재무제표 유형. BS(재무상태표) / IS(손익계산서) / CF(현금흐름표) / CIS(포괄손익계산서) / SCE(자본변동표).
3. 반드시 fs_div와 sj_div를 함께 사용할 것. fs_div에 'BS','IS' 등을 넣으면 안 됨.
   - 자산/부채/자본/유동자산/유동부채 등 → sj_div='BS'
   - 매출액/영업이익 → sj_div='IS'
   - 당기순이익/법인세비용/영업비용/매출총이익 → sj_div IN ('IS','CIS') (기업마다 다름)
   - 현금흐름 항목 → sj_div='CF'
4. 동일 계정이 IS와 CIS 양쪽에 존재할 수 있으므로, 손익 관련 계정은 sj_div IN ('IS','CIS')로 조회하고 LIMIT 1 또는 GROUP BY로 중복을 제거할 것.

[계정명(account_nm) 규칙]
1. DB에 '당기순이익(손실)' 처럼 괄호가 포함된 계정명이 있음
2. 따라서 계정명도 LIKE '%키워드%' 패턴 사용을 권장
   예: account_nm LIKE '%당기순이익%' (당기순이익, 당기순이익(손실) 모두 매칭)
3. '부채비율', 'ROE', '영업이익률' 등 파생지표는 DB에 저장되지 않음
   → 반드시 원본 계정(부채총계/자본총계/영업이익/매출액 등)을 조회하여 계산할 것

[예시 1]
질문: 삼성전자 2023년 영업이익 알려줘
SQL: SELECT c.corp_name, f.bsns_year, f.amount
     FROM financial_fact f
     JOIN company_dim c ON f.corp_code = c.corp_code
     WHERE c.corp_name LIKE '%삼성전자%'
     AND f.account_nm = '영업이익'
     AND f.bsns_year = 2023
     AND f.fs_div = 'CFS'
     AND f.sj_div IN ('IS','CIS')
     LIMIT 1

[예시 2]
질문: 현대오토에버 최근 3년 영업이익률 추이
SQL: SELECT f.bsns_year,
     MAX(CASE WHEN f.account_nm = '영업이익' THEN f.amount END) as 영업이익,
     MAX(CASE WHEN f.account_nm = '매출액' THEN f.amount END) as 매출액,
     ROUND(MAX(CASE WHEN f.account_nm = '영업이익' THEN f.amount END) * 100.0
           / NULLIF(MAX(CASE WHEN f.account_nm = '매출액' THEN f.amount END), 0), 2) as 영업이익률
     FROM financial_fact f
     JOIN company_dim c ON f.corp_code = c.corp_code
     WHERE c.corp_name LIKE '%현대오토에버%'
     AND f.bsns_year >= 2021
     AND f.fs_div = 'CFS'
     AND f.sj_div IN ('IS','CIS')
     GROUP BY f.bsns_year
     ORDER BY f.bsns_year

[예시 3]
질문: LG전자 2023년 부채비율 알려줘
SQL: SELECT c.corp_name, f.bsns_year,
     ROUND(MAX(CASE WHEN f.account_nm LIKE '%부채총계%' THEN f.amount END) * 100.0
           / NULLIF(MAX(CASE WHEN f.account_nm LIKE '%자본총계%' THEN f.amount END), 0), 2) as 부채비율
     FROM financial_fact f
     JOIN company_dim c ON f.corp_code = c.corp_code
     WHERE c.corp_name LIKE '%LG전자%'
     AND f.bsns_year = 2023
     AND f.fs_div = 'CFS'
     AND f.sj_div = 'BS'
     GROUP BY c.corp_name, f.bsns_year

[DB 기업명 매칭 결과]
아래는 company_dim 테이블에서 조회한 실제 기업명이다.
SQL의 WHERE절에서 corp_name 조건은 반드시 아래 이름 중 하나를 LIKE 패턴에 사용할 것.
사용자가 입력한 이름이 아닌, 아래 DB 실제 이름을 사용해야 한다.
{company_matches}

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
    company_matches: str = "",
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
        company_matches=company_matches or "없음",
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
