# Text2SQL & MCP 기반 기업 재무/뉴스 인사이트 에이전트

자연어로 기업 재무 데이터를 조회하고 인사이트 리포트를 생성하는 대화형 BI 에이전트입니다. DART 공시 재무 데이터를 PostgreSQL에 적재하고, LangGraph 에이전트가 ChromaDB 스키마 RAG → Text2SQL → SQL 실행 → 리포트 생성을 오케스트레이션합니다.

## 아키텍처

```
사용자 (Streamlit)
      │  /query
      ▼
MCP Server (FastAPI)  →  LangGraph Agent
                              │
        retrieve_schema   ── ChromaDB RAG로 관련 테이블 DDL 검색
                              │
        resolve_company   ── company_dim에서 실제 기업명 매칭 (LIKE 보정)
                              │
        generate_sql      ── Brain(Qwen 72B) Few-shot Text2SQL
                              │
        execute_sql       ── PostgreSQL 실행 (SELECT 전용)
                              │
          ├─ SQL 에러     ─→ generate_sql 재시도 (retry_count < 3)
          ├─ 데이터 있음  ─→ write_report (Qwen 72B)
          └─ 데이터 없음  ─→ run_scraper (Playwright) ─→ write_report
```

- **Brain (Qwen 72B):** SQL 생성 + 자연어 리포트 작성 (`agent/brain.py`)
- **Worker (Llama 8B):** 공시/뉴스 감성분석 + 3줄 요약 (`agent/worker.py`)
- 두 모델 모두 Together.ai API로 서빙

## 빠른 시작

### 1. 환경변수 설정

```bash
cp infra/.env.example infra/.env
# infra/.env 파일을 열어 필수 값 입력:
# DART_API_KEY, POSTGRES_USER, POSTGRES_PASSWORD, TOGETHER_API_KEY
# (선택) MODEL_BRAIN, MODEL_WORKER — 미설정 시 코드 기본값 사용
```

### 2. Docker Compose 실행

```bash
cd infra
docker compose up -d
```

서비스 포트 (호스트 기준):

| 서비스 | URL | 컨테이너 내부 포트 |
|---|---|---|
| Frontend (Streamlit) | http://localhost:8501 | 8501 |
| MCP Server (FastAPI) | http://localhost:8001 | 8001 |
| Airflow | http://localhost:8088 | 8080 |
| ChromaDB | http://localhost:8002 | 8000 |
| MLflow | http://localhost:5000 | 5000 |
| PostgreSQL | localhost:5433 | 5432 |

> 컨테이너 간 통신은 내부 포트(예: `postgres:5432`)를 사용하고, 호스트에서 접속할 때는 위 표의 호스트 포트를 사용합니다.

### 3. DB 초기화

```bash
python pipeline/load/postgres_loader.py
```

### 4. 스키마 임베딩 (ChromaDB)

```bash
python -c "from rag.embedder import embed_schema_metadata; embed_schema_metadata()"
```

### 5. 재무 데이터 수집

Airflow UI (`http://localhost:8088`)에서 `dart_financial_pipeline` DAG를 수동 트리거하거나:

```bash
python -c "
from pipeline.extract.dart_api import extract_dart_api
from pipeline.transform.financials import transform_and_load_financials
from pipeline.load.postgres_loader import upsert_financials

df_raw = extract_dart_api('00126380', 2023)   # 삼성전자
df_long = transform_and_load_financials(df_raw)
upsert_financials(df_long)
"
```

### 6. 프론트엔드 접속

브라우저에서 `http://localhost:8501` 접속

## 데이터 파이프라인 (Airflow DAG)

| DAG | 스케줄 | 흐름 |
|---|---|---|
| `dart_financial_pipeline` | `@quarterly` | seed_company_dim → extract(DART API) → transform(Wide→Long) → upsert(financial_fact) |
| `sentiment_pipeline` | 3일 주기 | fetch_naver_news(Playwright) → sentiment_analysis(Llama 8B) → load(event_logs_fact) |

## 평가 실행

```bash
# Text2SQL 정확도 평가 (Brain)
python mlflow_experiments/text2sql_eval/eval/run_eval.py

# 감성분석 평가 (Worker vs Judge: Llama 8B를 Qwen 72B로 채점)
python mlflow_experiments/sentiment_eval/eval/run_eval.py
```

MLflow UI에서 결과 확인: `http://localhost:5000`

## 주요 파일 구조

| 경로 | 역할 |
|------|------|
| `agent/graph.py` | LangGraph 상태 그래프 (노드/조건부 엣지/재시도) |
| `agent/brain.py` | Qwen 72B SQL 생성 / 리포트 작성 |
| `agent/worker.py` | Llama 8B 감성분석 / 요약 |
| `mcp_server/server.py` | FastAPI MCP 서버 |
| `mcp_server/tools/sql_tool.py` | SELECT 전용 SQL 실행 + 보안 검사 |
| `mcp_server/tools/playwright_tool.py` | Playwright 크롤러 (DART / 네이버 뉴스) |
| `rag/embedder.py` | pplx-embed + ChromaDB 스키마 RAG |
| `pipeline/extract/dart_api.py` | OpenDART API 호출 |
| `pipeline/transform/financials.py` | Wide→Long Unpivot (당기/전기/전전기) |
| `pipeline/load/postgres_loader.py` | PostgreSQL DDL + Upsert |

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/query` | 자연어 → 재무 인사이트 리포트 |
| GET | `/health` | 서버 상태 |
| GET | `/tools/schema` | ChromaDB 스키마 검색 |
| POST | `/tools/sql` | SQL 실행 (SELECT만, DDL/DML은 422 차단) |
| POST | `/tools/scrape` | Playwright 범용 크롤링 |
| POST | `/tools/naver-news` | 네이버 증권 뉴스 크롤링 |
| POST | `/tools/sentiment` | 감성분석 / 요약 (Llama 8B) |

## 데이터 모델

| 테이블 | 설명 |
|---|---|
| `company_dim` | 기업 메타정보 (corp_code PK, 상장사는 stock_code 보유) |
| `financial_fact` | DART 재무 수치 (fs_div=CFS/OFS, sj_div=BS/IS/CF/CIS) |
| `event_logs_fact` | 공시·뉴스 로그 (sentiment_score, 3줄 summary) |

## 보안 정책

- SQL 실행 시 SELECT/WITH 외 DDL/DML 키워드(INSERT·UPDATE·DELETE·DROP 등)는 422 에러로 차단
- 모든 환경변수는 `.env` 파일에서 로드 (코드 하드코딩 금지)
- LangGraph SQL 재시도는 최대 3회(`MAX_RETRY`)로 제한