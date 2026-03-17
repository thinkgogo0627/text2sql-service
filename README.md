# Text2SQL & MCP 기반 기업 재무/뉴스 인사이트 에이전트

자연어로 기업 재무 데이터를 조회하고 인사이트 리포트를 생성하는 대화형 BI 에이전트입니다.

## 아키텍처

```
사용자 (Streamlit) → MCP Server (FastAPI) → LangGraph Agent
                                                ↓
                                    retrieve_schema (ChromaDB RAG)
                                                ↓
                                    generate_sql (Qwen 72B)
                                                ↓
                                    execute_sql (PostgreSQL)
                                                ↓
                                    write_report (Qwen 72B)
```

## 빠른 시작

### 1. 환경변수 설정

```bash
cp infra/.env.example infra/.env
# infra/.env 파일을 열어 필수 값 입력:
# DART_API_KEY, POSTGRES_USER, POSTGRES_PASSWORD, TOGETHER_API_KEY
```

### 2. Docker Compose 실행

```bash
cd infra
docker compose up -d
```

서비스 포트:
- PostgreSQL: `5432`
- Airflow: `8080`
- MCP Server: `8001`
- ChromaDB: `8002`
- MLflow: `5000`
- Frontend (Streamlit): `8501`

### 3. DB 초기화

```bash
python pipeline/load/postgres_loader.py
```

### 4. 스키마 임베딩 (ChromaDB)

```bash
python -c "from rag.embedder import embed_schema_metadata; embed_schema_metadata()"
```

### 5. 재무 데이터 수집

Airflow UI (`http://localhost:8080`)에서 `dart_financial_pipeline` DAG를 수동 트리거하거나:

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

## 평가 실행

```bash
# Text2SQL 정확도 평가
python mlflow_experiments/text2sql_eval/eval/run_eval.py

# 감성분석 평가
python mlflow_experiments/sentiment_eval/eval/run_eval.py
```

MLflow UI에서 결과 확인: `http://localhost:5000`

## 주요 파일 구조

| 경로 | 역할 |
|------|------|
| `agent/graph.py` | LangGraph 상태 그래프 |
| `agent/brain.py` | Qwen 72B SQL 생성 / 리포트 작성 |
| `agent/worker.py` | Llama 8B 감성분석 / 요약 |
| `mcp_server/server.py` | FastAPI MCP 서버 |
| `rag/embedder.py` | pplx-embed + ChromaDB |
| `pipeline/extract/dart_api.py` | OpenDART API 호출 |
| `pipeline/transform/financials.py` | Wide→Long Unpivot |
| `pipeline/load/postgres_loader.py` | PostgreSQL DDL + Upsert |

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/query` | 자연어 → 재무 인사이트 리포트 |
| GET | `/health` | 서버 상태 |
| GET | `/tools/schema` | ChromaDB 스키마 검색 |
| POST | `/tools/sql` | SQL 실행 (SELECT만) |
| POST | `/tools/scrape` | Playwright 크롤링 |
| POST | `/tools/sentiment` | 감성분석 / 요약 |

## 보안 정책

- SQL 실행 시 SELECT 외 DDL/DML 키워드는 422 에러로 차단
- 모든 환경변수는 `.env` 파일에서 로드 (코드 하드코딩 금지)
- LangGraph retry_count 최대 3회 제한
