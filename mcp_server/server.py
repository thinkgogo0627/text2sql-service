import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mcp_server.tools.schema_tool import get_database_schema_tool
from mcp_server.tools.sql_tool import execute_financial_sql_tool
from mcp_server.tools.playwright_tool import run_playwright_scraper_tool

app = FastAPI(title="Text2SQL MCP Server", version="1.0.0")


# ---------- Request/Response 모델 ----------

class QueryRequest(BaseModel):
    user_query: str
    chat_history: list = []
    stream: bool = False


class QueryResponse(BaseModel):
    report: str
    sql_query: str
    raw_data: list
    sources: list[str]
    execution_time_ms: int


class SchemaRequest(BaseModel):
    keyword: str
    top_k: int = 3


class SqlRequest(BaseModel):
    sql_query: str


class ScrapeRequest(BaseModel):
    target_url: str
    intent: str = ""


class NaverNewsRequest(BaseModel):
    stock_code: str
    corp_code: str = ""
    max_items: int = 10


class SentimentRequest(BaseModel):
    text: str
    corp_code: str = ""


# ---------- 엔드포인트 ----------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "mcp_server"}


@app.get("/tools/schema")
async def schema_search(keyword: str, top_k: int = 3):
    return await get_database_schema_tool(keyword, top_k=top_k)


@app.post("/tools/sql")
async def sql_execute(req: SqlRequest):
    result = await execute_financial_sql_tool(req.sql_query)
    if result.get("status") == "error" and result.get("error_type") == "SecurityViolation":
        raise HTTPException(status_code=422, detail=result)
    return result


@app.post("/tools/scrape")
async def scrape(req: ScrapeRequest):
    return await run_playwright_scraper_tool(req.target_url, req.intent)


@app.post("/tools/naver-news")
async def naver_news(req: NaverNewsRequest):
    from mcp_server.tools.playwright_tool import scrape_naver_finance_news
    return await scrape_naver_finance_news(req.stock_code, req.corp_code, req.max_items)


@app.post("/tools/sentiment")
async def sentiment_analysis(req: SentimentRequest):
    from agent.worker import analyze_sentiment
    try:
        result = await analyze_sentiment(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    from agent.graph import run_agent

    start = time.time()
    result = await run_agent(
        user_query=req.user_query,
        chat_history=req.chat_history,
    )
    elapsed_ms = int((time.time() - start) * 1000)

    return QueryResponse(
        report=result.get("final_report", "결과를 생성할 수 없습니다."),
        sql_query=result.get("generated_sql", ""),
        raw_data=result.get("query_result", []),
        sources=result.get("sources", []),
        execution_time_ms=elapsed_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server.server:app", host="0.0.0.0", port=8001, reload=False)
