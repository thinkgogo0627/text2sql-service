import os
import operator
from typing import TypedDict, Annotated

import httpx
from langgraph.graph import StateGraph, END

from agent.brain import generate_sql, write_report

MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8001")
MAX_RETRY = 3


class AgentState(TypedDict):
    user_query: str
    chat_history: Annotated[list, operator.add]   # 세션 누적
    schema_context: str        # RAG 결과 캐싱 — 동일 테이블 재질문 시 재검색 스킵
    cached_schema_key: str     # 마지막으로 검색한 스키마 키
    generated_sql: str
    query_result: list
    final_report: str
    error: str
    retry_count: int
    sources: list


# ---------- 노드 구현 ----------

async def retrieve_schema(state: AgentState) -> AgentState:
    """ChromaDB에서 관련 DDL 검색 (schema_context 캐시 활용)."""
    query = state["user_query"]

    # 캐시 히트: 동일 쿼리 키는 재검색 스킵
    if state.get("cached_schema_key") == query and state.get("schema_context"):
        return state

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{MCP_BASE_URL}/tools/schema",
            params={"keyword": query, "top_k": 3},
        )
        data = resp.json()

    schema_context = data.get("schema_context", "")
    return {
        **state,
        "schema_context": schema_context,
        "cached_schema_key": query,
        "error": "",
    }


async def generate_sql_node(state: AgentState) -> AgentState:
    """Brain(Qwen 72B)에게 Few-shot 프롬프트로 SQL 생성 요청."""
    error_feedback = state.get("error", "")
    sql = await generate_sql(
        user_query=state["user_query"],
        schema_context=state["schema_context"],
        chat_history=state["chat_history"],
        error_feedback=error_feedback,
    )
    return {**state, "generated_sql": sql, "error": ""}


async def execute_sql_node(state: AgentState) -> AgentState:
    """/tools/sql 호출 → PostgreSQL 실행."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{MCP_BASE_URL}/tools/sql",
            json={"sql_query": state["generated_sql"]},
        )
        result = resp.json()

    if result.get("status") == "error":
        return {
            **state,
            "error": f"{result.get('error_type')}: {result.get('error_message')}",
            "query_result": [],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    return {
        **state,
        "query_result": result.get("data", []),
        "sources": ["financial_fact", "company_dim"],
        "error": "",
    }


async def check_data_node(state: AgentState) -> AgentState:
    """DB에 데이터 없으면 Playwright 전환 분기 플래그 설정."""
    return state  # 라우팅은 조건부 엣지에서 처리


async def run_scraper_node(state: AgentState) -> AgentState:
    """/tools/scrape 호출 → 실시간 크롤링."""
    # DART 공시 검색 URL 구성
    search_url = (
        "https://dart.fss.or.kr/dsab007/main.do"
        f"?textCrpNm={state['user_query'][:20]}"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{MCP_BASE_URL}/tools/scrape",
            json={"target_url": search_url, "intent": state["user_query"]},
        )
        result = resp.json()

    scraped_text = result.get("text", "")
    return {
        **state,
        "query_result": [{"scraped_text": scraped_text[:3000]}],
        "sources": ["playwright_scraper"],
    }


async def write_report_node(state: AgentState) -> AgentState:
    """Brain에게 Raw Data 기반 자연어 리포트 작성 요청."""
    report = await write_report(
        user_query=state["user_query"],
        raw_data=state["query_result"],
        chat_history=state["chat_history"],
    )

    # 대화 히스토리에 현재 턴 추가
    new_history = [
        {"role": "user", "content": state["user_query"]},
        {"role": "assistant", "content": report},
    ]

    return {
        **state,
        "final_report": report,
        "chat_history": new_history,
    }


def _route_after_sql(state: AgentState) -> str:
    """execute_sql 후 라우팅."""
    if state.get("error"):
        if state.get("retry_count", 0) < MAX_RETRY:
            return "generate_sql"
        return "end_with_error"
    return "check_data"


def _route_after_check(state: AgentState) -> str:
    """check_data 후 라우팅."""
    if not state.get("query_result"):
        return "run_scraper"
    return "write_report"


async def end_with_error_node(state: AgentState) -> AgentState:
    return {
        **state,
        "final_report": "데이터를 찾을 수 없습니다. 질문을 다시 입력하거나 조건을 확인해 주세요.",
    }


# ---------- 그래프 빌드 ----------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_schema", retrieve_schema)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("check_data", check_data_node)
    graph.add_node("run_scraper", run_scraper_node)
    graph.add_node("write_report", write_report_node)
    graph.add_node("end_with_error", end_with_error_node)

    graph.set_entry_point("retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")

    graph.add_conditional_edges(
        "execute_sql",
        _route_after_sql,
        {
            "generate_sql": "generate_sql",
            "check_data": "check_data",
            "end_with_error": "end_with_error",
        },
    )

    graph.add_conditional_edges(
        "check_data",
        _route_after_check,
        {
            "run_scraper": "run_scraper",
            "write_report": "write_report",
        },
    )

    graph.add_edge("run_scraper", "write_report")
    graph.add_edge("write_report", END)
    graph.add_edge("end_with_error", END)

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_agent(user_query: str, chat_history: list = None) -> dict:
    """
    LangGraph 에이전트를 실행한다.

    Returns:
        final_report, generated_sql, query_result, sources 포함 dict
    """
    graph = _get_graph()

    initial_state: AgentState = {
        "user_query": user_query,
        "chat_history": chat_history or [],
        "schema_context": "",
        "cached_schema_key": "",
        "generated_sql": "",
        "query_result": [],
        "final_report": "",
        "error": "",
        "retry_count": 0,
        "sources": [],
    }

    result = await graph.ainvoke(initial_state)
    return result
