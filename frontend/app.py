import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

MCP_URL = os.getenv("MCP_SERVER_URL", "http://mcp_server:8001")

st.set_page_config(
    page_title="기업 재무 인사이트 에이전트",
    page_icon="📊",
    layout="wide",
)

st.title("📊 기업 재무 인사이트 에이전트")
st.caption("자연어로 기업 재무 데이터를 조회하세요. (예: '현대오토에버 2023년 영업이익률 알려줘')")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------- 헬퍼 함수 ----------

def _try_render_chart(df: pd.DataFrame):
    """DataFrame에서 연도별 수치를 감지하여 라인 차트 생성."""
    year_cols = [
        c for c in df.columns
        if "year" in c.lower() or "연도" in c or "bsns_year" in c.lower()
    ]
    numeric_cols = [
        c for c in df.columns
        if c not in year_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if year_cols and numeric_cols:
        year_col = year_cols[0]
        try:
            df_chart = df[[year_col] + numeric_cols].copy()
            df_chart[year_col] = df_chart[year_col].astype(str)
            fig = px.line(
                df_chart,
                x=year_col,
                y=numeric_cols,
                markers=True,
                title="연도별 재무 추이",
            )
            fig.update_layout(xaxis_title="연도", yaxis_title="금액 (원)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


def _render_assistant_message(msg: dict):
    """어시스턴트 메시지 렌더링 (리포트 + 테이블 + SQL + 차트)."""
    st.markdown(msg.get("report", ""))

    raw_data = msg.get("raw_data", [])

    if raw_data:
        df = pd.DataFrame(raw_data)
        st.dataframe(df, use_container_width=True)
        _try_render_chart(df)

    sql = msg.get("sql_query", "")
    if sql:
        with st.expander("SQL 쿼리 보기"):
            st.code(sql, language="sql")

    sources = msg.get("sources", [])
    if sources:
        st.caption(f"데이터 출처: {', '.join(sources)}")

    exec_ms = msg.get("execution_time_ms")
    if exec_ms:
        st.caption(f"응답 시간: {exec_ms}ms")


# ---------- 채팅 히스토리 표시 ----------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            _render_assistant_message(msg)


# ---------- 채팅 입력 ----------

user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            try:
                resp = requests.post(
                    f"{MCP_URL}/query",
                    json={
                        "user_query": user_input,
                        "chat_history": st.session_state.chat_history,
                        "stream": False,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                assistant_msg = {
                    "role": "assistant",
                    "report": data.get("report", ""),
                    "sql_query": data.get("sql_query", ""),
                    "raw_data": data.get("raw_data", []),
                    "sources": data.get("sources", []),
                    "execution_time_ms": data.get("execution_time_ms"),
                }

                _render_assistant_message(assistant_msg)

                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": data.get("report", "")}
                )
                st.session_state.messages.append(assistant_msg)

            except requests.exceptions.ConnectionError:
                st.error("MCP 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            except requests.exceptions.Timeout:
                st.error("요청 시간이 초과되었습니다. 다시 시도해 주세요.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")


# ---------- 사이드바 ----------

with st.sidebar:
    st.header("설정")

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.subheader("예시 질문")
    examples = [
        "삼성전자 2023년 영업이익 알려줘",
        "현대오토에버 최근 3년 영업이익률 추이",
        "LG전자 2022년 자기자본비율",
        "SK하이닉스 매출액 상위 3개 연도",
    ]
    for example in examples:
        if st.button(example, key=f"ex_{example}"):
            st.session_state["_prefill"] = example
            st.rerun()

    st.divider()
    st.caption("Powered by Qwen 72B + LangGraph + DART API")
