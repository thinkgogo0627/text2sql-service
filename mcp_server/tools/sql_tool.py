import os
import re
from typing import Any

import sqlalchemy
from sqlalchemy import create_engine, text

# SELECT 외 허용되지 않는 키워드 목록
BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|CREATE|ALTER|GRANT|REVOKE|EXEC|EXECUTE|COPY|MERGE)\b",
    re.IGNORECASE,
)


def _get_engine():
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "text2sql_db")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def _is_safe_select(sql: str) -> bool:
    """SELECT 외 DDL/DML 키워드가 없는지 검사."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return False
    if BLOCKED_KEYWORDS.search(stripped):
        return False
    return True


async def execute_financial_sql_tool(sql_query: str) -> dict:
    """
    SELECT 쿼리만 허용하여 PostgreSQL에서 실행.

    Args:
        sql_query: 실행할 SQL 문자열

    Returns:
        성공: {"status": "success", "row_count": N, "data": [...]}
        실패: {"status": "error", "error_type": "...", "error_message": "...", "hint": "..."}
    """
    if not _is_safe_select(sql_query):
        return {
            "status": "error",
            "error_type": "SecurityViolation",
            "error_message": "SELECT 쿼리만 허용됩니다. DDL/DML은 실행할 수 없습니다.",
            "hint": "쿼리가 SELECT로 시작하고 INSERT/UPDATE/DELETE/DROP 등이 없는지 확인하세요.",
        }

    try:
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            columns = list(result.keys())
            rows = result.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
        return {
            "status": "success",
            "row_count": len(data),
            "data": data,
        }
    except sqlalchemy.exc.ProgrammingError as e:
        return {
            "status": "error",
            "error_type": "ProgrammingError",
            "error_message": str(e.orig),
            "hint": "SQL 문법 또는 테이블/컬럼명을 확인하세요.",
        }
    except sqlalchemy.exc.OperationalError as e:
        return {
            "status": "error",
            "error_type": "OperationalError",
            "error_message": str(e.orig),
            "hint": "DB 연결 상태를 확인하세요.",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "hint": "예상치 못한 오류가 발생했습니다.",
        }
