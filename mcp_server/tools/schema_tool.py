import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rag.embedder import search_schema


async def get_database_schema_tool(keyword: str, top_k: int = 3) -> dict:
    """
    ChromaDB에서 keyword 관련 테이블 DDL 및 컬럼 명세 반환.

    Args:
        keyword: 검색 키워드 (예: "영업이익", "기업코드", "재무제표")
        top_k:   반환할 최대 결과 수

    Returns:
        {"status": "success", "schema_context": "...", "keyword": "..."}
    """
    try:
        schema_context = search_schema(keyword, top_k=top_k)
        return {
            "status": "success",
            "schema_context": schema_context,
            "keyword": keyword,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "keyword": keyword,
        }
