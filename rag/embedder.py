import os
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

METADATA_PATH = Path(__file__).parent / "metadata" / "schema_metadata.json"
COLLECTION_NAME = "schema_metadata"

# 문서 청크 임베딩 모델 (pplx-embed-context)
DOC_MODEL_NAME = "perplexity-ai/pplx-embed-context-v1-0.6b"
# 쿼리 임베딩 모델 (pplx-embed)
QUERY_MODEL_NAME = "perplexity-ai/pplx-embed-v1-0.6b"


def _get_chroma_client() -> chromadb.HttpClient:
    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8002"))
    return chromadb.HttpClient(host=host, port=port)


def _load_doc_model() -> SentenceTransformer:
    return SentenceTransformer(DOC_MODEL_NAME)


def _load_query_model() -> SentenceTransformer:
    return SentenceTransformer(QUERY_MODEL_NAME)


def embed_schema_metadata():
    """
    schema_metadata.json을 읽어 ChromaDB에 임베딩 적재.
    이미 존재하는 컬렉션은 초기화 후 재적재.
    """
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    client = _get_chroma_client()

    # 기존 컬렉션 삭제 후 재생성
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    doc_model = _load_doc_model()

    documents = []
    metadatas = []
    ids = []

    for i, item in enumerate(metadata):
        # DDL + description + join_hint를 결합하여 임베딩 텍스트 구성
        doc_text = (
            f"Table: {item['table_name']}\n"
            f"Description: {item.get('description', '')}\n"
            f"DDL: {item.get('ddl', '')}\n"
            f"Join hint: {item.get('join_hint', '')}\n"
            f"Key columns: {', '.join(item.get('key_columns', []))}"
        )
        documents.append(doc_text)
        metadatas.append({
            "table_name": item["table_name"],
            "description": item.get("description", ""),
            "join_hint": item.get("join_hint", ""),
        })
        ids.append(f"schema_{i}_{item['table_name']}")

    embeddings = doc_model.encode(documents, show_progress_bar=True).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"Embedded {len(documents)} schema entries into ChromaDB.")


def search_schema(keyword: str, top_k: int = 3) -> str:
    """
    ChromaDB에서 keyword와 유사한 스키마 DDL을 검색한다.

    Returns:
        관련 DDL 텍스트를 합친 문자열
    """
    client = _get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    query_model = _load_query_model()
    query_embedding = query_model.encode([keyword]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs)
