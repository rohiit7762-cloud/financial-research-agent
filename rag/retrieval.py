"""Retrieval Engine for SEC Filing RAG.

Coordinates query embedding, dense vector similarity search, symbol filtering,
and formatting structured RetrievedFilingResult models with provenance citations.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from rag.embeddings import embed_text
from rag.vector_store import FAISSVectorStore, build_and_persist_vector_store

_CACHED_STORE: Optional[FAISSVectorStore] = None


class RetrievedFilingResult(BaseModel):
    """Schema representing an individual retrieved filing passage."""
    chunk_id: str = Field(description="Unique chunk citation ID, e.g. NVDA-2024-CHK-002")
    symbol: str = Field(description="Associated stock ticker")
    company_name: str = Field(description="Corporate entity name")
    section: str = Field(description="Filing section title, e.g. Item 1A. Risk Factors")
    similarity_score: float = Field(description="Cosine similarity score (0.0 to 1.0)")
    source_url: str = Field(description="Official SEC EDGAR reference link")
    excerpt: str = Field(description="Exact retrieved narrative text snippet")


class FilingSearchResponse(BaseModel):
    """Schema for complete filing RAG search output."""
    status: str = Field(default="success", description="Status indicator")
    query: str = Field(description="Natural language query submitted")
    symbol_filter: Optional[str] = Field(default=None, description="Applied ticker filter")
    result_count: int = Field(description="Number of relevant chunks found")
    results: List[RetrievedFilingResult] = Field(default_factory=list, description="Ranked list of matching passages")
    timestamp: str = Field(description="ISO timestamp of retrieval")
    source: str = Field(default="SEC 10-K Filing Vector Index", description="Knowledge base source")


def get_vector_store(
    store_dir: str = "data/vector_store",
    filings_dir: str = "data/filings"
) -> FAISSVectorStore:
    """Returns a cached FAISSVectorStore, building it automatically if not yet initialized."""
    global _CACHED_STORE
    if _CACHED_STORE is not None:
        return _CACHED_STORE

    store_path = Path(store_dir)
    if (store_path / "index.faiss").exists() and (store_path / "metadata.json").exists():
        _CACHED_STORE = FAISSVectorStore.load(store_dir)
    else:
        _CACHED_STORE = build_and_persist_vector_store(filings_dir, store_dir)
        
    return _CACHED_STORE


def retrieve_filing_chunks(
    query: str,
    symbol: Optional[str] = None,
    top_k: int = 3,
    store_dir: str = "data/vector_store"
) -> FilingSearchResponse:
    """Performs semantic similarity search against indexed corporate 10-K filings.
    
    Args:
        query: Conceptual risk or disclosure question (e.g. 'export restrictions to China').
        symbol: Optional ticker to restrict results to (e.g. 'NVDA', 'AAPL').
        top_k: Maximum number of relevant chunks to retrieve (default 3).
        store_dir: Directory of serialized vector index.
        
    Returns:
        FilingSearchResponse containing ranked passages and metadata.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    clean_query = query.strip()
    clean_symbol = symbol.strip().upper().lstrip("$") if symbol else None

    try:
        store = get_vector_store(store_dir)
        
        # Check if requested symbol exists in our indexed corpus
        indexed_symbols = {chunk.symbol.upper() for chunk in store.chunks}
        if clean_symbol and clean_symbol not in indexed_symbols:
            return FilingSearchResponse(
                status="success",
                query=clean_query,
                symbol_filter=clean_symbol,
                result_count=0,
                results=[],
                timestamp=now_str,
                source=f"SEC 10-K Vector Store (Symbol '{clean_symbol}' not in curated index. Indexed: {', '.join(sorted(indexed_symbols))})"
            )

        query_vec = embed_text(clean_query)
        raw_matches = store.search(query_vec, top_k=top_k, filter_symbol=clean_symbol)

        retrieved: List[RetrievedFilingResult] = []
        for chunk, score in raw_matches:
            retrieved.append(
                RetrievedFilingResult(
                    chunk_id=chunk.chunk_id,
                    symbol=chunk.symbol,
                    company_name=chunk.company_name,
                    section=chunk.section,
                    similarity_score=score,
                    source_url=chunk.source_url,
                    excerpt=chunk.text
                )
            )

        return FilingSearchResponse(
            status="success",
            query=clean_query,
            symbol_filter=clean_symbol,
            result_count=len(retrieved),
            results=retrieved,
            timestamp=now_str,
            source="SEC 10-K Filing Vector Index"
        )

    except Exception as e:
        return FilingSearchResponse(
            status="error",
            query=clean_query,
            symbol_filter=clean_symbol,
            result_count=0,
            results=[],
            timestamp=now_str,
            source=f"Error executing vector retrieval: {str(e)}"
        )
