"""Comprehensive Unit & Integration Tests for SEC Filing RAG Subsystem.

Tests:
1. Document ingestion and metadata parsing.
2. Recursive chunking with overlap.
3. FAISS index construction, disk persistence, and reload.
4. Semantic retrieval precision (verifies correct company retrieved for distinct risk topics).
5. Symbol-based filtering.
6. Graceful containment for unindexed companies.
7. LangChain @tool execution.
"""

import json
from pathlib import Path
import pytest

from rag.ingestion import load_all_filings, FilingDocument
from rag.chunking import chunk_all_documents, FilingChunk
from rag.vector_store import build_and_persist_vector_store, FAISSVectorStore
from rag.retrieval import retrieve_filing_chunks, FilingSearchResponse
from tools.filing_search import search_filing_rag


@pytest.fixture(scope="module")
def initialized_vector_store():
    """Builds and serializes vector store once for test suite."""
    store = build_and_persist_vector_store(
        filings_dir="data/filings",
        vector_store_dir="data/vector_store"
    )
    return store


def test_document_ingestion():
    """Verify that all 4 curated benchmark filings are discovered and parsed."""
    docs = load_all_filings("data/filings")
    assert len(docs) >= 4
    symbols = {d.symbol for d in docs}
    assert {"AAPL", "MSFT", "NVDA", "TSLA"}.issubset(symbols)
    
    for doc in docs:
        assert doc.clean_text and len(doc.clean_text) > 200
        assert doc.source_url.startswith("http")
        assert doc.section == "Item 1A. Risk Factors"


def test_chunking_with_overlap():
    """Verify recursive character splitting and metadata continuity."""
    docs = load_all_filings("data/filings")
    chunks = chunk_all_documents(docs, chunk_size=800, chunk_overlap=150)
    
    assert len(chunks) > len(docs)
    for chunk in chunks:
        assert isinstance(chunk, FilingChunk)
        assert len(chunk.text) <= 900
        assert chunk.chunk_id.startswith(chunk.symbol)
        assert chunk.source_url.startswith("http")


def test_vector_store_persistence(initialized_vector_store):
    """Verify that FAISS index and metadata serialize to and reload from disk."""
    assert initialized_vector_store.total_vectors > 0
    
    # Reload from disk
    reloaded_store = FAISSVectorStore.load("data/vector_store")
    assert reloaded_store.total_vectors == initialized_vector_store.total_vectors
    assert len(reloaded_store.chunks) == len(initialized_vector_store.chunks)


def test_semantic_retrieval_nvidia_export_controls(initialized_vector_store):
    """Verify that a query about export curbs accurately surfaces NVIDIA risk factors."""
    response = retrieve_filing_chunks(
        query="US government export controls on high performance GPU chips to China",
        top_k=2
    )
    
    assert isinstance(response, FilingSearchResponse)
    assert response.status == "success"
    assert response.result_count > 0
    top_hit = response.results[0]
    assert top_hit.symbol == "NVDA"
    assert "China" in top_hit.excerpt or "export" in top_hit.excerpt.lower()
    assert top_hit.similarity_score > 0.40


def test_semantic_retrieval_apple_antitrust(initialized_vector_store):
    """Verify that a query about App Store litigation surfaces Apple risk factors."""
    response = retrieve_filing_chunks(
        query="App Store antitrust European Union Digital Markets Act gatekeeper",
        top_k=2
    )
    
    assert response.status == "success"
    assert response.result_count > 0
    top_hit = response.results[0]
    assert top_hit.symbol == "AAPL"
    assert "App Store" in top_hit.excerpt or "Digital Markets Act" in top_hit.excerpt


def test_semantic_retrieval_microsoft_openai_dependency(initialized_vector_store):
    """Verify that a query about AI partnerships surfaces Microsoft risk factors."""
    response = retrieve_filing_chunks(
        query="OpenAI partnership dependence and model safety risks",
        top_k=2
    )
    
    assert response.status == "success"
    assert response.result_count > 0
    top_hit = response.results[0]
    assert top_hit.symbol == "MSFT"
    assert "OpenAI" in top_hit.excerpt


def test_symbol_filtering(initialized_vector_store):
    """Verify that specifying a symbol filter strictly restricts retrieved chunks."""
    response = retrieve_filing_chunks(
        query="supply chain concentration and component vendor disruptions",
        symbol="AAPL",
        top_k=3
    )
    
    assert response.status == "success"
    assert response.result_count > 0
    for result in response.results:
        assert result.symbol == "AAPL"


def test_unindexed_symbol_graceful_handling(initialized_vector_store):
    """Verify clean response when user queries a company not yet in the index."""
    response = retrieve_filing_chunks(
        query="advertising revenue and search antitrust",
        symbol="GOOGL",
        top_k=3
    )
    
    assert response.status == "success"
    assert response.result_count == 0
    assert "not in curated index" in response.source


def test_langchain_filing_rag_tool_invocation(initialized_vector_store):
    """Verify invocation through the LangChain @tool interface."""
    raw_json = search_filing_rag.invoke({
        "query": "manufacturing partner TSMC single foundry dependency",
        "symbol": "NVDA"
    })
    assert isinstance(raw_json, str)
    
    payload = json.loads(raw_json)
    assert payload["status"] == "success"
    assert payload["result_count"] > 0
    assert payload["results"][0]["symbol"] == "NVDA"


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_rag.py"])
