"""Local FAISS Vector Store Manager for SEC Filing RAG.

Provides persistent in-memory dense vector similarity search, metadata mapping,
symbol-based filtering, and disk serialization (index.faiss + metadata.json).
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss

from rag.chunking import FilingChunk, chunk_all_documents
from rag.ingestion import load_all_filings
from rag.embeddings import embed_documents, EMBEDDING_DIM


class FAISSVectorStore:
    """Manages local FAISS index and companion chunk metadata ledger."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        # Use IndexFlatIP (Inner Product) on L2-normalized vectors (= Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[FilingChunk] = []

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def add_chunks(self, chunks: List[FilingChunk], embeddings: np.ndarray) -> None:
        """Adds text chunks and their pre-computed dense embeddings to the index."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} vectors.")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dim {self.dimension}, got {embeddings.shape[1]}")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        filter_symbol: Optional[str] = None
    ) -> List[Tuple[FilingChunk, float]]:
        """Performs cosine similarity search against indexed corporate chunks.
        
        Args:
            query_vector: 1D or 2D array of shape (384,) or (1, 384).
            top_k: Number of highest-similarity chunks to return.
            filter_symbol: Optional stock symbol filter (e.g. 'AAPL', 'NVDA').
            
        Returns:
            List of (FilingChunk, similarity_score) sorted in descending order of similarity.
        """
        if self.total_vectors == 0:
            return []

        # Ensure query is 2D with shape (1, 384)
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # Retrieve a broader candidate pool if filtering by symbol
        fetch_k = min(self.total_vectors, top_k * 5 if filter_symbol else top_k)
        scores, indices = self.index.search(query_vector, fetch_k)

        results: List[Tuple[FilingChunk, float]] = []
        clean_symbol = filter_symbol.strip().upper().lstrip("$") if filter_symbol else None

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
                
            chunk = self.chunks[idx]
            if clean_symbol and chunk.symbol.upper() != clean_symbol:
                continue
                
            results.append((chunk, round(float(score), 4)))
            if len(results) >= top_k:
                break

        return results

    def save(self, directory: str = "data/vector_store") -> None:
        """Serializes FAISS C++ binary index and metadata JSON to disk."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS binary index
        index_file = dir_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save metadata ledger
        metadata_file = dir_path / "metadata.json"
        metadata_payload = [chunk.model_dump() for chunk in self.chunks]
        metadata_file.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str = "data/vector_store") -> "FAISSVectorStore":
        """Loads serialized FAISS index and metadata from disk."""
        dir_path = Path(directory)
        index_file = dir_path / "index.faiss"
        metadata_file = dir_path / "metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Vector store files missing in '{directory}'")

        raw_index = faiss.read_index(str(index_file))
        instance = cls(dimension=raw_index.d)
        instance.index = raw_index

        metadata_raw = json.loads(metadata_file.read_text(encoding="utf-8"))
        instance.chunks = [FilingChunk(**item) for item in metadata_raw]
        return instance


def build_and_persist_vector_store(
    filings_dir: str = "data/filings",
    vector_store_dir: str = "data/vector_store"
) -> FAISSVectorStore:
    """End-to-end builder: Ingests documents, chunks text, embeds, and saves to disk."""
    docs = load_all_filings(filings_dir)
    if not docs:
        raise ValueError(f"No filing documents found in '{filings_dir}'.")

    chunks = chunk_all_documents(docs, chunk_size=800, chunk_overlap=150)
    texts = [c.text for c in chunks]
    embeddings = embed_documents(texts)

    store = FAISSVectorStore(dimension=EMBEDDING_DIM)
    store.add_chunks(chunks, embeddings)
    store.save(vector_store_dir)
    return store
