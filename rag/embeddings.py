"""Dense Embeddings Generator for SEC Filing RAG.

Wraps Hugging Face sentence-transformers/all-MiniLM-L6-v2 to generate 384-dimensional
dense vector embeddings on local CPU with L2 normalization.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_MODEL_INSTANCE: Optional[SentenceTransformer] = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Returns a cached singleton instance of the SentenceTransformer model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = SentenceTransformer(model_name)
    return _MODEL_INSTANCE


def embed_text(text: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Generates a single L2-normalized 1D vector embedding for an input string.
    
    Args:
        text: Query or sentence to embed.
        model_name: Model identifier.
        
    Returns:
        np.ndarray of shape (384,) and dtype float32.
    """
    model = get_embedding_model(model_name)
    vector = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vector.astype(np.float32)


def embed_documents(texts: List[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Generates batch L2-normalized 2D vector embeddings for multiple text passages.
    
    Args:
        texts: List of chunk texts to embed.
        model_name: Model identifier.
        
    Returns:
        np.ndarray of shape (N, 384) and dtype float32.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        
    model = get_embedding_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vectors.astype(np.float32)
