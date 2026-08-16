"""bge-small embedding helpers, shared by indexing (passages) and retrieval (queries).

Two things that are easy to get wrong and expensive to debug later:

1. bge is an *asymmetric* model. Passages are embedded bare; queries get the
   instruction prefix from config ("Represent this sentence for searching relevant
   passages: "). Embedding a query without the prefix costs real recall, and the
   failure is silent — scores just get worse. Index-time and query-time paths are
   therefore separate functions, not one function with a flag someone forgets.
2. Vectors are L2-normalized at encode time, so Qdrant's COSINE distance and a raw
   dot product agree, and reranking/dedup code can use cosine without re-normalizing.

The model is a module-level singleton: loading it takes seconds, encoding a batch
takes milliseconds.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from src.common.config import CFG

_MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(CFG["embedding"]["model"])
    return _MODEL


def embed_passages(texts: list[str], batch_size: int | None = None,
                   show_progress: bool = False) -> np.ndarray:
    """Encode chunk texts for indexing. Returns (n, dim) float32, L2-normalized."""
    if not texts:
        return np.zeros((0, CFG["embedding"]["dim"]), dtype="float32")
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size or CFG["embedding"].get("batch_size", 32),
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    ).astype("float32")


def embed_query(query: str) -> list[float]:
    """Encode a search query (with the bge instruction prefix) as a plain list."""
    vec = get_model().encode(
        CFG["embedding"]["query_prefix"] + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.astype("float32").tolist()
