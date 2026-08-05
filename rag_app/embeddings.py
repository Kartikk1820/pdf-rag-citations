"""Local, free embeddings via fastembed (ONNX, no PyTorch).

fastembed is Qdrant's official embedding library. The model runs fully offline
after a one-time download, so there is no per-call cost and nothing to rate
limit. bge-small is trained with an asymmetric query instruction; ``query_embed``
applies that instruction automatically, while ``embed`` treats text as a passage.
"""

from functools import lru_cache

from fastembed import TextEmbedding

from rag_app.config import get_settings


@lru_cache
def _model() -> TextEmbedding:
    settings = get_settings()
    return TextEmbedding(model_name=settings.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    vec = next(_model().query_embed(text))
    return vec.tolist()
