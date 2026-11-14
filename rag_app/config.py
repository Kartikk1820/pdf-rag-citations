"""Environment-driven configuration.

The app targets an OpenAI-compatible chat endpoint, so choosing a provider is
just a matter of resolving the right ``(api_key, base_url, model)`` triple.
Both OpenRouter and Groq expose OpenAI-compatible APIs, which is what lets a
single client serve either one.

Qdrant runs in *embedded* mode (a local on-disk directory, no server process),
so there is nothing to install or run beyond the Python package itself.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv, set_key

load_dotenv()

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

_PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
    },
}


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str
    base_url: str
    model: str

    qdrant_path: str
    qdrant_collection: str

    embedding_model: str
    embedding_dim: int
    reranker_model: str

    data_dir: str
    storage_dir: str

    retrieval_k: int
    top_k: int
    score_threshold: float
    enable_hybrid: bool
    enable_query_rewrite: bool
    enable_general_fallback: bool
    hybrid_bm25_weight: float


# fastembed model name -> output dimensionality. Keeping this explicit avoids a
# surprise mismatch between the collection size and the vectors we upsert.
_EMBEDDING_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

# Settings a user can change at runtime from the Settings screen.
EDITABLE_KEYS = (
    "LLM_PROVIDER",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "RETRIEVAL_K",
    "TOP_K",
    "SCORE_THRESHOLD",
    "ENABLE_HYBRID",
    "ENABLE_QUERY_REWRITE",
    "ENABLE_GENERAL_FALLBACK",
    "HYBRID_BM25_WEIGHT",
)


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@lru_cache
def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. Expected one of {list(_PROVIDERS)}."
        )

    spec = _PROVIDERS[provider]
    # Deliberately NOT validated here: an empty/placeholder key must not stop
    # the app from starting (ingest, Dashboard, Settings all work without an
    # LLM). It's checked lazily in llm.py, right before the LLM is actually
    # called, via ``is_llm_configured``.
    api_key = os.getenv(spec["key_env"], "").strip()

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()
    if embedding_model not in _EMBEDDING_DIMS:
        raise ValueError(
            f"Unknown EMBEDDING_MODEL={embedding_model!r}. "
            f"Add its dimensionality to config._EMBEDDING_DIMS."
        )

    return Settings(
        provider=provider,
        api_key=api_key,
        base_url=spec["base_url"],
        model=os.getenv(spec["model_env"], spec["default_model"]).strip(),
        qdrant_path=os.getenv("QDRANT_PATH", "./storage/qdrant").strip(),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "pdf_documents").strip(),
        embedding_model=embedding_model,
        embedding_dim=_EMBEDDING_DIMS[embedding_model],
        reranker_model=os.getenv(
            "RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"
        ).strip(),
        data_dir=os.getenv("DATA_DIR", "./data").strip(),
        storage_dir=os.getenv("STORAGE_DIR", "./storage").strip(),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "20")),
        top_k=int(os.getenv("TOP_K", "5")),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.3")),
        enable_hybrid=_bool("ENABLE_HYBRID", "true"),
        enable_query_rewrite=_bool("ENABLE_QUERY_REWRITE", "true"),
        enable_general_fallback=_bool("ENABLE_GENERAL_FALLBACK", "true"),
        hybrid_bm25_weight=float(os.getenv("HYBRID_BM25_WEIGHT", "0.4")),
    )


def is_llm_configured(settings: "Settings | None" = None) -> bool:
    settings = settings or get_settings()
    return bool(settings.api_key) and not settings.api_key.startswith("your_")


def update_settings(values: dict[str, str]) -> None:
    """Persist editable settings to ``.env`` and refresh the in-memory cache."""
    if not _ENV_PATH.exists():
        _ENV_PATH.touch()
    for key, value in values.items():
        if key not in EDITABLE_KEYS:
            continue
        os.environ[key] = value
        set_key(str(_ENV_PATH), key, value, quote_mode="never")
    get_settings.cache_clear()
