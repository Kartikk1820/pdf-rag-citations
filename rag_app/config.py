"""Environment-driven configuration.

The app targets an OpenAI-compatible chat endpoint, so choosing a provider is
just a matter of resolving the right ``(api_key, base_url, model)`` triple.
Both OpenRouter and Groq expose OpenAI-compatible APIs, which is what lets a
single client serve either one.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

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
        "default_model": "llama-3.3-70b-versatile",
    },
}


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str
    base_url: str
    model: str

    qdrant_url: str
    qdrant_collection: str

    embedding_model: str
    embedding_dim: int

    top_k: int
    score_threshold: float


# fastembed model name -> output dimensionality. Keeping this explicit avoids a
# surprise mismatch between the collection size and the vectors we upsert.
_EMBEDDING_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


@lru_cache
def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. Expected one of {list(_PROVIDERS)}."
        )

    spec = _PROVIDERS[provider]
    api_key = os.getenv(spec["key_env"], "").strip()
    if not api_key or api_key.startswith("your_"):
        raise ValueError(
            f"Missing {spec['key_env']} for provider {provider!r}. "
            f"Set it in your .env file."
        )

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
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "pdf_documents").strip(),
        embedding_model=embedding_model,
        embedding_dim=_EMBEDDING_DIMS[embedding_model],
        top_k=int(os.getenv("TOP_K", "5")),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.3")),
    )
