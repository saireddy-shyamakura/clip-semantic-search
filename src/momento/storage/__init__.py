"""
storage — Data persistence layer for Momento.

Provides separate backends for vector embeddings, metadata,
exact search index, and embedding cache.
"""
from .vector_store import Index
from .metadata_store import MetadataStore
from .exact_index import ExactIndex
from .cache import load_embedding, save_embedding, clear_cache
from .manager import StorageManager

__all__ = [
    "Index",
    "MetadataStore",
    "ExactIndex",
    "load_embedding", "save_embedding", "clear_cache",
    "StorageManager",
]