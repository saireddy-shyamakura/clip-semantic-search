"""
search — DEPRECATED: Search facade for backward compatibility.

All search logic has moved to ``retrieval/pipeline.py``.
This module re-exports the public API for backward compatibility.
New code should import from ``retrieval.pipeline`` directly.
"""
from typing import List, Tuple

from ..retrieval.pipeline import (
    image_search,
    text_search,
    search_hybrid,
    register_paths_for_exact_search,
)
from ..storage.exact_index import ExactIndex

__all__ = [
    "ExactIndex",
    "image_search",
    "text_search",
    "search_hybrid",
    "register_paths_for_exact_search",
]