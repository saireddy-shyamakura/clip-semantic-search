"""
retrieval — Multi-stage retrieval pipeline for Momento V3.

Architecture:
    Query → Classification → ExactIndex → QueryExpansion → Recall → (optional) Rerank → Fusion

Exposed from ``retrieval.pipeline`` for backward compatibility.
"""
from .recall import recall_search
from .rerank import rerank_results
from .fusion import fuse_scores, FusionWeights
from .router import QueryRouter, QueryType
from .query_expansion import expand_query
from .pipeline import image_search, text_search, search_hybrid, register_paths_for_exact_search

__all__ = [
    "recall_search",
    "rerank_results",
    "fuse_scores", "FusionWeights",
    "QueryRouter", "QueryType",
    "expand_query",
    "image_search",
    "text_search",
    "search_hybrid",
    "register_paths_for_exact_search",
]
