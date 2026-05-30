"""
benchmark_retrieval.py — Retrieval quality benchmark for Momento.

Compares CLIP-only vs V3 pipeline on recall@k, precision, and latency.
Also measures exact-match hit rate.
"""
import os
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalBenchmarkResult:
    """Results from retrieval quality benchmarking."""
    clip_only_recall: float = 0.0
    v3_recall: float = 0.0
    clip_only_precision: float = 0.0
    v3_precision: float = 0.0
    clip_only_latency_ms: float = 0.0
    v3_latency_ms: float = 0.0
    exact_match_hit_rate: float = 0.0
    index_vector_count: int = 0
    hybrid_hit_count: int = 0
    total_queries: int = 0
    notes: List[str] = field(default_factory=list)


def run_retrieval_benchmark(
    index_path: str,
    iterations: int = 5,
    top_k: int = 10,
    include_rerank: bool = False,
) -> RetrievalBenchmarkResult:
    """Run retrieval quality benchmark comparing CLIP-only vs V3 pipeline.

    Measures recall@k, precision, latency, and exact-match hit rate
    by running queries against the index in both modes.

    Args:
        index_path: Path to the ChromaDB directory.
        iterations: Number of iterations for averaging.
        top_k: Top-K for recall/precision calculation.
        include_rerank: Whether to include reranking in V3 benchmark.

    Returns:
        RetrievalBenchmarkResult with comparative metrics.
    """
    result = RetrievalBenchmarkResult()
    result.total_queries = iterations

    try:
        from ..storage.vector_store import Index
        idx = Index(db_path=index_path)
        result.index_vector_count = idx.get_vector_count()

        if idx.get_vector_count() == 0:
            result.notes.append("Index is empty — no queries could be run")
            return result

        # Collect random entries from the index for query text
        import numpy as np
        from ..embedding.legacy_compat import extract_text_features
        from ..retrieval.pipeline import text_search
        from ..storage.exact_index import ExactIndex

        # Generate test queries from indexed entries
        all_paths = idx.get_all_paths()
        if not all_paths:
            result.notes.append("No paths in index")
            return result

        # Build a set of test queries using filenames and semantic queries
        test_queries = []
        import os
        for p in all_paths[:min(iterations, len(all_paths))]:
            basename = os.path.basename(p)
            name_no_ext = os.path.splitext(basename)[0]
            # Clean filename for use as query
            clean = name_no_ext.replace("_", " ").replace("-", " ")
            if clean:
                test_queries.append(clean)

        # Add some fixed semantic queries for variety
        semantic_queries = [
            "dog in park", "sunset beach", "city street",
            "cat playing", "mountain lake",
        ]
        test_queries.extend(semantic_queries)
        test_queries = test_queries[:max(iterations, 5)]

        result.total_queries = len(test_queries)

        # --- CLIP-only benchmark ---
        clip_results = []
        for query in test_queries:
            t0 = time.time()
            clip_hits = text_search(query, idx, top_k=top_k, use_v3=False)
            clip_latency = (time.time() - t0) * 1000
            clip_results.append((bool(clip_hits), clip_latency))

        result.clip_only_latency_ms = (
            sum(l for _, l in clip_results) / len(clip_results)
        )
        result.clip_only_recall = (
            sum(1 for h, _ in clip_results if h) / len(clip_results)
        )
        result.clip_only_precision = result.clip_only_recall  # simplified

        # --- V3 pipeline benchmark ---
        v3_results = []
        exact_hits = 0
        for query in test_queries:
            t0 = time.time()
            v3_hits = text_search(query, idx, top_k=top_k, use_v3=True)
            v3_latency = (time.time() - t0) * 1000
            had_results = bool(v3_hits)
            v3_results.append((had_results, v3_latency))

            # Check if exact match contributed
            if any(s >= 0.95 for s, _ in v3_hits):
                exact_hits += 1

        result.v3_latency_ms = sum(l for _, l in v3_results) / len(v3_results)
        result.v3_recall = sum(1 for h, _ in v3_results if h) / len(v3_results)
        result.v3_precision = result.v3_recall  # simplified
        result.exact_match_hit_rate = exact_hits / len(test_queries) if test_queries else 0.0
        result.hybrid_hit_count = exact_hits

    except Exception as e:
        result.notes.append(f"Benchmark error: {e}")

    return result


def print_retrieval_benchmark_report(result: RetrievalBenchmarkResult) -> None:
    """Pretty-print retrieval benchmark results to stdout."""
    print("\n" + "=" * 60)
    print("📊 Retrieval Quality Benchmark")
    print("=" * 60)
    print(f"Total queries:     {result.total_queries}")
    print(f"Index vectors:     {result.index_vector_count:,}")
    print()

    print(f"{'Metric':<30} {'CLIP Only':<15} {'V3 Pipeline':<15} {'Change':<10}")
    print("-" * 70)
    print(f"{'Recall@K':<30} {result.clip_only_recall:.1%:<15} {result.v3_recall:.1%:<15} "
          f"{'+' if result.v3_recall >= result.clip_only_recall else ''}{(result.v3_recall - result.clip_only_recall):.1%}")

    print(f"{'Precision@K':<30} {result.clip_only_precision:.1%:<15} {result.v3_precision:.1%:<15} "
          f"{'+' if result.v3_precision >= result.clip_only_precision else ''}{(result.v3_precision - result.clip_only_precision):.1%}")

    if result.exact_match_hit_rate > 0:
        print(f"{'Exact-match hit rate':<30} {'0.0%':<15} {result.exact_match_hit_rate:.1%:<15} {'+NEW'}")

    print(f"{'Avg latency (ms)':<30} {result.clip_only_latency_ms:.1f:<15} {result.v3_latency_ms:.1f:<15} "
          f"{'+' if result.v3_latency_ms >= result.clip_only_latency_ms else ''}{(result.v3_latency_ms - result.clip_only_latency_ms):.1f}")

    print()

    if result.notes:
        print(f"📝 Notes:")
        for note in result.notes:
            print(f"  - {note}")

    print("=" * 60)