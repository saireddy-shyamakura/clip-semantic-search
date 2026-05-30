"""
benchmark_perf.py — Performance benchmarks for Momento (``momento benchmark``).

Measures embedding extraction latency, search latency, and batch throughput.
"""
import os
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from performance benchmarking."""
    embedding_extraction_ms: float = 0.0
    search_latency_ms: float = 0.0
    batch_throughput_items_per_sec: float = 0.0
    index_vector_count: int = 0
    notes: List[str] = field(default_factory=list)


def run_benchmark(index_path: str, iterations: int = 5) -> BenchmarkResult:
    """Run performance benchmarks.

    Tests:
    - Embedding extraction latency (single image)
    - Search latency (top-10 query)
    - Batch throughput estimate

    Args:
        index_path: Path to the ChromaDB directory.
        iterations: Number of iterations for averaging.

    Returns:
        BenchmarkResult with timing data.
    """
    result = BenchmarkResult()
    import numpy as np

    try:
        from ..storage.vector_store import Index
        idx = Index(db_path=index_path)
        result.index_vector_count = idx.get_vector_count()

        # Embedding extraction benchmark (if there are images to test)
        from ..embedding.legacy_compat import extract_text_features
        timings = []
        for _ in range(iterations):
            t0 = time.time()
            extract_text_features("a photo of a cat")
            timings.append((time.time() - t0) * 1000)
        result.embedding_extraction_ms = sum(timings) / len(timings)

        # Search latency benchmark (only if index is non-empty)
        if idx.get_vector_count() > 0:
            query_vec = np.random.randn(512).astype(np.float32)
            query_vec /= np.linalg.norm(query_vec)

            timings = []
            for _ in range(iterations):
                t0 = time.time()
                idx.search(query_vec, top_k=10)
                timings.append((time.time() - t0) * 1000)
            result.search_latency_ms = sum(timings) / len(timings)

        # Batch throughput (simulated)
        if idx.get_vector_count() > 0:
            batch_sizes = [1, 10, 50]
            batch_times = []
            for bs in batch_sizes:
                t0 = time.time()
                for _ in range(bs):
                    extract_text_features("a photo of a dog")
                batch_times.append((time.time() - t0) / bs)
            avg_time = sum(batch_times) / len(batch_times)
            result.batch_throughput_items_per_sec = 1.0 / avg_time if avg_time > 0 else 0

    except Exception as e:
        result.notes.append(f"Benchmark error: {e}")

    return result


def print_benchmark_report(result: BenchmarkResult) -> None:
    """Pretty-print benchmark results to stdout.

    Args:
        result: BenchmarkResult from run_benchmark().
    """
    print("\n" + "=" * 50)
    print("⚡ Performance Benchmark")
    print("=" * 50)
    print(f"Index size:        {result.index_vector_count:,} vectors")
    print(f"Embedding latency: {result.embedding_extraction_ms:.1f} ms")
    print(f"Search latency:    {result.search_latency_ms:.1f} ms")
    if result.batch_throughput_items_per_sec > 0:
        print(f"Batch throughput:  {result.batch_throughput_items_per_sec:.1f} items/sec")

    if result.notes:
        print(f"\n📝 Notes:")
        for note in result.notes:
            print(f"  - {note}")

    print("=" * 50)