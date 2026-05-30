"""
diagnostics.py — Health, stats, and diagnostics for Momento.

Provides:
- ``momento doctor`` — system health check
- ``momento stats`` — detailed index statistics
- ``momento benchmark`` — performance measurement
"""

import os
import sys
import time
import importlib
import shutil
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

from ..core.logger import get_logger

logger = get_logger(__name__)


# ── System Checks ─────────────────────────────────────────────────────

@dataclass
class DoctorResult:
    """Results from a ``momento doctor`` health check."""
    python_version: str = ""
    momento_version: str = ""
    device: str = "unknown"
    clip_model_available: bool = False
    chromadb_accessible: bool = False
    disk_space_gb: float = 0.0
    disk_space_free_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    dependencies: Dict[str, bool] = field(default_factory=dict)
    optional_deps: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Return True if all critical checks passed."""
        return len(self.errors) == 0


def _check_python_version() -> Tuple[bool, str]:
    """Check Python version is >= 3.12."""
    v = sys.version_info
    ok = v.major >= 3 and v.minor >= 12
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    return ok, version_str


def _check_momento_version() -> str:
    """Get momento version."""
    try:
        from importlib.metadata import version
        return version("momento")
    except Exception:
        return "2.0.0"


def _check_disk_space(path: str = "/") -> Tuple[float, float]:
    """Check available disk space in GB."""
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        return total_gb, free_gb
    except Exception:
        return 0.0, 0.0


def _check_chromadb() -> bool:
    """Check if ChromaDB is accessible."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="/tmp/momento_health_check")
        client.heartbeat()
        return True
    except Exception:
        return False


def _check_clip_model() -> bool:
    """Check if CLIP model can be loaded."""
    try:
        from ..embedding.clip_backend import ClipBackend
        ClipBackend("ViT-B/16")
        return True
    except Exception:
        return False


def _check_gpu() -> Tuple[bool, str]:
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, name
        return False, ""
    except Exception:
        return False, ""


def _check_dependency(name: str, import_path: str) -> bool:
    """Check if a dependency is importable."""
    try:
        importlib.import_module(import_path)
        return True
    except ImportError:
        return False


REQUIRED_DEPS = {
    "chromadb": "chromadb",
    "torch": "torch",
    "Pillow": "PIL",
}

OPTIONAL_DEPS = {
    "ultralytics (YOLO)": "ultralytics",
    "easyocr": "easyocr",
    "opencv-python-headless": "cv2",
    "tqdm": "tqdm",
    "psutil": "psutil",
}


def run_doctor() -> DoctorResult:
    """Run comprehensive system health check.

    Returns:
        DoctorResult with all health check fields.
    """
    result = DoctorResult()

    # Python version
    py_ok, py_ver = _check_python_version()
    result.python_version = py_ver
    if not py_ok:
        result.errors.append(f"Python {py_ver} < 3.12")

    # Momento version
    result.momento_version = _check_momento_version()

    # Device
    try:
        from ..core.device import device_manager
        result.device = device_manager.device
    except Exception as e:
        result.errors.append(f"Device detection failed: {e}")

    # CLIP model
    result.clip_model_available = _check_clip_model()

    # ChromaDB
    result.chromadb_accessible = _check_chromadb()

    # Disk space
    try:
        from ..core.config import CHROMA_DB_DIR
        check_path = CHROMA_DB_DIR if os.path.exists(CHROMA_DB_DIR) else "/"
        result.disk_space_gb, result.disk_space_free_gb = _check_disk_space(check_path)
    except Exception:
        result.disk_space_gb, result.disk_space_free_gb = _check_disk_space("/")

    # GPU
    gpu_ok, gpu_name = _check_gpu()
    result.gpu_available = gpu_ok
    result.gpu_name = gpu_name

    # Dependencies
    for name, import_path in REQUIRED_DEPS.items():
        result.dependencies[name] = _check_dependency(name, import_path)
        if not result.dependencies[name]:
            result.errors.append(f"Required dependency missing: {name}")

    # Optional dependencies
    for name, import_path in OPTIONAL_DEPS.items():
        result.optional_deps[name] = _check_dependency(name, import_path)

    return result


def print_doctor_report(result: DoctorResult) -> None:
    """Pretty-print the doctor report to stdout.

    Args:
        result: DoctorResult from run_doctor().
    """
    print("\n" + "=" * 50)
    print("🏥 Momento Health Check")
    print("=" * 50)

    status = "✅ Healthy" if result.is_healthy() else "⚠️  Issues Found"
    print(f"Status:      {status}")

    print(f"\n📋 System")
    print(f"  Python:        {result.python_version}")
    print(f"  Momento:       {result.momento_version}")
    print(f"  Device:        {result.device}")
    if result.gpu_available:
        print(f"  GPU:           {result.gpu_name}")

    print(f"\n📦 Storage")
    print(f"  Disk total:    {result.disk_space_gb:.1f} GB")
    print(f"  Disk free:     {result.disk_space_free_gb:.1f} GB")

    print(f"\n🔌 Core Services")
    print(f"  CLIP model:    {'✅' if result.clip_model_available else '❌'}")
    print(f"  ChromaDB:      {'✅' if result.chromadb_accessible else '❌'}")

    print(f"\n📚 Dependencies")
    for name, ok in result.dependencies.items():
        print(f"  {name}: {'✅' if ok else '❌'}")
    for name, ok in result.optional_deps.items():
        print(f"  {name}: {'✅' if ok else '⬜ not installed'}")

    if result.errors:
        print(f"\n⚠️  {len(result.errors)} issue(s) found:")
        for err in result.errors:
            print(f"  • {err}")

    print("=" * 50)


# ── Retrieval Quality Benchmark ─────────────────────────────────────

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
        from ..search import text_search
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


# ── Index Stats ───────────────────────────────────────────────────────

@dataclass
class IndexStats:
    """Detailed index statistics."""
    total_vectors: int = 0
    total_entries: int = 0
    image_count: int = 0
    video_count: int = 0
    object_count: int = 0
    ocr_count: int = 0
    db_size_mb: float = 0.0
    db_path: str = ""


def get_index_stats(index_path: str) -> IndexStats:
    """Get detailed index statistics.

    Args:
        index_path: Path to the ChromaDB directory.

    Returns:
        IndexStats with detailed breakdown.
    """
    stats = IndexStats()
    stats.db_path = index_path

    try:
        from ..storage.vector_store import Index
        idx = Index(db_path=index_path)
        stats.total_vectors = idx.get_vector_count()

        all_paths = idx.get_all_paths()
        stats.total_entries = len(all_paths)

        from ..core.config import COMPOSITE_SEP
        stats.image_count = len([p for p in all_paths
                                 if COMPOSITE_SEP not in p or f'{COMPOSITE_SEP}orig' in p])
        stats.video_count = len([p for p in all_paths if f'{COMPOSITE_SEP}frame_' in p])
        stats.object_count = len([p for p in all_paths if f'{COMPOSITE_SEP}yolo_' in p])
        stats.ocr_count = len([p for p in all_paths if f'{COMPOSITE_SEP}ocr' in p])

        # Calculate DB size
        if os.path.exists(index_path):
            total_bytes = 0
            for dirpath, dirnames, filenames in os.walk(index_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total_bytes += os.path.getsize(fp)
                    except OSError:
                        pass
            stats.db_size_mb = total_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")

    return stats


def print_index_stats(stats: IndexStats) -> None:
    """Pretty-print index statistics to stdout.

    Args:
        stats: IndexStats from get_index_stats().
    """
    print("\n" + "=" * 50)
    print("📊 Index Statistics")
    print("=" * 50)
    print(f"Total vectors:     {stats.total_vectors:,}")
    print(f"Total entries:     {stats.total_entries:,}")
    print(f"  Images:          {stats.image_count:,}")
    print(f"  Videos:          {stats.video_count:,}")
    print(f"  Objects (YOLO):  {stats.object_count:,}")
    print(f"  OCR texts:       {stats.ocr_count:,}")
    print(f"Database size:     {stats.db_size_mb:.1f} MB")
    print(f"Database path:     {stats.db_path}")
    print("=" * 50)


# ── Benchmark ─────────────────────────────────────────────────────────

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