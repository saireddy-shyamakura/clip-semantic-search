"""
stats.py — Index statistics for Momento (``momento stats``).

Provides detailed breakdown of indexed content by type (images, videos,
objects, OCR) and database size.
"""
import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from ..core.logger import get_logger

logger = get_logger(__name__)


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