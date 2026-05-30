"""
doctor.py — System health check for Momento (``momento doctor``).

Checks Python version, dependencies, GPU, disk space, ChromaDB, and CLIP model.
"""
import os
import sys
import importlib
import shutil
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

from ..core.logger import get_logger

logger = get_logger(__name__)


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