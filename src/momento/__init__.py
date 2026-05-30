"""
momento — Multi-Modal Semantic Search Engine.

Powered by OpenAI CLIP and ChromaDB with support for:
- Image search (text-to-image and image-to-image)
- Video keyframe indexing
- YOLO object detection for fine-grained search
- OCR text extraction for text-in-image search
- Multi-embedding via image augmentation for better recall
- V3 hybrid search with reranking and cross-modal fusion
"""

__version__ = "3.0.0"

# Core
from momento.core.config import MomentoConfig, load_config, save_config
from momento.core.device import DeviceManager, device_manager

# Storage
from momento.storage.vector_store import Index
from momento.storage.cache import load_embedding, save_embedding, clear_cache
from momento.storage.exact_index import ExactIndex
from momento.storage.metadata_store import MetadataStore

# Embedding
from momento.embedding import EmbeddingBackend, ClipBackend

# Search (V3 pipeline)
from momento.retrieval.pipeline import image_search, text_search, search_hybrid

# CLI
from momento.cli import main, AppController

__all__ = [
    "MomentoConfig", "load_config", "save_config",
    "DeviceManager", "device_manager",
    "Index",
    "load_embedding", "save_embedding", "clear_cache",
    "ExactIndex",
    "MetadataStore",
    "EmbeddingBackend", "ClipBackend",
    "image_search", "text_search", "search_hybrid",
    "main", "AppController",
]