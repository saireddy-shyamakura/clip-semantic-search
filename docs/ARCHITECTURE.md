# Architecture Guide

## Overview

Momento is a **local-first, multimodal retrieval engine** built on a staged pipeline architecture. This document describes the system design, component responsibilities, and data flow.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          CLI Layer                               │
│  momento --dir PATH  │  momento doctor  │  momento benchmark     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     AppController                                │
│  Orchestrates indexing workflow, search interface, lifecycle     │
└──────┬────────────────────────────────────────────────┬─────────┘
       │                                                │
┌──────▼──────────┐                           ┌────────▼─────────┐
│  Index Pipeline  │                           │ Retrieval Pipeline│
│                  │                           │                   │
│  ┌─────┐ ┌────┐ │                           │  ┌──────────┐   │
│  │Img  │ │Vid │ │                           │  │ExactIndex│   │
│  │     │ │    │ │                           │  │ (FTS5)   │   │
│  └──┬──┘ └──┬─┘ │                           │  └────┬─────┘   │
│  ┌──▼──┐ ┌──▼─┐ │                           │  ┌────▼─────┐   │
│  │YOLO │ │OCR │ │                           │  │   V3     │   │
│  │     │ │    │ │                           │  │ Pipeline  │   │
│  └─────┘ └────┘ │                           │  │ Expand    │   │
│                  │                           │  │ Recall    │   │
│  Checkpoint      │                           │  │ Rerank    │   │
│  Manager         │                           │  │ Fusion    │   │
└──────────────────┘                           │  │ Router    │   │
                                                │  └──────────┘   │
                                                └──────────────────┘
```

---

## Layer Responsibilities

### 1. CLI Layer (`cli.py`)

- Parses command-line arguments
- Handles subcommands (doctor, stats, benchmark, config, storage)
- Acquires process lock
- Delegates to AppController

### 2. Controller Layer (`app_controller.py`)

- Indexing workflow orchestration
- Search interface lifecycle
- Utility flag handling (--reset, --verify, --count)

### 3. Index Pipeline

#### 3.1 Indexer (`indexer.py`)
- Orchestrates all indexing features
- Manages checkpoint/resume
- Runs independent features in parallel via `ThreadPoolExecutor`

#### 3.2 Image Indexing (`add_images.py`, `ingest.py`)
- Reads images with feature extraction
- Optional multi-embedding augmentation
- Batch processing with cache

#### 3.3 Video Indexing (`video.py`)
- Keyframe extraction (interval or scene-change detection)
- Frame-to-embedding conversion

#### 3.4 YOLO Detection (`yolo.py`)
- Object detection via Ultralytics YOLO
- Crop-and-embed each detected object
- Returns `Detection` dataclass with metadata

#### 3.5 OCR Extraction (`ocr.py`)
- Text extraction via EasyOCR
- Text-to-embedding via CLIP text encoder

### 4. Storage Layer

| Component | File | Technology | Data |
|-----------|------|------------|------|
| Vector Store | `index.py` | ChromaDB | Embeddings + metadata |
| Metadata Store | `storage/metadata_store.py` | SQLite | File attributes, OCR, objects |
| Exact Index | `search/exact_index.py` | SQLite FTS5 | Filename/path tokens |
| Embedding Cache | `cache.py` | Disk (hash-based) | Precomputed embeddings |

### 5. Embedding Layer

| Component | File | Role |
|-----------|------|------|
| Abstract Interface | `embedding/base.py` | `EmbeddingBackend` ABC |
| CLIP Backend | `embedding/clip_backend.py` | OpenAI CLIP implementation |

### 6. Retrieval Layer

```
Query Input
    │
    ▼
┌──────────────┐
│  QueryRouter │  Classifies: EXACT / HYBRID / SEMANTIC
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  ExactIndex      │  If EXACT or HYBRID, check FTS5 first
│  (SQLite FTS5)   │  Score >= 0.95 → return immediately
└──────────────────┘
       │ (no exact match or SEMANTIC query)
       ▼
┌──────────────────┐
│  QueryExpansion  │  Generate synonym variants
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│  Recall      │  CLIP vector search (top-K * 3)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Reranker    │  Optional: cross-encoder scoring
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Fusion      │  Combine embedding + YOLO + OCR scores
└──────┬───────┘
       │
       ▼
   Final Results
```

---

## Data Flow: Indexing

```
File Discovered
    │
    ▼
validate_image_path() / validate_folder_path()
    │
    ▼
Indexer.index_all_features()
    │
    ├── Step 1: Images (serial)
    │   ├── extract_image_features() → CLIP embedding
    │   ├── [if multi-embed] generate_augmentations() → 5 variants
    │   └── index.add() → ChromaDB
    │
    ├── Step 2a: Videos (parallel)
    │   ├── extract_keyframes() → PIL Images
    │   └── embed + index frames
    │
    ├── Step 2b: YOLO (parallel)
    │   ├── detect_objects() → Detection[]
    │   └── embed each crop + index
    │
    └── Step 2c: OCR (parallel)
        ├── extract_text() → string
        └── embed_text() + index
            │
            ▼
    Checkpoint saved after each step
```

---

## Data Flow: Search

```
text_search("dog in park")
    │
    ▼
validate_text_query()
    │
    ▼
_text_should_be_prefixed("dog in park") → False (3+ words)
    │
    ▼
_v3_search_pipeline(query_vector, "dog in park", index)
    │
    ├── router.classify("dog in park") → SEMANTIC
    │
    ├── ENABLE_HYBRID_SEARCH? → False for SEMANTIC
    │
    ├── ENABLE_QUERY_EXPANSION? → expand_query("dog in park")
    │   ├── ["dog in park", "puppy in park", "canine in park"]
    │
    ├── For each variant:
    │   └── recall_search(variant_embedding, index, top_k=10)
    │       ├── recall_k = min(30, index.count)
    │       └── deduplicate by entry_id
    │
    ├── ENABLE_RERANK? → rerank_results(query, candidates)
    │
    └── fuse_scores(embedding_scores)
        ├── score = 0.6*embedding + 0.2*object + 0.2*OCR
        └── sort descending, apply threshold
            │
            ▼
        Return [(score, path), ...]
```

---

## Checkpoint System

The checkpoint system (`checkpoint.py`) tracks indexing progress at the feature level:

```
IndexingCheckpoint
├── folder: str                    # Source folder path
├── collection_id: str             # ChromaDB collection ID
├── status: str                    # "in_progress" | "completed"
├── features_status: {
│     "images": FeatureCheckpoint  # {status, count, processed_files}
│     "multi_embed": FeatureCheckpoint
│     "videos": FeatureCheckpoint
│     "yolo": FeatureCheckpoint
│     "ocr": FeatureCheckpoint
│   }
└── config_snapshot: dict          # Config at time of checkpoint
```

On resume:
1. Completed features are skipped
2. In-progress features restart from last processed file
3. Failed features are re-attempted

---

## Storage Layout

```
~/.local/share/momento/
├── chroma_db/
│   ├── chroma.sqlite3           # Vector embeddings
│   └── <uuid>/
│       ├── data_level0.bin
│       ├── header.bin
│       ├── length.bin
│       └── link_lists.bin
├── embedding_cache/             # Hash-based emb cache
│   └── *.npz
├── logs/
│   └── momento.log              # Structured log output
└── indexing_checkpoint.json     # Crash recovery state
```

---

## Key Interfaces

### EmbeddingBackend (abstract)

```python
class EmbeddingBackend(ABC):
    def embed_image(self, image_path: str) -> np.ndarray
    def embed_image_pil(self, image: Image.Image) -> np.ndarray
    def embed_text(self, text: str) -> np.ndarray
    def embed_images_batch(self, image_paths: List[str], batch_size: int) -> Tuple[List[str], List[np.ndarray]]
    def embed_pil_batch(self, images: List[Image.Image], batch_size: int) -> List[np.ndarray]
    def clear_cache(self) -> None
    @property
    def name(self) -> str
    @property
    def dimension(self) -> int
```

### Index (vector store wrapper)

```python
class Index:
    def add(self, entry_id: str, embedding: np.ndarray, metadata: dict) -> None
    def search(self, query: np.ndarray, top_k: int, where: dict) -> List[Tuple[float, str]]
    def search_with_metadata(self, query: np.ndarray, top_k: int, where: dict) -> List[Tuple[float, str, dict]]
    def get_vector_count(self) -> int
    def is_built(self) -> bool
```

---

## Modularity Boundaries

| Concern | Must live in | Must NOT leak to |
|---------|--------------|------------------|
| Embedding | `embedding/` | retrieval, search, CLI |
| Retrieval | `retrieval/` | CLI, AppController |
| Storage | `index.py`, `storage/` | embedding, retrieval |
| CLI | `cli.py` | business logic |
| Indexing | `indexer.py` | search, retrieval |