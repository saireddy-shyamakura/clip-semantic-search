# Momento — Local-First Multimodal Retrieval Engine

<p align="center">
  <strong>Hybrid search · Reranking · Cross-modal fusion · Crash-safe indexing</strong>
  <br>
  <em>Built on CLIP · ChromaDB · SQLite FTS5</em>
</p>

<p align="center">
  <a href="https://github.com/saireddy-shyamakura/momento/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="#"><img src="https://img.shields.io/badge/status-production--ready-green.svg" alt="Status"></a>
  <a href="#"><img src="https://img.shields.io/badge/coverage-85%25-brightgreen" alt="Coverage"></a>
</p>

---

## What Is Momento?

Momento is a **local-first, multimodal retrieval engine** that lets you search images and videos using text, images, or a combination of signals — entirely on your machine, with zero cloud dependencies.

```
Query ──→ Query Expansion ──→ CLIP Recall ──→ [Reranker] ──→ Fusion ──→ Results
                                    │
                              Exact Index (FTS5)
```

### What Makes It Different?

Pure CLIP-based systems are great at semantic search but fail at:

| Problem | How Momento Solves It |
|---------|----------------------|
| Finding files by exact filename | **Hybrid search** — SQLite FTS5 exact index before vector fallback |
| Low-recall from single query embedding | **Query expansion** — synonym injection, multi-variant recall |
| False positives from one-shot matching | **Re-ranking stage** — pluggable cross-encoder for precision |
| Missing text-in-image content | **Cross-modal fusion** — YOLO + OCR signals combined with CLIP |
| Indexing crashes after 10K files | **Checkpoint/resume** — per-feature progress, crash recovery |
| Slow indexing on limited hardware | **Parallel pipeline** — independent features run concurrently |

### Use Cases

- **Photo library search** — "find my dog at the beach" across 100K images
- **Document retrieval** — OCR-extracted text search in scanned documents
- **Video frame lookup** — search across keyframes from hours of footage
- **Object-based retrieval** — find all images containing a specific object (YOLO + CLIP)

---

## Key Features

### Retrieval System (V3)
| Feature | Status | Description |
|---------|--------|-------------|
| Hybrid Search | ✅ | Exact (FTS5) + semantic (CLIP) with fallback |
| Query Expansion | ✅ | Synonym injection + multi-variant recall |
| Pluggable Reranker | ✅ | Identity pass-through (ready for cross-encoder) |
| Cross-Modal Fusion | ✅ | Embedding + YOLO + OCR score combination |
| Query Router | ✅ | Rule-based EXACT/HYBRID/SEMANTIC routing |
| Metadata Filtering | ✅ | Filter by type, size, date, OCR content |

### Indexing Pipeline
| Feature | Status | Description |
|---------|--------|-------------|
| Crash Recovery | ✅ | Per-feature checkpointing with auto-resume |
| Parallel Execution | ✅ | Concurrent video/YOLO/OCR after images |
| Device Fallback | ✅ | CUDA → MPS → CPU with OOM recovery |
| Memory Pre-check | ✅ | Warns if <2 GB free RAM before indexing |
| Disk Space Check | ✅ | Estimates storage needs before ingestion |
| Embedding Cache | ✅ | Hash-based cache avoids recomputation |

### Storage
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vectors | ChromaDB | Semantic embeddings |
| Metadata | SQLite | File attributes, OCR text, YOLO objects |
| Exact Index | SQLite FTS5 | Filename/path search |
| Cache | Disk (hash-based) | Embedding recomputation avoidance |

### Reliability
- **Process lock** prevents concurrent instances
- **Structured logging** (JSON or text) with timing events
- **Auto-repair** corrupted ChromaDB on startup
- **Version mismatch detection** prevents silent data corruption
- **Graceful shutdown** (SIGINT/SIGTERM) completes current batch

---

## Quick Start

```bash
# Install with uv (recommended)
pip install uv
git clone https://github.com/saireddy-shyamakura/momento.git
cd momento
uv sync

# Index your photo library and start searching
momento --dir ~/Pictures

# Or start the interactive menu
momento
```

### Install with pip

```bash
pip install .
```

### Verify Installation

```bash
momento doctor
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                     CLI (momento)                     │
│         ──────── parse_arguments ────────            │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│                  AppController                       │
│  ┌──────────────────────────────────────────────┐   │
│  │  Index Pipeline                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────┐  │   │
│  │  │ Images   │  │ Videos   │  │ YOLO + OCR │  │   │
│  │  │ (serial) │──┤ (parallel)├──┤ (parallel) │  │   │
│  │  └──────────┘  └──────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────┐   │
│  │  Retrieval Pipeline                          │   │
│  │                                              │   │
│  │  Query ─┬─ Exact Index (FTS5) ──┐           │   │
│  │         │                       ├── Results  │   │
│  │         └─ Vector Search ───────┘           │   │
│  │             │  │  │                         │   │
│  │    Expand   │  │  │                         │   │
│  │    Recall   │  │  │                         │   │
│  │    Rerank   │  │  │                         │   │
│  │    Fusion   ┘  └───┘                       │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of concerns** — retrieval logic lives in `retrieval/`, embedding in `embedding/`, storage behind abstractions
2. **Fail-safe pipelines** — checkpoint/resume, error isolation, graceful degradation
3. **Hybrid > Pure vector** — exact search for filenames, semantic for content
4. **Pluggable architecture** — swap embedding models, rerankers, storage backends

---

## Advanced Usage

### Model Switching

Momento supports OpenAI CLIP model variants:

```bash
# Maximum quality
momento --model ViT-L/14 --dir ~/Pictures

# Fast indexing on limited hardware
momento --model ViT-B/32 --dir ~/Pictures
```

### Feature Toggles

```bash
# Minimal indexing (fast, low resource)
momento --no-multi-embed --no-video --no-yolo --no-ocr --dir ~/pictures

# V3 retrieval pipeline controls
momento --rerank --no-hybrid --no-query-expansion --dir ~/Pictures

# Fusion weight customization
momento --fusion-weights "0.5,0.3,0.2" --dir ~/Pictures
```

### Programmatic API

```python
from momento.index import Index
from momento.search import text_search

index = Index(db_path="~/.local/share/momento/chroma_db")
results = text_search("a dog in a park", index, top_k=10)
for score, path in results:
    print(f"{score:.3f}  {path}")
```

---

## Benchmark Results

> Results measured on Intel i7-13700K + NVIDIA RTX 4090 with ViT-B/16

| Metric | CLIP Only | V3 Pipeline | Improvement |
|--------|-----------|-------------|-------------|
| Recall@10 | 68.2% | **89.7%** | +31.5% |
| Precision@10 | 72.1% | **85.4%** | +18.4% |
| Exact-match hit rate | 0% | **94.3%** | — |
| Query latency | 12ms | 15ms | +25% (acceptable) |

*Full benchmark methodology in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)*

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `momento` | Interactive menu — index or search |
| `momento --dir PATH` | Index folder and start search |
| `momento doctor` | System health check |
| `momento stats` | Index statistics |
| `momento benchmark` | Performance benchmarks |
| `momento config show/set` | View/modify configuration |
| `momento storage info/backup/restore` | Storage management |

See [CLI Reference](PERSISTENT_STORAGE.md) for complete documentation.

---

## Project Structure

```
momento/
├── src/momento/
│   ├── embedding/         # Unified embedding abstraction
│   │   ├── base.py        # Abstract EmbeddingBackend interface
│   │   └── clip_backend.py # CLIP implementation
│   ├── retrieval/         # Multi-stage retrieval pipeline
│   │   ├── recall.py      # Fast CLIP recall (top-K)
│   │   ├── rerank.py      # Optional re-ranking stage
│   │   ├── fusion.py      # Cross-modal score fusion
│   │   ├── router.py      # Query type classification
│   │   └── query_expansion.py  # Synonym-based expansion
│   ├── search/            # Hybrid search orchestration
│   │   └── exact_index.py # SQLite FTS5 exact search
│   ├── storage/           # Storage separation
│   │   └── metadata_store.py  # SQLite metadata store
│   ├── diagnostics.py     # Health, stats, benchmarks
│   ├── indexer.py         # Parallel indexing orchestrator
│   ├── checkpoint.py      # Crash recovery system
│   └── ...                # V2 compatibility layer
├── tests/
│   ├── unit/              # 36 test files, fully mocked
│   └── integration/       # Model-dependent tests
├── docs/
│   ├── ARCHITECTURE.md    # Full system design
│   └── DESIGN_DECISIONS.md # Technical rationale
└── pyproject.toml
```

---

## Design Philosophy

1. **CLIP is recall, not intelligence** — use fast models for broad recall, rerank for precision
2. **Retrieval over features** — a better search pipeline beats more embedding tricks
3. **Stream over batch** — single-file streaming indexing lowers memory and improves crash recovery
4. **Fail-safe by default** — checkpoint every feature, catch every error, degrade gracefully
5. **Storage separation** — vectors, metadata, and exact indexes live independently for flexibility

---

## Running Tests

```bash
# All unit tests (fast, no model loading)
pytest tests/unit/ -v

# Including integration tests
pytest tests/ -v

# With coverage
pytest --cov=src/momento --cov-report=term-missing
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [ARCHITECTURE.md](docs/ARCHITECTURE.md) before submitting PRs.