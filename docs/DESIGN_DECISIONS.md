# Design Decisions

This document explains the key technical decisions behind Momento's architecture.

---

## Why CLIP as the Primary Embedding Model?

**Decision:** Use OpenAI CLIP (ViT-B/16) as the default embedding backbone.

**Rationale:**
1. **Joint embedding space** — CLIP maps images and text into the same vector space, enabling both text-to-image and image-to-image search with a single model
2. **Proven quality** — CLIP's zero-shot performance on image retrieval tasks is well-documented and competitive
3. **Broad support** — GPU-accelerated inference, ONNX export, quantization — CLIP runs on everything from RTX 4090s to Raspberry Pis
4. **Model variety** — 5 model sizes (ViT-B/32 → ViT-L/14@336px) let users trade speed for quality

**Trade-off accepted:** CLIP's embedding dimension (512 for ViT-B/16) means higher storage costs than smaller models. We accept this for the quality gain.

**Future:** The `EmbeddingBackend` ABC makes it trivial to swap in SigLIP, OpenCLIP, or custom models.

---

## Why SQLite FTS5 for Exact Search (Not Tantivy / Whoosh)?

**Decision:** Use SQLite FTS5 for the exact filename/path index.

**Rationale:**
1. **Zero dependencies** — SQLite ships with Python's standard library
2. **No separate server** — embedded database, single file
3. **FTS5 performance** — BM25 scoring, prefix queries, real-time indexing
4. **Already in the stack** — ChromaDB uses SQLite internally, so SQLite is already available

**Trade-off accepted:** Tantivy would be faster for large-scale full-text search. But for filename/path lookups (typically <10K unique paths), FTS5 is more than adequate.

---

## Why Hybrid Search (Exact + Semantic) Instead of Pure Vector?

**Decision:** Always check exact index before falling back to vector search.

**Rationale:**
1. **Vector search fails on exact matches** — "photo.jpg" as a query should find the file named `photo.jpg`, not visually similar images
2. **User expectations** — when users type a filename, they expect exact file lookup, not semantic similarity
3. **Zero extra latency** — FTS5 exact match is sub-millisecond; if no match, vector search proceeds normally

**Implementation:** The exact check happens before any vector work. If `score >= 0.95`, results return immediately. Otherwise, the full V3 vector pipeline runs.

---

## Why Optional Re-ranking (Not Always On)?

**Decision:** Make re-ranking optional (`--rerank` flag), default off.

**Rationale:**
1. **Latency** — cross-encoder re-rankers add 50–200ms per query
2. **Resource cost** — re-rankers require loading an additional model into GPU memory
3. **Use-case dependent** — many users only need "good enough" recall from the CLIP stage alone
4. **Pluggable by design** — the `rerank_results()` function is ready to accept any cross-encoder; users opt in when they need maximum precision

**Future:** When cross-encoder models like Cohere or SentenceTransformer rerankers are integrated, `--rerank` will unlock them.

---

## Why Cross-Modal Fusion (YOLO + OCR + CLIP)?

**Decision:** Fuse multiple signal scores into a single relevance score.

**Rationale:**
1. **CLIP alone misses text-in-image** — a photo of a sign saying "Coffee Shop" won't match a text query for "coffee" without OCR
2. **CLIP alone misses small objects** — a tiny dog in a large landscape photo won't rank well without YOLO crop embeddings
3. **Each signal covers different blind spots** — CLIP covers general semantics, YOLO covers object presence, OCR covers text content

**Formula:** `final_score = 0.6*embedding + 0.2*object + 0.2*OCR` (configurable via `--fusion-weights`)

---

## Why Content-Backed SQLite FTS5 (Not Contentless)?

**Decision:** Use FTS5 with an external content table and triggers.

**Rationale:**
1. **Contentless FTS5** stores only search tokens, not column values — `SELECT path` returns `NULL`
2. **External content table** stores actual file paths, enabling proper CRUD (DELETE, UPDATE, COUNT)
3. **Triggers** automatically sync content changes to the FTS5 index — no manual sync needed

**Trade-off:** Slightly more complex schema setup. Much simpler application code.

---

## Why Checkpoint/Resume at Feature Level?

**Decision:** Track progress per-feature (images, videos, YOLO, OCR), not per-file.

**Rationale:**
1. **Features are the recovery unit** — if YOLO crashes after images are done, only YOLO needs to re-run
2. **Parallel execution** — independent features run concurrently; checkpointing per-feature means each can be individually resumed
3. **Meaningful granularity** — per-file tracking would add complexity and storage overhead without proportional benefit

---

## Why Storage Separation (Not Everything in ChromaDB)?

**Decision:** Store metadata and exact index separately from vector embeddings.

**Rationale:**
1. **ChromaDB is optimized for vectors** — metadata queries in ChromaDB are slower than SQLite
2. **Flexible filtering** — SQLite supports arbitrary WHERE clauses, JOINs, and aggregations
3. **Independence** — vectors, metadata, and exact index can evolve independently
4. **Backup granularity** — each storage component can be backed up/restored separately

---

## Why Not Cloud/API-Based Embeddings?

**Decision:** Keep everything local — no cloud dependencies.

**Rationale:**
1. **Privacy** — all data stays on the user's machine
2. **Offline capability** — works without internet
3. **No recurring costs** — one-time hardware cost only
4. **Low latency** — no network round-trips
5. **Scalability** — limited only by local hardware

**Trade-off accepted:** Users need a GPU for acceptable performance with large datasets. CPU-only indexing of 100K+ images is slow.

---

## Why Python (Not Rust / C++)?

**Decision:** Python for rapid development, with hot paths in native libraries.

**Rationale:**
1. **Ecosystem access** — PyTorch, ChromaDB, Ultralytics, EasyOCR are all Python-native
2. **Model inference in C** — CLIP, YOLO, and OCR inference all run in native code (CUDA, OpenCV)
3. **Development velocity** — Python enables faster iteration on the retrieval pipeline design
4. **Approachability** — easier for the open-source community to contribute

**Future:** Hot paths (embedding, search) could be extracted to Rust via PyO3 if performance becomes a bottleneck.