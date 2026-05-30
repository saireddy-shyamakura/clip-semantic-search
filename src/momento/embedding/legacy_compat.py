"""
legacy_compat.py — Backward-compatibility wrappers for feature extraction (V3).

Provides the legacy API surface from the old ``features.py`` module.
All functions delegate to ``ClipBackend`` under the hood.

New code should import from ``embedding`` directly:
    from momento.embedding import ClipBackend
    backend = ClipBackend()
    emb = backend.embed_image("photo.jpg")
"""
import numpy as np
from typing import List, Optional, Tuple

from .clip_backend import ClipBackend
from ..core.logger import get_logger

logger = get_logger(__name__)

# Global backend singleton for backward compatibility
_backend: Optional[ClipBackend] = None


def _get_backend(model_name: Optional[str] = None) -> ClipBackend:
    """Get or create the global ClipBackend singleton."""
    global _backend
    if _backend is None:
        from ..core.config import MODEL_NAME
        _backend = ClipBackend(model_name or MODEL_NAME)
    return _backend


def clear_model_cache() -> None:
    """Clear the cached model to free GPU/CPU memory."""
    global _backend
    if _backend is not None:
        _backend.clear_cache()
    _backend = None


def extract_image_features(image_path: str, model_name: Optional[str] = None) -> np.ndarray:
    """Extract normalized feature vector from an image.

    Delegates to ``ClipBackend.embed_image()``.
    """
    backend = _get_backend(model_name)
    return backend.embed_image(image_path)


def _encode_pil_image(image, source_label: str = "<pil>", model_name: Optional[str] = None) -> np.ndarray:
    """Encode a single PIL Image to a normalized CLIP embedding."""
    from PIL import Image
    backend = _get_backend(model_name)
    return backend.embed_image_pil(image)


def extract_image_features_batch(
    image_paths: List[str], batch_size: int = 32
) -> Tuple[List[str], List[np.ndarray]]:
    """Extract normalized feature vectors from a batch of images.

    Delegates to ``ClipBackend.embed_images_batch()``.
    """
    backend = _get_backend()
    return backend.embed_images_batch(image_paths, batch_size)


def extract_pil_features_batch(
    images, batch_size: int = 32
) -> List[np.ndarray]:
    """Encode a list of PIL Images in batches."""
    backend = _get_backend()
    return backend.embed_pil_batch(images, batch_size)


def extract_text_features(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """Extract normalized feature vector from text.

    Delegates to ``ClipBackend.embed_text()``.
    """
    backend = _get_backend(model_name)
    return backend.embed_text(text)


# ── Multi-embedding helpers ───────────────────────────────────────────

def extract_multi_embeddings(image_path: str, model_name: Optional[str] = None) -> List[Tuple[str, np.ndarray]]:
    """Generate embeddings for the original image + augmented views.

    Kept as-is for backward compatibility.  Uses ClipBackend internally.
    """
    from PIL import Image
    from ..ingestion.augment import generate_augmentations
    from ..core.config import COMPOSITE_SEP

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")

    backend = _get_backend(model_name)

    # Original embedding
    orig_emb = backend.embed_image_pil(img_rgb)
    results: List[Tuple[str, np.ndarray]] = [("orig", orig_emb)]

    # Augmented views
    views = generate_augmentations(img_rgb)
    for suffix, aug_img in views:
        try:
            emb = backend.embed_image_pil(aug_img)
            results.append((suffix, emb))
        except Exception as e:
            logger.warning(f"Augmentation embed failed ({suffix}): {e}")

    return results


# ── YOLO + CLIP object embeddings ─────────────────────────────────────

def extract_object_embeddings(image_path: str) -> List[Tuple[dict, np.ndarray]]:
    """Run YOLO on image, CLIP-encode each detected object crop.

    Kept as-is for backward compatibility.
    """
    from ..ingestion.yolo import detect_objects, is_available as yolo_available
    from ..core.config import COMPOSITE_SEP

    if not yolo_available():
        logger.warning("YOLO not available — skipping object detection")
        return []

    detections = detect_objects(image_path)
    backend = _get_backend()
    results: List[Tuple[dict, np.ndarray]] = []

    for det in detections:
        try:
            emb = backend.embed_image_pil(det.cropped_image)
            results.append((det.to_metadata(), emb))
        except Exception as e:
            logger.warning(f"Object embed failed ({det.label}): {e}")

    return results


# ── OCR + CLIP text embedding ─────────────────────────────────────────

def extract_ocr_embedding(image_path: str):
    """Run OCR on image, encode extracted text via CLIP text encoder.

    Kept as-is for backward compatibility.
    """
    from ..ingestion.ocr import extract_text as ocr_extract, is_available as ocr_available
    from ..core.config import OCR_MIN_TEXT_LENGTH

    if not ocr_available():
        logger.warning("EasyOCR not available — skipping OCR")
        return None

    text = ocr_extract(image_path)
    if not text or len(text.strip()) < OCR_MIN_TEXT_LENGTH:
        return None

    try:
        emb = extract_text_features(text)
        return (text, emb)
    except Exception as e:
        logger.warning(f"OCR embedding failed: {e}")
        return None