"""
ingestion — Media processing pipeline for Momento.

Handles image ingestion, video keyframe extraction, YOLO object detection,
OCR text extraction, and image augmentation.
"""
from .image_ingest import add_images
from .media_ingest import add_images_multi, add_videos, add_objects, add_ocr
from .augment import generate_augmentations
from .video import extract_keyframes, get_video_info, extract_keyframes_interval, extract_keyframes_scene_change
from .yolo import detect_objects, detect_objects_from_pil, is_available as yolo_available, Detection
from .ocr import extract_text, extract_text_from_pil, is_available as ocr_available

__all__ = [
    "add_images", "add_images_multi",
    "add_videos", "add_objects", "add_ocr",
    "generate_augmentations",
    "extract_keyframes", "get_video_info", "extract_keyframes_interval", "extract_keyframes_scene_change",
    "detect_objects", "detect_objects_from_pil", "yolo_available", "Detection",
    "extract_text", "extract_text_from_pil", "ocr_available",
]