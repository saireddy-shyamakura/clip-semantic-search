import torch
import clip
from PIL import Image
import numpy as np
import logging
from typing import Tuple, Optional
from config import DEVICE, MODEL_NAME

logger = logging.getLogger(__name__)

_model: Optional[torch.nn.Module] = None
_preprocess: Optional[callable] = None


def get_model() -> Tuple[torch.nn.Module, callable]:
    """
    Get or load the CLIP model and preprocessing function.
    
    Model is cached globally to avoid reloading on each use.
    
    Returns:
        Tuple of (model, preprocess_function)
    """
    global _model, _preprocess
    if _model is None:
        logger.info(f"Loading CLIP model ({MODEL_NAME}) on device: {DEVICE}")
        _model, _preprocess = clip.load(MODEL_NAME, device=DEVICE)
    return _model, _preprocess


def extract_image_features(image_path: str) -> np.ndarray:
    """
    Extract normalized feature vector from an image.
    
    Handles various image formats and validates file before processing.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Normalized feature vector as numpy array (float32)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        RuntimeError: If image cannot be loaded or processed
    """
    import os
    
    # Validate file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not os.path.isfile(image_path):
        raise RuntimeError(f"Path is not a file: {image_path}")
    
    if not os.access(image_path, os.R_OK):
        raise RuntimeError(f"Image file is not readable: {image_path}")
    
    model, preprocess = get_model()

    try:
        # Try to open and convert image to RGB
        with Image.open(image_path) as img:
            # Convert to RGB to handle RGBA, grayscale, etc.
            image = img.convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except IOError as e:
        raise RuntimeError(f"Cannot read image file {image_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {type(e).__name__}: {e}")

    try:
        with torch.inference_mode():
            # Preprocess and add batch dimension
            preprocessed = preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Extract features and normalize
            features = model.encode_image(preprocessed)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype(np.float32)
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(f"GPU out of memory processing {image_path}. Try CPU mode.")
        raise RuntimeError(f"Failed to extract features from {image_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error processing {image_path}: {type(e).__name__}: {e}")


def extract_text_features(text: str) -> np.ndarray:
    """
    Extract normalized feature vector from text.
    
    Validates text before processing.
    
    Args:
        text: Text description
        
    Returns:
        Normalized feature vector as numpy array (float32)
        
    Raises:
        ValueError: If text is empty or invalid
        RuntimeError: If feature extraction fails
    """
    # Validate text
    if not text or not text.strip():
        raise ValueError("Text query cannot be empty")
    
    if len(text) > 1000:
        raise ValueError("Text query too long (max 1000 characters)")
    
    model, _ = get_model()

    try:
        with torch.inference_mode():
            # Tokenize and move to device
            tokens = clip.tokenize([text]).to(DEVICE)
            
            # Extract features and normalize
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype(np.float32)
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError("GPU out of memory. Try CPU mode.")
        raise RuntimeError(f"Failed to extract text features: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error extracting text features: {type(e).__name__}: {e}")