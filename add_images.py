import os
import logging
from features import extract_image_features
from index import Index
from store import Store
from validation import validate_folder_path, validate_image_path

logger = logging.getLogger(__name__)

def add_images(folder: str, store: Store, index: Index) -> int:
    """
    Add images from a folder to the store using batch processing.
    
    Validates folder exists and is readable. Skips invalid files with logging.
    Batches feature extraction and index updates to minimize index rebuilds.
    
    Args:
        folder: Path to folder containing images
        store: Store instance
        index: Index instance
        
    Returns:
        Number of images successfully added
    """
    # Validate folder path
    is_valid, error_msg = validate_folder_path(folder)
    if not is_valid:
        logger.error(f"Cannot add images: {error_msg}")
        return 0

    new_features = []  # Batch new features before adding to index
    new_paths = []     # Batch new paths for ChromaDB IDs
    added = 0
    failed = 0
    skipped = 0
    
    logger.info(f"Scanning folder: {folder}")
    
    for file in os.listdir(folder):
        path = os.path.abspath(os.path.join(folder, file))

        # Skip non-image files
        if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Skip already indexed files
        if store.item_exists(path):
            skipped += 1
            continue

        # Validate file before processing
        is_valid, error_msg = validate_image_path(path)
        if not is_valid:
            logger.warning(f"Skipping {file}: {error_msg}")
            failed += 1
            continue

        logger.info(f"Processing: {file}")

        try:
            features = extract_image_features(path)
            store.add_item(path)
            new_paths.append(path)
            new_features.append(features)
            added += 1
        except Exception as e:
            logger.error(f"Failed to process {file}: {e}")
            failed += 1

    if added > 0:
        store.save()
        index.add_vectors(new_paths, new_features)
        logger.info(f"Added {added} images, skipped {skipped}, failed {failed}")
    else:
        if skipped > 0:
            logger.info(f"No new images. Skipped {skipped} already-indexed files.")
        else:
            logger.info("No new images found")
    
    return added
