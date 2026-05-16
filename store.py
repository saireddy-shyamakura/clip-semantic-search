import os
import pickle
import logging
from config import STORE_PATH
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Store:
    """Manages persistent storage of image features and metadata."""
    
    def __init__(self, store_path: str = STORE_PATH):
        """Initialize the store with optional custom path."""
        self.store_path = store_path
        self._items: List[Dict[str, Any]] = []
    
    def load(self) -> None:
        """Load store from disk. Initializes fresh if corrupted."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "rb") as f:
                    self._items = pickle.load(f)
                logger.info(f"Loaded {len(self._items)} images from store")
            except FileNotFoundError:
                logger.error(f"Store file not found: {self.store_path}")
                self._items = []
            except pickle.UnpicklingError as e:
                logger.error(f"Store file corrupted (pickle error): {e}")
                self._items = []
            except Exception as e:
                logger.error(f"Failed to load store: {type(e).__name__}: {e}")
                self._items = []
        else:
            logger.info("Store file not found, starting fresh")
            self._items = []
    
    def save(self) -> None:
        """Save store to disk atomically."""
        try:
            temp_path = self.store_path + ".tmp"
            with open(temp_path, "wb") as f:
                pickle.dump(self._items, f)
            os.replace(temp_path, self.store_path)
            logger.info(f"Saved {len(self._items)} items to store")
        except IOError as e:
            logger.error(f"Failed to save store: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving store: {e}")
            raise
    
    def add_item(self, path: str, features: Any) -> bool:
        """
        Add an item to the store.
        
        Args:
            path: File path to image
            features: Feature vector (numpy array)
            
        Returns:
            True if added, False if already exists
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not path or not isinstance(path, str):
            logger.error("Invalid path: must be a non-empty string")
            raise ValueError("Path must be a non-empty string")
        
        if features is None:
            logger.error("Invalid features: cannot be None")
            raise ValueError("Features cannot be None")
        
        abs_path = os.path.abspath(path)
        
        if self.item_exists(abs_path):
            logger.debug(f"Item already exists: {abs_path}")
            return False
        
        self._items.append({
            "path": abs_path,
            "features": features
        })
        logger.debug(f"Added item: {abs_path}")
        return True
    
    def item_exists(self, path: str) -> bool:
        """Check if an image path already exists in store."""
        if not path:
            return False
        abs_path = os.path.abspath(path)
        return any(os.path.abspath(item["path"]) == abs_path for item in self._items)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all items in store."""
        return self._items
    
    def get_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get item by index, returns None if out of bounds."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None
    
    def count(self) -> int:
        """Get total number of items in store."""
        return len(self._items)
    
    def clear(self) -> None:
        """Clear all items from store."""
        self._items = []
        logger.info("Store cleared")