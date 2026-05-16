import os
import json
import logging
from config import STORE_PATH
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Store:
    """Manages persistent JSON storage of image metadata."""
    
    def __init__(self, store_path: str = STORE_PATH):
        """Initialize the store with optional custom path."""
        self.store_path = store_path
        self._items: Dict[str, Dict[str, Any]] = {}
    
    def load(self) -> None:
        """Load store from disk. Initializes fresh if corrupted."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r") as f:
                    self._items = json.load(f)
                logger.info(f"Loaded {len(self._items)} images from JSON store")
            except Exception as e:
                logger.error(f"Failed to load JSON store: {e}")
                self._items = {}
        else:
            logger.info("JSON store file not found, starting fresh")
            self._items = {}
    
    def save(self) -> None:
        """Save store to disk atomically."""
        try:
            temp_path = self.store_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(self._items, f, indent=2)
            os.replace(temp_path, self.store_path)
            logger.info(f"Saved {len(self._items)} items to JSON store")
        except IOError as e:
            logger.error(f"Failed to save JSON store: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving JSON store: {e}")
            raise
    
    def add_item(self, path: str) -> bool:
        """
        Add an item to the store.
        
        Args:
            path: File path to image
            
        Returns:
            True if added, False if already exists
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not path or not isinstance(path, str):
            logger.error("Invalid path: must be a non-empty string")
            raise ValueError("Path must be a non-empty string")
        
        abs_path = os.path.abspath(path)
        
        if self.item_exists(abs_path):
            logger.debug(f"Item already exists: {abs_path}")
            return False
        
        self._items[abs_path] = {"path": abs_path}
        logger.debug(f"Added item: {abs_path}")
        return True
    
    def item_exists(self, path: str) -> bool:
        """Check if an image path already exists in store."""
        if not path:
            return False
        abs_path = os.path.abspath(path)
        return abs_path in self._items
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all items in store."""
        return list(self._items.values())
    
    def count(self) -> int:
        """Get total number of items in store."""
        return len(self._items)
    
    def clear(self) -> None:
        """Clear all items from store."""
        self._items = {}
        logger.info("JSON store cleared")