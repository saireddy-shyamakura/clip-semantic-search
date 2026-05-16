import numpy as np
import faiss
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class Index:
    """Manages FAISS vector indexing for fast image search with incremental updates."""
    
    def __init__(self):
        """Initialize empty index."""
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._features_matrix: Optional[np.ndarray] = None
        self._vector_count: int = 0
    
    def build(self, features_list: List[np.ndarray]) -> None:
        """
        Build FAISS index from scratch from list of feature vectors.
        
        Args:
            features_list: List of numpy arrays (each normalized to unit length)
        """
        if len(features_list) > 0:
            logger.info(f"Building index for {len(features_list)} items")
            
            self._features_matrix = np.vstack(features_list).astype(np.float32)
            dim = self._features_matrix.shape[1]
            
            # Inner Product (works as cosine similarity since vectors are normalized)
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self._features_matrix)
            self._vector_count = len(features_list)
            logger.info(f"Index built successfully with {self._vector_count} vectors")
        else:
            self._faiss_index = None
            self._features_matrix = None
            self._vector_count = 0
            logger.warning("Cannot build index: features list is empty")
    
    def add_vectors(self, new_features_list: List[np.ndarray]) -> None:
        """
        Incrementally add new vectors to existing index.
        Creates new index if not already built.
        
        Args:
            new_features_list: List of new numpy arrays to add
        """
        if len(new_features_list) == 0:
            logger.warning("No new features to add")
            return
        
        new_matrix = np.vstack(new_features_list).astype(np.float32)
        
        if self._faiss_index is None:
            # Build new index if not exists
            logger.info(f"Creating new index with {len(new_features_list)} items")
            self.build(new_features_list)
        else:
            # Incrementally add to existing index
            logger.info(f"Adding {len(new_features_list)} new vectors to existing index")
            
            if self._features_matrix is not None:
                self._features_matrix = np.vstack([self._features_matrix, new_matrix])
            else:
                self._features_matrix = new_matrix
            
            self._faiss_index.add(new_matrix)
            self._vector_count += len(new_features_list)
            logger.info(f"Index updated: now has {self._vector_count} total vectors")
    
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[float, int]]:
        """
        Search for top_k nearest neighbors.
        
        Args:
            query_vector: Query vector (shape: 1, dim) - should be normalized
            top_k: Number of top results to return
            
        Returns:
            List of (score, index) tuples sorted by descending score
        """
        if self._faiss_index is None:
            logger.debug("Search attempted on unbuilt index")
            return []
        
        # Clamp top_k to avoid requesting more results than exist
        effective_k = min(top_k, self._vector_count)
        if effective_k == 0:
            return []
        
        scores, indices = self._faiss_index.search(query_vector, effective_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((scores[0][i], idx))
        
        logger.debug(f"Search returned {len(results)} results")
        return results
    
    def is_built(self) -> bool:
        """Check if index has been built."""
        return self._faiss_index is not None
    
    def get_dimension(self) -> Optional[int]:
        """Get vector dimension of the index."""
        if self._faiss_index is not None:
            return self._faiss_index.d
        return None
    
    def get_vector_count(self) -> int:
        """Get total number of vectors in index."""
        return self._vector_count