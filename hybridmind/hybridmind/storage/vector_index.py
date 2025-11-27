"""
FAISS-based vector index for HybridMind.
Handles vector storage, similarity search, and persistence.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorIndex:
    """
    FAISS-based vector index for similarity search.
    Uses IndexFlatIP for cosine similarity with normalized vectors.
    Falls back to NumPy if FAISS is not available.
    """
    
    def __init__(self, dimension: int = 384, index_path: Optional[str] = None):
        """
        Initialize vector index.
        
        Args:
            dimension: Embedding dimension (384 for MiniLM)
            index_path: Path for index persistence
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        
        # Mapping between FAISS indices and node IDs
        self.id_map: Dict[int, str] = {}  # FAISS idx -> node_id
        self.reverse_map: Dict[str, int] = {}  # node_id -> FAISS idx
        
        # Initialize index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dimension)
            self._use_faiss = True
        else:
            # Fallback to NumPy-based search
            self._vectors: List[np.ndarray] = []
            self._use_faiss = False
        
        # Load from disk if exists
        if self.index_path and self.index_path.exists():
            self.load()
    
    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        if self._use_faiss:
            return self.index.ntotal
        return len(self._vectors)
    
    def add(self, node_id: str, embedding: np.ndarray):
        """
        Add a vector to the index.
        
        Args:
            node_id: Unique node identifier
            embedding: Vector embedding (will be normalized)
        """
        # Normalize for cosine similarity
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized = embedding / norm
        else:
            normalized = embedding
        
        # Remove old entry if exists
        if node_id in self.reverse_map:
            self.remove(node_id)
        
        # Add to index
        idx = self.size
        
        if self._use_faiss:
            self.index.add(normalized.reshape(1, -1))
        else:
            self._vectors.append(normalized)
        
        # Update mappings
        self.id_map[idx] = node_id
        self.reverse_map[node_id] = idx
    
    def remove(self, node_id: str) -> bool:
        """
        Remove a vector from the index.
        Note: FAISS doesn't support efficient removal, so we rebuild.
        
        Args:
            node_id: Node ID to remove
            
        Returns:
            True if removed, False if not found
        """
        if node_id not in self.reverse_map:
            return False
        
        # Get all vectors except the one to remove
        remaining = []
        remaining_ids = []
        
        for idx, nid in sorted(self.id_map.items()):
            if nid != node_id:
                if self._use_faiss:
                    vec = faiss.rev_swig_ptr(
                        self.index.get_xb() + idx * self.dimension,
                        self.dimension
                    ).copy()
                else:
                    vec = self._vectors[idx]
                remaining.append(vec)
                remaining_ids.append(nid)
        
        # Rebuild index
        self._rebuild(remaining, remaining_ids)
        return True
    
    def _rebuild(self, vectors: List[np.ndarray], node_ids: List[str]):
        """Rebuild index with new vectors."""
        # Clear current index
        if self._use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self._vectors = []
        
        self.id_map = {}
        self.reverse_map = {}
        
        # Add all vectors
        for vec, nid in zip(vectors, node_ids):
            idx = self.size
            if self._use_faiss:
                self.index.add(vec.reshape(1, -1))
            else:
                self._vectors.append(vec)
            self.id_map[idx] = nid
            self.reverse_map[nid] = idx
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of (node_id, score) tuples sorted by score descending
        """
        if self.size == 0:
            return []
        
        # Normalize query
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            normalized_query = query / norm
        else:
            normalized_query = query
        
        # Limit top_k to index size
        k = min(top_k, self.size)
        
        if self._use_faiss:
            # FAISS search
            scores, indices = self.index.search(
                normalized_query.reshape(1, -1),
                k
            )
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx in self.id_map:
                    # Convert inner product to cosine similarity (already normalized)
                    sim_score = float(score)
                    if sim_score >= min_score:
                        results.append((self.id_map[idx], sim_score))
        else:
            # NumPy fallback
            if len(self._vectors) == 0:
                return []
            
            vectors_matrix = np.vstack(self._vectors)
            # Cosine similarity via dot product (vectors are normalized)
            similarities = np.dot(vectors_matrix, normalized_query)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                idx = int(idx)
                if idx in self.id_map:
                    sim_score = float(similarities[idx])
                    if sim_score >= min_score:
                        results.append((self.id_map[idx], sim_score))
        
        return results
    
    def get_vector(self, node_id: str) -> Optional[np.ndarray]:
        """Get vector by node ID."""
        if node_id not in self.reverse_map:
            return None
        
        idx = self.reverse_map[node_id]
        
        if self._use_faiss:
            # Extract vector from FAISS index
            return faiss.rev_swig_ptr(
                self.index.get_xb() + idx * self.dimension,
                self.dimension
            ).copy()
        else:
            return self._vectors[idx].copy()
    
    def save(self, path: Optional[str] = None):
        """Save index to disk."""
        save_path = Path(path) if path else self.index_path
        if save_path is None:
            raise ValueError("No path specified for saving")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "dimension": self.dimension,
            "id_map": self.id_map,
            "reverse_map": self.reverse_map,
            "use_faiss": self._use_faiss,
        }
        
        if self._use_faiss:
            # Save FAISS index separately
            faiss_path = save_path.with_suffix('.faiss')
            faiss.write_index(self.index, str(faiss_path))
        else:
            data["vectors"] = self._vectors
        
        # Save metadata
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: Optional[str] = None):
        """Load index from disk."""
        load_path = Path(path) if path else self.index_path
        if load_path is None or not load_path.exists():
            return
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data["dimension"]
        self.id_map = data["id_map"]
        self.reverse_map = data["reverse_map"]
        
        if data.get("use_faiss", False) and FAISS_AVAILABLE:
            faiss_path = load_path.with_suffix('.faiss')
            if faiss_path.exists():
                self.index = faiss.read_index(str(faiss_path))
                self._use_faiss = True
        elif "vectors" in data:
            self._vectors = data["vectors"]
            self._use_faiss = False
    
    def rebuild_from_embeddings(self, embeddings: List[Tuple[str, np.ndarray]]):
        """
        Rebuild entire index from list of (node_id, embedding) tuples.
        Used when loading from SQLite.
        """
        vectors = []
        ids = []
        
        for node_id, embedding in embeddings:
            # Normalize
            embedding = np.asarray(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                normalized = embedding / norm
            else:
                normalized = embedding
            vectors.append(normalized)
            ids.append(node_id)
        
        self._rebuild(vectors, ids)
    
    def clear(self):
        """Clear all vectors from index."""
        if self._use_faiss:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self._vectors = []
        self.id_map = {}
        self.reverse_map = {}

