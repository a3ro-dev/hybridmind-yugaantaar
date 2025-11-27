"""
Tests for HybridMind storage layer.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.graph_index import GraphIndex


class TestSQLiteStore:
    """Tests for SQLite storage."""
    
    @pytest.fixture
    def store(self):
        """Create temporary SQLite store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = SQLiteStore(str(db_path))
            yield store
            store.close()
    
    def test_create_node(self, store):
        """Test node creation."""
        embedding = np.random.randn(384).astype(np.float32)
        result = store.create_node(
            node_id="test-1",
            text="Test node content",
            metadata={"title": "Test"},
            embedding=embedding
        )
        
        assert result["id"] == "test-1"
        assert result["text"] == "Test node content"
        assert result["metadata"]["title"] == "Test"
    
    def test_get_node(self, store):
        """Test node retrieval."""
        embedding = np.random.randn(384).astype(np.float32)
        store.create_node("test-1", "Content", {}, embedding)
        
        node = store.get_node("test-1")
        assert node is not None
        assert node["text"] == "Content"
        assert node["embedding"] is not None
    
    def test_update_node(self, store):
        """Test node update."""
        store.create_node("test-1", "Original", {"v": 1}, None)
        
        result = store.update_node("test-1", text="Updated", metadata={"v": 2})
        
        assert result["text"] == "Updated"
        assert result["metadata"]["v"] == 2
    
    def test_delete_node(self, store):
        """Test node deletion."""
        store.create_node("test-1", "Content", {}, None)
        deleted, edges = store.delete_node("test-1")
        
        assert deleted
        assert store.get_node("test-1") is None
    
    def test_create_edge(self, store):
        """Test edge creation."""
        store.create_node("node-1", "Node 1", {}, None)
        store.create_node("node-2", "Node 2", {}, None)
        
        result = store.create_edge(
            edge_id="edge-1",
            source_id="node-1",
            target_id="node-2",
            edge_type="relates_to",
            weight=0.8
        )
        
        assert result["id"] == "edge-1"
        assert result["type"] == "relates_to"
        assert result["weight"] == 0.8
    
    def test_get_node_edges(self, store):
        """Test getting edges for a node."""
        store.create_node("node-1", "Node 1", {}, None)
        store.create_node("node-2", "Node 2", {}, None)
        store.create_edge("edge-1", "node-1", "node-2", "cites", 1.0)
        
        edges = store.get_node_edges("node-1")
        assert len(edges) == 1
        assert edges[0]["type"] == "cites"


class TestVectorIndex:
    """Tests for vector index."""
    
    @pytest.fixture
    def index(self):
        """Create temporary vector index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "vector.index"
            yield VectorIndex(dimension=384, index_path=str(index_path))
    
    def test_add_and_search(self, index):
        """Test adding vectors and searching."""
        # Add vectors
        for i in range(10):
            vec = np.random.randn(384).astype(np.float32)
            index.add(f"node-{i}", vec)
        
        # Search
        query = np.random.randn(384).astype(np.float32)
        results = index.search(query, top_k=5)
        
        assert len(results) <= 5  # May return fewer if scores are negative
        assert len(results) > 0  # Should return at least some results
        assert all(isinstance(r[0], str) for r in results)
        assert all(isinstance(r[1], float) for r in results)
    
    def test_remove(self, index):
        """Test vector removal (soft delete with automatic compaction)."""
        # Add multiple nodes so deletion doesn't trigger immediate compaction
        for i in range(5):
            vec = np.random.randn(384).astype(np.float32)
            index.add(f"node-{i}", vec)
        
        assert index.size == 5
        assert index.remove("node-0")
        assert index.size == 4
        # Node should be in deleted_ids (soft deleted)
        assert "node-0" in index.deleted_ids
    
    def test_soft_delete_excludes_from_search(self, index):
        """Test that soft deleted nodes are excluded from search."""
        base_vec = np.random.randn(384).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        for i in range(5):
            noise = np.random.randn(384).astype(np.float32) * 0.1
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            index.add(f"node-{i}", vec)
        
        # Soft delete
        index.remove("node-2")
        
        # Search should not return deleted node
        results = index.search(base_vec, top_k=10)
        result_ids = [r[0] for r in results]
        assert "node-2" not in result_ids
    
    def test_batch_add(self, index):
        """Test batch adding vectors."""
        nodes = []
        for i in range(10):
            vec = np.random.randn(384).astype(np.float32)
            nodes.append((f"batch-{i}", vec))
        
        index.add_batch(nodes)
        assert index.size == 10
    
    def test_get_stats(self, index):
        """Test get_stats method."""
        for i in range(5):
            vec = np.random.randn(384).astype(np.float32)
            index.add(f"node-{i}", vec)
        
        index.remove("node-0")
        
        stats = index.get_stats()
        assert stats["total_vectors"] == 5
        assert stats["active_vectors"] == 4
        assert stats["deleted_vectors"] == 1
    
    def test_remove_with_multiple_vectors(self, index):
        """Test removing one vector when multiple exist (soft delete)."""
        # Add multiple vectors with similar direction for better search results
        base_vec = np.random.randn(384).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        for i in range(5):
            # Add small perturbation to base vector
            noise = np.random.randn(384).astype(np.float32) * 0.1
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)  # Normalize
            index.add(f"node-{i}", vec)
        
        assert index.size == 5
        
        # Remove middle node (soft delete)
        assert index.remove("node-2")
        assert index.size == 4
        
        # Verify node is soft-deleted (in deleted_ids, not completely removed)
        assert "node-2" in index.deleted_ids
        
        # Search with a similar vector (should find remaining nodes)
        results = index.search(base_vec, top_k=10)
        result_ids = [r[0] for r in results]
        
        # Must return at least one result to verify search works
        assert len(results) > 0, "Search should return results after node removal"
        
        # Soft-deleted node should not be in results
        assert "node-2" not in result_ids
        
        # All results should be from active nodes
        for rid in result_ids:
            assert rid in ["node-0", "node-1", "node-3", "node-4"]
    
    def test_save_and_load(self, index):
        """Test persistence."""
        vec = np.random.randn(384).astype(np.float32)
        index.add("test-node", vec)
        
        # Save
        index.save()
        
        # Create new index and load
        new_index = VectorIndex(dimension=384, index_path=index.index_path)
        new_index.load()
        
        assert new_index.size == 1


class TestGraphIndex:
    """Tests for graph index."""
    
    @pytest.fixture
    def graph(self):
        """Create temporary graph index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "graph.pkl"
            yield GraphIndex(index_path=str(index_path))
    
    def test_add_nodes_and_edges(self, graph):
        """Test adding nodes and edges."""
        graph.add_node("node-1")
        graph.add_node("node-2")
        graph.add_edge("node-1", "node-2", "cites", 1.0)
        
        assert graph.node_count == 2
        assert graph.edge_count == 1
    
    def test_traverse_bfs(self, graph):
        """Test BFS traversal."""
        # Create a chain: 1 -> 2 -> 3 -> 4
        for i in range(1, 5):
            graph.add_node(f"node-{i}")
        for i in range(1, 4):
            graph.add_edge(f"node-{i}", f"node-{i+1}", "next", 1.0)
        
        results = graph.traverse_bfs("node-1", max_depth=2)
        
        # Should find node-2 (depth 1) and node-3 (depth 2)
        node_ids = [r[0] for r in results]
        assert "node-2" in node_ids
        assert "node-3" in node_ids
    
    def test_proximity_score(self, graph):
        """Test proximity score calculation."""
        graph.add_edge("node-1", "node-2", "cites", 1.0)
        graph.add_edge("node-2", "node-3", "cites", 1.0)
        
        # Direct connection
        score1 = graph.compute_proximity_score("node-2", ["node-1"])
        assert score1 > 0
        
        # 2-hop connection
        score2 = graph.compute_proximity_score("node-3", ["node-1"])
        assert score2 > 0
        assert score1 > score2  # Closer = higher score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

