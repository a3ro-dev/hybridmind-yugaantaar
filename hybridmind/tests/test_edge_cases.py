"""
Comprehensive edge case tests for HybridMind.
Tests validation, error handling, boundary conditions, and edge cases.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
import os
import uuid

# Set test environment before imports
os.environ["HYBRIDMIND_DATABASE_PATH"] = "test_data/hybridmind.db"
os.environ["HYBRIDMIND_VECTOR_INDEX_PATH"] = "test_data/vector.index"
os.environ["HYBRIDMIND_GRAPH_INDEX_PATH"] = "test_data/graph.pkl"

from hybridmind.main import app
from hybridmind.storage.sqlite_store import SQLiteStore
from hybridmind.storage.vector_index import VectorIndex
from hybridmind.storage.graph_index import GraphIndex
from hybridmind.engine.embedding import EmbeddingEngine


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def temp_store():
    """Create temporary SQLite store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = SQLiteStore(str(db_path))
        yield store
        store.close()


@pytest.fixture
def temp_vector_index():
    """Create temporary vector index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "vector.index"
        yield VectorIndex(dimension=384, index_path=str(index_path))


@pytest.fixture
def temp_graph_index():
    """Create temporary graph index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "graph.pkl"
        yield GraphIndex(index_path=str(index_path))


# ============================================================================
# INPUT VALIDATION EDGE CASES
# ============================================================================

class TestInputValidation:
    """Tests for input validation edge cases."""
    
    def test_create_node_empty_text(self, client):
        """Test creating node with empty text - should fail."""
        response = client.post("/nodes", json={
            "text": "",
            "metadata": {}
        })
        assert response.status_code == 422  # Validation error
    
    def test_create_node_whitespace_only_text(self, client):
        """Test creating node with whitespace-only text."""
        response = client.post("/nodes", json={
            "text": "   ",  # Just spaces
            "metadata": {}
        })
        # Should either fail validation or create with trimmed text
        assert response.status_code in [201, 422]
    
    def test_create_node_very_long_text(self, client):
        """Test creating node with very long text."""
        long_text = "A" * 10000  # 10K characters
        response = client.post("/nodes", json={
            "text": long_text,
            "metadata": {}
        })
        assert response.status_code == 201
        data = response.json()
        assert len(data["text"]) == 10000
    
    def test_create_node_unicode_text(self, client):
        """Test creating node with Unicode/multilingual text."""
        response = client.post("/nodes", json={
            "text": "这是中文测试 🚀 العربية हिंदी",
            "metadata": {"language": "mixed"}
        })
        assert response.status_code == 201
        data = response.json()
        assert "中文" in data["text"]
        assert "🚀" in data["text"]
    
    def test_create_node_special_characters(self, client):
        """Test creating node with special characters."""
        response = client.post("/nodes", json={
            "text": "Test <script>alert('xss')</script> & < > \" '",
            "metadata": {}
        })
        assert response.status_code == 201
    
    def test_create_node_complex_metadata(self, client):
        """Test creating node with complex nested metadata."""
        response = client.post("/nodes", json={
            "text": "Test node",
            "metadata": {
                "nested": {"deep": {"value": 123}},
                "array": [1, 2, 3, {"a": "b"}],
                "null_value": None,
                "bool": True,
                "float": 3.14159
            }
        })
        assert response.status_code == 201
        data = response.json()
        assert data["metadata"]["nested"]["deep"]["value"] == 123
    
    def test_create_edge_boundary_weights(self, client):
        """Test creating edges with boundary weight values."""
        # Create two nodes
        node1 = client.post("/nodes", json={"text": "Node 1", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Node 2", "metadata": {}}).json()
        
        # Test weight = 0.0
        response = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "test",
            "weight": 0.0
        })
        assert response.status_code == 201
        
        # Create another pair of nodes for max weight test
        node3 = client.post("/nodes", json={"text": "Node 3", "metadata": {}}).json()
        node4 = client.post("/nodes", json={"text": "Node 4", "metadata": {}}).json()
        
        # Test weight = 1.0
        response = client.post("/edges", json={
            "source_id": node3["id"],
            "target_id": node4["id"],
            "type": "test",
            "weight": 1.0
        })
        assert response.status_code == 201
    
    def test_create_edge_invalid_weight(self, client):
        """Test creating edges with invalid weight values."""
        node1 = client.post("/nodes", json={"text": "Node 1", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Node 2", "metadata": {}}).json()
        
        # Test weight > 1.0
        response = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "test",
            "weight": 1.5
        })
        assert response.status_code == 422
        
        # Test negative weight
        response = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "test",
            "weight": -0.5
        })
        assert response.status_code == 422


# ============================================================================
# NODE EDGE CASES
# ============================================================================

class TestNodeEdgeCases:
    """Tests for node-related edge cases."""
    
    def test_get_node_invalid_id_format(self, client):
        """Test getting node with invalid ID format."""
        response = client.get("/nodes/not-a-uuid")
        assert response.status_code == 404
    
    def test_update_nonexistent_node(self, client):
        """Test updating a non-existent node."""
        response = client.put("/nodes/nonexistent-id", json={
            "text": "Updated",
            "metadata": {}
        })
        assert response.status_code == 404
    
    def test_delete_nonexistent_node(self, client):
        """Test deleting a non-existent node."""
        response = client.delete("/nodes/nonexistent-id")
        assert response.status_code == 404
    
    def test_delete_node_with_edges_cascade(self, client):
        """Test that deleting a node also removes its edges."""
        # Create nodes and edge
        node1 = client.post("/nodes", json={"text": "Node 1", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Node 2", "metadata": {}}).json()
        edge = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "test"
        }).json()
        
        # Delete source node
        delete_response = client.delete(f"/nodes/{node1['id']}")
        assert delete_response.status_code == 200
        data = delete_response.json()
        assert data["edges_removed"] >= 1
        
        # Edge should no longer exist
        edge_response = client.get(f"/edges/{edge['id']}")
        assert edge_response.status_code == 404
    
    def test_list_nodes_pagination(self, client):
        """Test node listing with pagination."""
        # Create multiple nodes
        for i in range(5):
            client.post("/nodes", json={"text": f"Node {i}", "metadata": {}})
        
        # Test skip and limit
        response = client.get("/nodes", params={"skip": 0, "limit": 2})
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2
        
        # Test skip
        response2 = client.get("/nodes", params={"skip": 2, "limit": 2})
        assert response2.status_code == 200
    
    def test_update_node_partial(self, client):
        """Test partial node update (only metadata)."""
        node = client.post("/nodes", json={
            "text": "Original text",
            "metadata": {"v": 1}
        }).json()
        
        # Update only metadata
        response = client.put(f"/nodes/{node['id']}", json={
            "metadata": {"v": 2},
            "regenerate_embedding": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Original text"  # Text unchanged
        assert data["metadata"]["v"] == 2  # Metadata updated


# ============================================================================
# EDGE EDGE CASES
# ============================================================================

class TestEdgeEdgeCases:
    """Tests for edge-related edge cases."""
    
    def test_create_self_loop_edge(self, client):
        """Test creating an edge from a node to itself."""
        node = client.post("/nodes", json={"text": "Self-loop node", "metadata": {}}).json()
        
        response = client.post("/edges", json={
            "source_id": node["id"],
            "target_id": node["id"],
            "type": "self_reference"
        })
        # Self-loops may or may not be allowed depending on implementation
        assert response.status_code in [201, 400, 422]
    
    def test_create_duplicate_edge(self, client):
        """Test creating duplicate edges between same nodes."""
        node1 = client.post("/nodes", json={"text": "Node 1", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Node 2", "metadata": {}}).json()
        
        # First edge
        edge1 = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "cites"
        })
        assert edge1.status_code == 201
        
        # Second edge (same type)
        edge2 = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "cites"
        })
        # May allow duplicates or reject them
        assert edge2.status_code in [201, 400, 409]
        
        # Different type should be allowed
        edge3 = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "related_to"
        })
        assert edge3.status_code == 201
    
    def test_get_node_edges_empty(self, client):
        """Test getting edges for node with no edges."""
        node = client.post("/nodes", json={"text": "Isolated node", "metadata": {}}).json()
        
        # Endpoint is /edges/node/{node_id}
        response = client.get(f"/edges/node/{node['id']}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0
    
    def test_get_node_edges_direction_filter(self, client):
        """Test getting edges filtered by direction."""
        node1 = client.post("/nodes", json={"text": "Node 1", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Node 2", "metadata": {}}).json()
        node3 = client.post("/nodes", json={"text": "Node 3", "metadata": {}}).json()
        
        # Outgoing edge from node1
        client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "outgoing_test"
        })
        
        # Incoming edge to node1
        client.post("/edges", json={
            "source_id": node3["id"],
            "target_id": node1["id"],
            "type": "incoming_test"
        })
        
        # Endpoint is /edges/node/{node_id}
        # Get all edges
        response_both = client.get(f"/edges/node/{node1['id']}", params={"direction": "both"})
        assert response_both.status_code == 200
        
        # Get outgoing only
        response_out = client.get(f"/edges/node/{node1['id']}", params={"direction": "outgoing"})
        assert response_out.status_code == 200
        
        # Get incoming only
        response_in = client.get(f"/edges/node/{node1['id']}", params={"direction": "incoming"})
        assert response_in.status_code == 200


# ============================================================================
# SEARCH EDGE CASES
# ============================================================================

class TestSearchEdgeCases:
    """Tests for search-related edge cases."""
    
    def test_vector_search_empty_query(self, client):
        """Test vector search with empty query."""
        response = client.post("/search/vector", json={
            "query_text": "",
            "top_k": 5
        })
        assert response.status_code == 422
    
    def test_vector_search_very_long_query(self, client):
        """Test vector search with very long query."""
        long_query = "machine learning " * 500  # Very long query
        response = client.post("/search/vector", json={
            "query_text": long_query,
            "top_k": 5
        })
        assert response.status_code == 200
    
    def test_vector_search_high_min_score(self, client):
        """Test vector search with very high min_score that filters everything."""
        response = client.post("/search/vector", json={
            "query_text": "test query",
            "top_k": 100,
            "min_score": 0.99  # Very high threshold
        })
        assert response.status_code == 200
        data = response.json()
        # Should return empty or very few results
        assert len(data["results"]) <= 5
    
    def test_vector_search_top_k_bounds(self, client):
        """Test vector search with boundary top_k values."""
        # Minimum top_k
        response = client.post("/search/vector", json={
            "query_text": "test",
            "top_k": 1
        })
        assert response.status_code == 200
        assert len(response.json()["results"]) <= 1
        
        # Maximum top_k
        response = client.post("/search/vector", json={
            "query_text": "test",
            "top_k": 100
        })
        assert response.status_code == 200
    
    def test_vector_search_invalid_top_k(self, client):
        """Test vector search with invalid top_k."""
        # top_k = 0
        response = client.post("/search/vector", json={
            "query_text": "test",
            "top_k": 0
        })
        assert response.status_code == 422
        
        # top_k > 100
        response = client.post("/search/vector", json={
            "query_text": "test",
            "top_k": 200
        })
        assert response.status_code == 422
    
    def test_graph_search_nonexistent_start(self, client):
        """Test graph search with non-existent start node."""
        response = client.get("/search/graph", params={
            "start_id": "nonexistent-node-id",
            "depth": 2
        })
        assert response.status_code in [200, 404]  # May return empty or 404
    
    def test_graph_search_max_depth(self, client):
        """Test graph search with maximum depth."""
        # Create a node first
        node = client.post("/nodes", json={"text": "Start node", "metadata": {}}).json()
        
        response = client.get("/search/graph", params={
            "start_id": node["id"],
            "depth": 5  # Maximum allowed
        })
        assert response.status_code == 200
        
        # Exceed max depth
        response = client.get("/search/graph", params={
            "start_id": node["id"],
            "depth": 10
        })
        assert response.status_code == 422
    
    def test_hybrid_search_zero_weights(self, client):
        """Test hybrid search with zero weights."""
        # All vector weight
        response = client.post("/search/hybrid", json={
            "query_text": "test query",
            "top_k": 5,
            "vector_weight": 1.0,
            "graph_weight": 0.0
        })
        assert response.status_code == 200
        
        # All graph weight
        response = client.post("/search/hybrid", json={
            "query_text": "test query",
            "top_k": 5,
            "vector_weight": 0.0,
            "graph_weight": 1.0
        })
        assert response.status_code == 200
    
    def test_hybrid_search_with_nonexistent_anchors(self, client):
        """Test hybrid search with non-existent anchor nodes."""
        response = client.post("/search/hybrid", json={
            "query_text": "test query",
            "top_k": 5,
            "anchor_nodes": ["nonexistent-1", "nonexistent-2"]
        })
        # Should handle gracefully
        assert response.status_code == 200


# ============================================================================
# STORAGE LAYER EDGE CASES
# ============================================================================

class TestStorageEdgeCases:
    """Tests for storage layer edge cases."""
    
    def test_vector_index_empty_search(self, temp_vector_index):
        """Test searching empty vector index."""
        query = np.random.randn(384).astype(np.float32)
        results = temp_vector_index.search(query, top_k=5)
        assert len(results) == 0
    
    def test_vector_index_remove_nonexistent(self, temp_vector_index):
        """Test removing non-existent vector."""
        result = temp_vector_index.remove("nonexistent-id")
        assert result == False
    
    def test_vector_index_duplicate_add(self, temp_vector_index):
        """Test adding same node ID twice."""
        vec1 = np.random.randn(384).astype(np.float32)
        vec2 = np.random.randn(384).astype(np.float32)
        
        temp_vector_index.add("test-node", vec1)
        assert temp_vector_index.size == 1
        
        # Adding with same ID should replace
        temp_vector_index.add("test-node", vec2)
        assert temp_vector_index.size == 1
    
    def test_graph_index_empty_traversal(self, temp_graph_index):
        """Test traversing empty graph."""
        results = temp_graph_index.traverse_bfs("nonexistent", max_depth=2)
        assert len(results) == 0
    
    def test_graph_index_isolated_node(self, temp_graph_index):
        """Test traversal from isolated node."""
        temp_graph_index.add_node("isolated")
        results = temp_graph_index.traverse_bfs("isolated", max_depth=2)
        assert len(results) == 0
    
    def test_graph_index_circular_reference(self, temp_graph_index):
        """Test graph with circular references."""
        # Create cycle: A -> B -> C -> A
        temp_graph_index.add_node("A")
        temp_graph_index.add_node("B")
        temp_graph_index.add_node("C")
        temp_graph_index.add_edge("A", "B", "next", 1.0)
        temp_graph_index.add_edge("B", "C", "next", 1.0)
        temp_graph_index.add_edge("C", "A", "next", 1.0)
        
        # Should handle cycle without infinite loop
        results = temp_graph_index.traverse_bfs("A", max_depth=5)
        assert len(results) == 2  # B and C
    
    def test_graph_proximity_no_path(self, temp_graph_index):
        """Test proximity score when no path exists."""
        temp_graph_index.add_node("A")
        temp_graph_index.add_node("B")
        # No edge between them
        
        score = temp_graph_index.compute_proximity_score("A", ["B"])
        assert score == 0.0
    
    def test_sqlite_store_empty_queries(self, temp_store):
        """Test SQLite store with empty database."""
        # Get non-existent node
        node = temp_store.get_node("nonexistent")
        assert node is None
        
        # Get edges for non-existent node
        edges = temp_store.get_node_edges("nonexistent")
        assert len(edges) == 0
        
        # List nodes from empty db
        nodes = temp_store.list_nodes(skip=0, limit=10)
        assert len(nodes) == 0


# ============================================================================
# EMBEDDING ENGINE EDGE CASES
# ============================================================================

class TestEmbeddingEdgeCases:
    """Tests for embedding engine edge cases."""
    
    @pytest.fixture
    def embedding_engine(self):
        """Create embedding engine."""
        return EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    
    def test_embed_short_text(self, embedding_engine):
        """Test embedding very short text."""
        embedding = embedding_engine.embed("hi")
        assert embedding is not None
        assert len(embedding) == 384
    
    def test_embed_single_character(self, embedding_engine):
        """Test embedding single character."""
        embedding = embedding_engine.embed("a")
        assert embedding is not None
        assert len(embedding) == 384
    
    def test_embed_unicode(self, embedding_engine):
        """Test embedding Unicode text."""
        embedding = embedding_engine.embed("这是中文")
        assert embedding is not None
        assert len(embedding) == 384
    
    def test_embed_emoji(self, embedding_engine):
        """Test embedding emoji."""
        embedding = embedding_engine.embed("🚀🎉💻")
        assert embedding is not None
        assert len(embedding) == 384
    
    def test_embed_batch_empty(self, embedding_engine):
        """Test batch embedding with empty list."""
        embeddings = embedding_engine.embed_batch([])
        assert len(embeddings) == 0
    
    def test_embed_batch_single(self, embedding_engine):
        """Test batch embedding with single item."""
        embeddings = embedding_engine.embed_batch(["single text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
    
    def test_embed_batch_multiple(self, embedding_engine):
        """Test batch embedding with multiple items."""
        texts = ["first text", "second text", "third text"]
        embeddings = embedding_engine.embed_batch(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384
    
    def test_embed_normalization(self, embedding_engine):
        """Test that embeddings are normalized."""
        embedding = embedding_engine.embed("test text for normalization")
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be unit vector


# ============================================================================
# API UTILITY ENDPOINT EDGE CASES
# ============================================================================

class TestUtilityEdgeCases:
    """Tests for utility endpoint edge cases."""
    
    def test_stats_empty_database(self, client):
        """Test stats on empty/new database."""
        response = client.get("/search/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
        assert "total_edges" in data
        assert isinstance(data["total_nodes"], int)
        assert isinstance(data["total_edges"], int)
    
    def test_snapshot_endpoint(self, client):
        """Test snapshot creation."""
        response = client.post("/snapshot")
        assert response.status_code in [200, 501]  # May not be implemented
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data


# ============================================================================
# CONCURRENT ACCESS EDGE CASES
# ============================================================================

class TestConcurrencyEdgeCases:
    """Tests for concurrent access patterns."""
    
    def test_rapid_node_creation(self, client):
        """Test rapidly creating many nodes."""
        node_ids = []
        for i in range(20):
            response = client.post("/nodes", json={
                "text": f"Rapid node {i}",
                "metadata": {"index": i}
            })
            assert response.status_code == 201
            node_ids.append(response.json()["id"])
        
        # Verify all nodes exist
        for node_id in node_ids[:5]:  # Check first 5
            response = client.get(f"/nodes/{node_id}")
            assert response.status_code == 200
    
    def test_rapid_search(self, client):
        """Test rapid search operations."""
        for i in range(10):
            response = client.post("/search/vector", json={
                "query_text": f"search query {i}",
                "top_k": 5
            })
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

