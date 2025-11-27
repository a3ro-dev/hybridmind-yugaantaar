"""
Tests for HybridMind API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
import tempfile
import os

# Set test environment before imports
os.environ["HYBRIDMIND_DATABASE_PATH"] = "test_data/hybridmind.db"
os.environ["HYBRIDMIND_VECTOR_INDEX_PATH"] = "test_data/vector.index"
os.environ["HYBRIDMIND_GRAPH_INDEX_PATH"] = "test_data/graph.pkl"

from hybridmind.main import app


@pytest.fixture
def client():
    """Create test client."""
    client = TestClient(app)
    yield client


class TestRootEndpoints:
    """Tests for root endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "HybridMind"
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code in [200, 503]


class TestNodeEndpoints:
    """Tests for node CRUD endpoints."""
    
    def test_create_node(self, client):
        """Test node creation."""
        response = client.post("/nodes", json={
            "text": "Test node content for API testing",
            "metadata": {"title": "Test Node", "tags": ["test"]}
        })
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["text"] == "Test node content for API testing"
    
    def test_get_node(self, client):
        """Test node retrieval."""
        # First create a node
        create_response = client.post("/nodes", json={
            "text": "Node to retrieve",
            "metadata": {}
        })
        node_id = create_response.json()["id"]
        
        # Then retrieve it
        response = client.get(f"/nodes/{node_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == node_id
    
    def test_get_nonexistent_node(self, client):
        """Test getting nonexistent node."""
        response = client.get("/nodes/nonexistent-id-12345")
        assert response.status_code == 404
    
    def test_update_node(self, client):
        """Test node update."""
        # Create node
        create_response = client.post("/nodes", json={
            "text": "Original text",
            "metadata": {"version": 1}
        })
        node_id = create_response.json()["id"]
        
        # Update node
        response = client.put(f"/nodes/{node_id}", json={
            "text": "Updated text",
            "metadata": {"version": 2},
            "regenerate_embedding": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Updated text"
        assert data["metadata"]["version"] == 2
    
    def test_delete_node(self, client):
        """Test node deletion."""
        # Create node
        create_response = client.post("/nodes", json={
            "text": "Node to delete",
            "metadata": {}
        })
        node_id = create_response.json()["id"]
        
        # Delete node
        response = client.delete(f"/nodes/{node_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] == True
        
        # Verify deleted
        get_response = client.get(f"/nodes/{node_id}")
        assert get_response.status_code == 404
    
    def test_list_nodes(self, client):
        """Test listing nodes."""
        response = client.get("/nodes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestEdgeEndpoints:
    """Tests for edge CRUD endpoints."""
    
    @pytest.fixture
    def two_nodes(self, client):
        """Create two nodes for edge testing."""
        node1 = client.post("/nodes", json={"text": "Node 1", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Node 2", "metadata": {}}).json()
        return node1["id"], node2["id"]
    
    def test_create_edge(self, client, two_nodes):
        """Test edge creation."""
        source_id, target_id = two_nodes
        
        response = client.post("/edges", json={
            "source_id": source_id,
            "target_id": target_id,
            "type": "relates_to",
            "weight": 0.8
        })
        
        assert response.status_code == 201
        data = response.json()
        assert data["source_id"] == source_id
        assert data["target_id"] == target_id
        assert data["type"] == "relates_to"
    
    def test_create_edge_invalid_nodes(self, client):
        """Test edge creation with invalid nodes."""
        response = client.post("/edges", json={
            "source_id": "invalid-1",
            "target_id": "invalid-2",
            "type": "test"
        })
        assert response.status_code == 404
    
    def test_get_edge(self, client, two_nodes):
        """Test edge retrieval."""
        source_id, target_id = two_nodes
        
        # Create edge
        create_response = client.post("/edges", json={
            "source_id": source_id,
            "target_id": target_id,
            "type": "test"
        })
        edge_id = create_response.json()["id"]
        
        # Get edge
        response = client.get(f"/edges/{edge_id}")
        assert response.status_code == 200
    
    def test_delete_edge(self, client, two_nodes):
        """Test edge deletion."""
        source_id, target_id = two_nodes
        
        # Create edge
        create_response = client.post("/edges", json={
            "source_id": source_id,
            "target_id": target_id,
            "type": "test"
        })
        edge_id = create_response.json()["id"]
        
        # Delete edge
        response = client.delete(f"/edges/{edge_id}")
        assert response.status_code == 200


class TestSearchEndpoints:
    """Tests for search endpoints."""
    
    def test_vector_search(self, client):
        """Test vector search."""
        # Add some nodes first
        client.post("/nodes", json={
            "text": "Machine learning and artificial intelligence",
            "metadata": {}
        })
        
        response = client.post("/search/vector", json={
            "query_text": "AI machine learning",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query_time_ms" in data
    
    def test_hybrid_search(self, client):
        """Test hybrid search."""
        response = client.post("/search/hybrid", json={
            "query_text": "deep learning neural networks",
            "top_k": 5,
            "vector_weight": 0.6,
            "graph_weight": 0.4
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["search_type"] == "hybrid"
    
    def test_graph_search(self, client):
        """Test graph search."""
        # Create nodes and edge
        node1 = client.post("/nodes", json={"text": "Start node", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "End node", "metadata": {}}).json()
        client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "test"
        })
        
        response = client.get("/search/graph", params={
            "start_id": node1["id"],
            "depth": 2
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["search_type"] == "graph"
    
    def test_stats(self, client):
        """Test stats endpoint."""
        response = client.get("/search/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
        assert "total_edges" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

