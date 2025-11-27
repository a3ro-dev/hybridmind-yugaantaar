"""
API & CRUD Test Cases (TC-API-01 through TC-API-05)

Based on Devfolio Hackathon Test Case Markers.
Tests node and edge CRUD operations.
"""

import pytest
import numpy as np
from conftest import (
    P0, P1, integration,
    cosine_similarity,
    MOCK_EMBEDDINGS,
    CANONICAL_DOCUMENTS
)


# ==============================================================================
# TC-API-01 (P0) — Create node
# ==============================================================================

@P0
@integration
class TestCreateNode:
    """
    TC-API-01: POST /nodes creates a node with text, metadata and optional embedding.
    
    Expected: 201 Created; response contains id, stored text, metadata, and embedding 
    field present (generated or user-provided). GET /nodes/{id} returns same content.
    """
    
    def test_create_node_basic(self, client):
        """Create node with text and metadata."""
        response = client.post("/nodes", json={
            "text": "Venkat's note on caching",
            "metadata": {"type": "note", "author": "v"}
        })
        
        assert response.status_code == 201, f"Expected 201, got {response.status_code}"
        data = response.json()
        
        # Verify response structure
        assert "id" in data, "Response must contain 'id'"
        assert data["text"] == "Venkat's note on caching", "Text must match input"
        assert data["metadata"]["type"] == "note", "Metadata type must match"
        assert data["metadata"]["author"] == "v", "Metadata author must match"
    
    def test_create_node_verify_retrieval(self, client):
        """Create node and verify GET /nodes/{id} returns same content."""
        # Create node
        create_response = client.post("/nodes", json={
            "text": "Test content for retrieval verification",
            "metadata": {"test": True, "version": 1}
        })
        assert create_response.status_code == 201
        created = create_response.json()
        node_id = created["id"]
        
        # Retrieve and verify
        get_response = client.get(f"/nodes/{node_id}")
        assert get_response.status_code == 200
        retrieved = get_response.json()
        
        assert retrieved["id"] == node_id
        assert retrieved["text"] == "Test content for retrieval verification"
        assert retrieved["metadata"]["test"] == True
        assert retrieved["metadata"]["version"] == 1
    
    def test_create_node_with_custom_embedding(self, client):
        """Create node with user-provided embedding."""
        custom_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        response = client.post("/nodes", json={
            "text": "Node with custom embedding",
            "metadata": {},
            "embedding": custom_embedding
        })
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data


# ==============================================================================
# TC-API-02 (P0) — Read node with relationships
# ==============================================================================

@P0
@integration
class TestReadNodeWithRelationships:
    """
    TC-API-02: GET /nodes/{id} returns node properties plus outgoing/incoming relationships.
    
    Steps: Create two nodes A and B; create edge A->B. GET A.
    Expected: Response includes edge listing with type, target_id, and weight.
    """
    
    def test_read_node_with_edges(self, client):
        """Create nodes with edge and verify edge info is returned."""
        # Create node A
        node_a_response = client.post("/nodes", json={
            "text": "Node A - Source",
            "metadata": {"role": "source"}
        })
        assert node_a_response.status_code == 201
        node_a_id = node_a_response.json()["id"]
        
        # Create node B
        node_b_response = client.post("/nodes", json={
            "text": "Node B - Target",
            "metadata": {"role": "target"}
        })
        assert node_b_response.status_code == 201
        node_b_id = node_b_response.json()["id"]
        
        # Create edge A -> B
        edge_response = client.post("/edges", json={
            "source_id": node_a_id,
            "target_id": node_b_id,
            "type": "references",
            "weight": 0.75
        })
        assert edge_response.status_code == 201
        
        # GET node A and verify edges
        get_response = client.get(f"/nodes/{node_a_id}")
        assert get_response.status_code == 200
        node_data = get_response.json()
        
        # Verify edges are included
        assert "edges" in node_data, "Response must include 'edges' field"
        edges = node_data["edges"]
        assert len(edges) >= 1, "Node A should have at least one edge"
        
        # Find the edge we created
        found_edge = None
        for edge in edges:
            if edge["target_id"] == node_b_id:
                found_edge = edge
                break
        
        assert found_edge is not None, "Edge to node B not found"
        assert found_edge["type"] == "references", "Edge type must match"
        assert found_edge["weight"] == 0.75, "Edge weight must match"


# ==============================================================================
# TC-API-03 (P0) — Update node & re-generate embedding
# ==============================================================================

@P0
@integration
class TestUpdateNode:
    """
    TC-API-03: PUT /nodes/{id} updates text and triggers embedding regeneration when requested.
    
    Steps: PUT with new text and flag regen_embedding=true.
    Expected: 200 OK; embedding changed (cosine similarity < 0.99). GET returns updated text.
    """
    
    def test_update_node_text(self, client):
        """Update node text and verify change."""
        # Create node
        create_response = client.post("/nodes", json={
            "text": "Original text content",
            "metadata": {"version": 1}
        })
        assert create_response.status_code == 201
        node_id = create_response.json()["id"]
        
        # Update node
        update_response = client.put(f"/nodes/{node_id}", json={
            "text": "Updated text content - completely different",
            "metadata": {"version": 2},
            "regenerate_embedding": True
        })
        
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["text"] == "Updated text content - completely different"
        assert updated["metadata"]["version"] == 2
        
        # Verify via GET
        get_response = client.get(f"/nodes/{node_id}")
        assert get_response.status_code == 200
        retrieved = get_response.json()
        assert retrieved["text"] == "Updated text content - completely different"
    
    def test_update_nonexistent_node_returns_404(self, client):
        """Updating non-existent node should return 404."""
        response = client.put("/nodes/nonexistent-node-id-12345", json={
            "text": "New text",
            "regenerate_embedding": False
        })
        assert response.status_code == 404


# ==============================================================================
# TC-API-04 (P0) — Delete node cascading edges
# ==============================================================================

@P0
@integration
class TestDeleteNodeCascade:
    """
    TC-API-04: DELETE /nodes/{id} removes node and all associated edges.
    
    Steps: Create node with edges; DELETE node; GET node; GET edges.
    Expected: DELETE returns 204 or 200; subsequent GETs return 404 or empty.
    """
    
    def test_delete_node_cascades_edges(self, client):
        """Delete node and verify edges are also removed."""
        # Create nodes
        node_a = client.post("/nodes", json={"text": "Node A", "metadata": {}}).json()
        node_b = client.post("/nodes", json={"text": "Node B", "metadata": {}}).json()
        node_c = client.post("/nodes", json={"text": "Node C", "metadata": {}}).json()
        
        # Create edges: A -> B, A -> C
        edge1 = client.post("/edges", json={
            "source_id": node_a["id"],
            "target_id": node_b["id"],
            "type": "links_to",
            "weight": 0.5
        }).json()
        
        edge2 = client.post("/edges", json={
            "source_id": node_a["id"],
            "target_id": node_c["id"],
            "type": "links_to",
            "weight": 0.5
        }).json()
        
        # Delete node A
        delete_response = client.delete(f"/nodes/{node_a['id']}")
        assert delete_response.status_code == 200
        delete_data = delete_response.json()
        assert delete_data["deleted"] == True
        assert delete_data["edges_removed"] >= 2, "At least 2 edges should be removed"
        
        # Verify node A is gone
        get_node_response = client.get(f"/nodes/{node_a['id']}")
        assert get_node_response.status_code == 404
        
        # Verify edges are gone
        get_edge1_response = client.get(f"/edges/{edge1['id']}")
        assert get_edge1_response.status_code == 404
        
        get_edge2_response = client.get(f"/edges/{edge2['id']}")
        assert get_edge2_response.status_code == 404
    
    def test_delete_nonexistent_node_returns_404(self, client):
        """Deleting non-existent node should return 404."""
        response = client.delete("/nodes/nonexistent-node-id-12345")
        assert response.status_code == 404


# ==============================================================================
# TC-API-05 (P1) — Relationship CRUD
# ==============================================================================

@P1
@integration
class TestEdgeCRUD:
    """
    TC-API-05: POST /edges, GET /edges/{id}, update weight and delete.
    
    Expected: Edge lifecycle works; weight update reflected in traversal results.
    """
    
    def test_create_edge(self, client):
        """Create edge between two nodes."""
        # Create nodes
        node1 = client.post("/nodes", json={"text": "Source node", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Target node", "metadata": {}}).json()
        
        # Create edge
        response = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "cites",
            "weight": 0.9
        })
        
        assert response.status_code == 201
        edge = response.json()
        assert edge["source_id"] == node1["id"]
        assert edge["target_id"] == node2["id"]
        assert edge["type"] == "cites"
        assert edge["weight"] == 0.9
    
    def test_get_edge(self, client):
        """Retrieve edge by ID."""
        # Create nodes and edge
        node1 = client.post("/nodes", json={"text": "Node 1", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Node 2", "metadata": {}}).json()
        
        create_response = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "mentions",
            "weight": 0.7
        })
        edge_id = create_response.json()["id"]
        
        # Get edge
        get_response = client.get(f"/edges/{edge_id}")
        assert get_response.status_code == 200
        edge = get_response.json()
        assert edge["id"] == edge_id
        assert edge["type"] == "mentions"
        assert edge["weight"] == 0.7
    
    def test_update_edge_weight(self, client):
        """Update edge weight."""
        # Create nodes and edge
        node1 = client.post("/nodes", json={"text": "Node 1", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Node 2", "metadata": {}}).json()
        
        create_response = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "related_to",
            "weight": 0.5
        })
        edge_id = create_response.json()["id"]
        
        # Update weight
        update_response = client.put(f"/edges/{edge_id}", json={
            "weight": 0.95
        })
        
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["weight"] == 0.95
    
    def test_delete_edge(self, client):
        """Delete edge."""
        # Create nodes and edge
        node1 = client.post("/nodes", json={"text": "Node 1", "metadata": {}}).json()
        node2 = client.post("/nodes", json={"text": "Node 2", "metadata": {}}).json()
        
        create_response = client.post("/edges", json={
            "source_id": node1["id"],
            "target_id": node2["id"],
            "type": "test_edge"
        })
        edge_id = create_response.json()["id"]
        
        # Delete edge
        delete_response = client.delete(f"/edges/{edge_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted"] == True
        
        # Verify deleted
        get_response = client.get(f"/edges/{edge_id}")
        assert get_response.status_code == 404
    
    def test_create_edge_invalid_source_returns_404(self, client):
        """Creating edge with invalid source node should return 404."""
        node = client.post("/nodes", json={"text": "Valid node", "metadata": {}}).json()
        
        response = client.post("/edges", json={
            "source_id": "invalid-source-id",
            "target_id": node["id"],
            "type": "test"
        })
        
        assert response.status_code == 404
    
    def test_create_edge_invalid_target_returns_404(self, client):
        """Creating edge with invalid target node should return 404."""
        node = client.post("/nodes", json={"text": "Valid node", "metadata": {}}).json()
        
        response = client.post("/edges", json={
            "source_id": node["id"],
            "target_id": "invalid-target-id",
            "type": "test"
        })
        
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
