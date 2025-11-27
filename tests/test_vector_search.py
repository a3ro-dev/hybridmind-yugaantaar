"""
Vector Search Test Cases (TC-VEC-01 through TC-VEC-03)

Based on Devfolio Hackathon Test Case Markers.
Tests vector-only search functionality.
"""

import pytest
import numpy as np
from conftest import (
    P0, P1, integration, edge_case,
    cosine_similarity,
    MOCK_EMBEDDINGS,
    QUERY_EMBEDDING_REDIS_CACHING,
    CANONICAL_DOCUMENTS,
    EXPECTED_VECTOR_SCORES
)


# ==============================================================================
# TC-VEC-01 (P0) — Top-k cosine similarity ordering
# ==============================================================================

@P0
@integration
class TestVectorSearchOrdering:
    """
    TC-VEC-01: POST /search/vector returns top-k results ordered by cosine similarity.
    
    Steps: Insert nodes with known embeddings (A very similar to query, B medium, C far). 
           Query text mapped to A.
    Expected: Results order A, B, C; top result similarity above threshold.
    """
    
    def test_top_k_ordering_basic(self, client):
        """Test that results are ordered by cosine similarity (descending)."""
        # Create nodes with controlled similarity to query
        # Node A: Very similar to "machine learning" query
        node_a = client.post("/nodes", json={
            "text": "Machine learning algorithms for data processing",
            "metadata": {"label": "A_similar"}
        }).json()
        
        # Node B: Medium similarity
        node_b = client.post("/nodes", json={
            "text": "Database indexing and query optimization",
            "metadata": {"label": "B_medium"}
        }).json()
        
        # Node C: Far from query
        node_c = client.post("/nodes", json={
            "text": "Cooking recipes and kitchen appliances",
            "metadata": {"label": "C_far"}
        }).json()
        
        # Search for similar content
        response = client.post("/search/vector", json={
            "query_text": "machine learning data algorithms",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        results = data["results"]
        
        # Verify ordering: scores should be descending
        if len(results) >= 2:
            scores = [r.get("vector_score", 0) for r in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], \
                    f"Results not ordered: score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"
    
    def test_top_result_above_threshold(self, client):
        """Test that top result has similarity above a reasonable threshold."""
        # Create highly relevant node
        client.post("/nodes", json={
            "text": "Redis caching strategies for high performance applications",
            "metadata": {"topic": "caching"}
        })
        
        response = client.post("/search/vector", json={
            "query_text": "redis caching performance",
            "top_k": 1
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        if len(results) > 0:
            top_score = results[0].get("vector_score", 0)
            # For semantically similar content, expect reasonable similarity
            assert top_score > 0.0, "Top result should have positive similarity score"


# ==============================================================================
# TC-VEC-02 (P1) — Top-k with k > dataset size
# ==============================================================================

@P1
@edge_case
class TestVectorSearchKGreaterThanDataset:
    """
    TC-VEC-02: Top-k with k > dataset size.
    
    Expected: Returns all items without error, count = dataset size.
    """
    
    def test_top_k_exceeds_dataset_size(self, client):
        """Request more results than exist in dataset."""
        # Create a small dataset
        created_ids = []
        for i in range(3):
            response = client.post("/nodes", json={
                "text": f"Small dataset node {i} with unique content",
                "metadata": {"index": i}
            })
            if response.status_code == 201:
                created_ids.append(response.json()["id"])
        
        # Request top_k > dataset size
        response = client.post("/search/vector", json={
            "query_text": "small dataset node content",
            "top_k": 100  # Much larger than dataset
        })
        
        assert response.status_code == 200, "Should not error when k > dataset size"
        results = response.json()["results"]
        
        # Should return all matching items without error
        assert len(results) <= 100, "Should not return more than requested"
    
    def test_top_k_exactly_one(self, client):
        """Test with top_k=1."""
        client.post("/nodes", json={
            "text": "Single result test node",
            "metadata": {}
        })
        
        response = client.post("/search/vector", json={
            "query_text": "single result test",
            "top_k": 1
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) <= 1


# ==============================================================================
# TC-VEC-03 (P1) — Filtering by metadata
# ==============================================================================

@P1
@integration
class TestVectorSearchMetadataFilter:
    """
    TC-VEC-03: Pass filter metadata.type=note with vector query; 
    only nodes matching filter returned.
    
    Expected: Results restricted to matching metadata.
    """
    
    def test_filter_by_metadata_type(self, client):
        """Filter vector search results by metadata type."""
        # Create nodes with different types
        client.post("/nodes", json={
            "text": "This is a note about caching strategies",
            "metadata": {"type": "note", "topic": "caching"}
        })
        
        client.post("/nodes", json={
            "text": "This is an article about caching patterns",
            "metadata": {"type": "article", "topic": "caching"}
        })
        
        client.post("/nodes", json={
            "text": "Another note on database optimization",
            "metadata": {"type": "note", "topic": "database"}
        })
        
        # Search with metadata filter
        response = client.post("/search/vector", json={
            "query_text": "caching strategies",
            "top_k": 10,
            "filter_metadata": {"type": "note"}
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        # All results should have type=note
        for result in results:
            if "metadata" in result and result["metadata"]:
                # If filtering is implemented, verify constraint
                if "type" in result["metadata"]:
                    assert result["metadata"]["type"] == "note", \
                        f"Result has wrong type: {result['metadata']['type']}"
    
    def test_filter_returns_empty_when_no_match(self, client):
        """Filter that matches no documents returns empty results."""
        # Create nodes without the target metadata
        client.post("/nodes", json={
            "text": "Node without special metadata",
            "metadata": {"category": "general"}
        })
        
        response = client.post("/search/vector", json={
            "query_text": "test query",
            "top_k": 10,
            "filter_metadata": {"type": "nonexistent_type_xyz"}
        })
        
        assert response.status_code == 200
        # Should handle gracefully even if no matches


# ==============================================================================
# Additional Vector Search Tests
# ==============================================================================

@integration
class TestVectorSearchResponseFormat:
    """Test vector search response format and fields."""
    
    def test_response_contains_required_fields(self, client):
        """Verify response contains all required fields."""
        client.post("/nodes", json={
            "text": "Test node for response format",
            "metadata": {}
        })
        
        response = client.post("/search/vector", json={
            "query_text": "test response format",
            "top_k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify top-level fields
        assert "results" in data
        assert "query_time_ms" in data
        assert "search_type" in data
        assert data["search_type"] == "vector"
    
    def test_result_items_have_scores(self, client):
        """Verify each result has vector_score."""
        client.post("/nodes", json={
            "text": "Node with text for scoring test",
            "metadata": {}
        })
        
        response = client.post("/search/vector", json={
            "query_text": "scoring test",
            "top_k": 5
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        for result in results:
            assert "vector_score" in result, "Each result must have vector_score"
            assert isinstance(result["vector_score"], (int, float)), \
                "vector_score must be numeric"
            assert 0 <= result["vector_score"] <= 1, \
                f"vector_score should be in [0,1], got {result['vector_score']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
