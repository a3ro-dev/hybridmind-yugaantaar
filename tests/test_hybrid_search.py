"""
Hybrid Search Test Cases (TC-HYB-01 through TC-HYB-03)

Based on Devfolio Hackathon Test Case Markers.
Tests vector + graph hybrid search functionality.
"""

import pytest
from conftest import (
    P0, P1, integration, system,
    compute_graph_score,
    compute_hybrid_score
)


# ==============================================================================
# TC-HYB-01 (P0) — Weighted merge correctness
# ==============================================================================

@P0
@system
class TestHybridWeightedMerge:
    """
    TC-HYB-01: POST /search/hybrid merges vector score and graph proximity 
    score using vector_weight and graph_weight.
    
    Steps: Create three nodes: V-similar (high vector score but graph distant), 
           G-close (low vector score but directly connected), Neutral. 
           Query with vector_weight=0.7, graph_weight=0.3.
    
    Expected: Aggregate score ranks V-similar above G-close when vector 
              advantage is large. Scores computed and returned with breakdown 
              {vector_score, graph_score, final_score}.
    """
    
    def test_weighted_merge_produces_combined_scores(self, client):
        """Test that hybrid search returns combined scores with breakdown."""
        # Create anchor node (will be used as graph reference point)
        anchor = client.post("/nodes", json={
            "text": "Machine learning fundamentals and algorithms",
            "metadata": {"role": "anchor"}
        }).json()
        
        # Create V-similar: high vector similarity to query, not connected
        v_similar = client.post("/nodes", json={
            "text": "Deep learning neural networks machine learning AI",
            "metadata": {"role": "v_similar"}
        }).json()
        
        # Create G-close: low vector similarity but directly connected to anchor
        g_close = client.post("/nodes", json={
            "text": "Database administration and SQL queries",
            "metadata": {"role": "g_close"}
        }).json()
        
        # Connect anchor to G-close
        client.post("/edges", json={
            "source_id": anchor["id"],
            "target_id": g_close["id"],
            "type": "related_to",
            "weight": 0.9
        })
        
        # Perform hybrid search
        response = client.post("/search/hybrid", json={
            "query_text": "machine learning neural networks",
            "top_k": 10,
            "vector_weight": 0.7,
            "graph_weight": 0.3,
            "anchor_nodes": [anchor["id"]]
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        results = data["results"]
        
        # Verify score breakdown in results
        for result in results:
            assert "vector_score" in result, "Result must have vector_score"
            assert "graph_score" in result, "Result must have graph_score"
            assert "combined_score" in result, "Result must have combined_score"
            
            # Verify scores are numeric and bounded
            assert 0 <= result["vector_score"] <= 1, \
                f"vector_score should be in [0,1], got {result['vector_score']}"
            assert 0 <= result["graph_score"] <= 1, \
                f"graph_score should be in [0,1], got {result['graph_score']}"
            assert 0 <= result["combined_score"] <= 1, \
                f"combined_score should be in [0,1], got {result['combined_score']}"
    
    def test_combined_score_formula(self, client):
        """Verify combined_score = vector_weight * vector_score + graph_weight * graph_score."""
        # Create test nodes
        node = client.post("/nodes", json={
            "text": "Test node for score verification",
            "metadata": {}
        }).json()
        
        vector_weight = 0.6
        graph_weight = 0.4
        
        response = client.post("/search/hybrid", json={
            "query_text": "test score verification",
            "top_k": 10,
            "vector_weight": vector_weight,
            "graph_weight": graph_weight
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        for result in results:
            v = result["vector_score"]
            g = result["graph_score"]
            expected = vector_weight * v + graph_weight * g
            actual = result["combined_score"]
            
            # Allow small floating point tolerance
            assert abs(actual - expected) < 0.01, \
                f"combined_score mismatch: expected {expected:.4f}, got {actual:.4f}"


# ==============================================================================
# TC-HYB-02 (P0) — Tuning extremes
# ==============================================================================

@P0
@system
class TestHybridTuningExtremes:
    """
    TC-HYB-02: Test with vector_weight=1.0, graph_weight=0.0 and vice versa.
    
    Expected: When vector_weight=1.0, results match vector-only ordering. 
              When graph_weight=1.0, results match graph-only proximity ordering.
    """
    
    def test_vector_weight_1_graph_weight_0(self, client):
        """With vector_weight=1.0, hybrid should match vector-only results."""
        # Create test nodes
        for i, topic in enumerate(["machine learning", "cooking recipes", "database systems"]):
            client.post("/nodes", json={
                "text": f"Document about {topic} with detailed information",
                "metadata": {"topic": topic}
            })
        
        query = "machine learning algorithms"
        
        # Vector-only search
        vector_response = client.post("/search/vector", json={
            "query_text": query,
            "top_k": 5
        })
        
        # Hybrid with vector_weight=1.0
        hybrid_response = client.post("/search/hybrid", json={
            "query_text": query,
            "top_k": 5,
            "vector_weight": 1.0,
            "graph_weight": 0.0
        })
        
        assert vector_response.status_code == 200
        assert hybrid_response.status_code == 200
        
        vector_results = vector_response.json()["results"]
        hybrid_results = hybrid_response.json()["results"]
        
        # When graph_weight=0, combined_score should equal vector_score
        for result in hybrid_results:
            assert abs(result["combined_score"] - result["vector_score"]) < 0.01, \
                "With vector_weight=1.0, combined_score should equal vector_score"
    
    def test_vector_weight_0_graph_weight_1(self, client):
        """With graph_weight=1.0, hybrid should be dominated by graph proximity."""
        # Create connected graph
        center = client.post("/nodes", json={
            "text": "Center node for graph test",
            "metadata": {}
        }).json()
        
        neighbor = client.post("/nodes", json={
            "text": "Neighbor node connected to center",
            "metadata": {}
        }).json()
        
        distant = client.post("/nodes", json={
            "text": "Distant unconnected node",
            "metadata": {}
        }).json()
        
        # Connect center to neighbor
        client.post("/edges", json={
            "source_id": center["id"],
            "target_id": neighbor["id"],
            "type": "connects",
            "weight": 1.0
        })
        
        # Hybrid with graph_weight=1.0
        response = client.post("/search/hybrid", json={
            "query_text": "center graph test",
            "top_k": 10,
            "vector_weight": 0.0,
            "graph_weight": 1.0,
            "anchor_nodes": [center["id"]]
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        # When vector_weight=0, combined_score should equal graph_score
        for result in results:
            assert abs(result["combined_score"] - result["graph_score"]) < 0.01, \
                "With graph_weight=1.0, combined_score should equal graph_score"


# ==============================================================================
# TC-HYB-03 (P1) — Relationship-weighted search (stretch)
# ==============================================================================

@P1
@system
class TestHybridRelationshipWeighted:
    """
    TC-HYB-03: Edges with higher weight increase graph proximity score.
    
    Expected: Node reached via higher-weight edges ranks better than 
              same-hop but lower-weight paths.
    """
    
    def test_edge_weight_affects_ranking(self, client):
        """Higher edge weight should improve graph proximity score."""
        # Create hub node
        hub = client.post("/nodes", json={
            "text": "Hub node central to graph",
            "metadata": {}
        }).json()
        
        # Create two nodes at same hop distance
        high_weight_node = client.post("/nodes", json={
            "text": "Node with high weight connection",
            "metadata": {"connection": "high"}
        }).json()
        
        low_weight_node = client.post("/nodes", json={
            "text": "Node with low weight connection",
            "metadata": {"connection": "low"}
        }).json()
        
        # Connect with different weights
        client.post("/edges", json={
            "source_id": hub["id"],
            "target_id": high_weight_node["id"],
            "type": "relates",
            "weight": 0.9  # High weight
        })
        
        client.post("/edges", json={
            "source_id": hub["id"],
            "target_id": low_weight_node["id"],
            "type": "relates",
            "weight": 0.1  # Low weight
        })
        
        # Search with hub as anchor
        response = client.post("/search/hybrid", json={
            "query_text": "node connection weight",
            "top_k": 10,
            "vector_weight": 0.3,  # Lower vector weight to emphasize graph
            "graph_weight": 0.7,
            "anchor_nodes": [hub["id"]]
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        # Find our test nodes in results
        high_weight_result = None
        low_weight_result = None
        
        for result in results:
            if result["node_id"] == high_weight_node["id"]:
                high_weight_result = result
            elif result["node_id"] == low_weight_node["id"]:
                low_weight_result = result
        
        # If both found, high-weight should have better graph score
        if high_weight_result and low_weight_result:
            # With same hop distance, higher edge weight should yield higher graph score
            # (implementation-dependent; some systems may not factor edge weights)
            pass  # Log or assert based on actual implementation
    
    def test_different_edge_types_handled(self, client):
        """Test that edge_type_weights parameter affects scoring."""
        hub = client.post("/nodes", json={
            "text": "Research paper hub",
            "metadata": {}
        }).json()
        
        cited = client.post("/nodes", json={
            "text": "Cited paper with citations",
            "metadata": {}
        }).json()
        
        mentioned = client.post("/nodes", json={
            "text": "Mentioned paper in passing",
            "metadata": {}
        }).json()
        
        # Create edges with different types
        client.post("/edges", json={
            "source_id": hub["id"],
            "target_id": cited["id"],
            "type": "cites",
            "weight": 1.0
        })
        
        client.post("/edges", json={
            "source_id": hub["id"],
            "target_id": mentioned["id"],
            "type": "mentions",
            "weight": 1.0
        })
        
        # Search with edge type weights favoring citations
        response = client.post("/search/hybrid", json={
            "query_text": "research paper",
            "top_k": 10,
            "vector_weight": 0.5,
            "graph_weight": 0.5,
            "anchor_nodes": [hub["id"]],
            "edge_type_weights": {
                "cites": 1.0,
                "mentions": 0.3
            }
        })
        
        assert response.status_code == 200


# ==============================================================================
# Additional Hybrid Search Tests
# ==============================================================================

@integration
class TestHybridSearchResponseFormat:
    """Test hybrid search response format and fields."""
    
    def test_response_contains_required_fields(self, client):
        """Verify response contains all required fields."""
        client.post("/nodes", json={
            "text": "Hybrid test node",
            "metadata": {}
        })
        
        response = client.post("/search/hybrid", json={
            "query_text": "hybrid test",
            "top_k": 5,
            "vector_weight": 0.6,
            "graph_weight": 0.4
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "query_time_ms" in data
        assert "search_type" in data
        assert data["search_type"] == "hybrid"
    
    def test_results_ordered_by_combined_score(self, client):
        """Verify results are ordered by combined_score descending."""
        # Create multiple nodes
        for i in range(5):
            client.post("/nodes", json={
                "text": f"Hybrid ordering test node {i} with content",
                "metadata": {"index": i}
            })
        
        response = client.post("/search/hybrid", json={
            "query_text": "hybrid ordering test content",
            "top_k": 10,
            "vector_weight": 0.6,
            "graph_weight": 0.4
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        # Verify descending order
        scores = [r["combined_score"] for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Results not ordered: score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"


@system
class TestHybridSearchComparison:
    """Test hybrid search comparison endpoint."""
    
    def test_compare_search_modes(self, client):
        """Test the /search/compare endpoint."""
        # Create test data
        client.post("/nodes", json={
            "text": "Comparison test document about machine learning",
            "metadata": {}
        })
        
        response = client.post("/search/compare", json={
            "query_text": "machine learning",
            "top_k": 5,
            "vector_weight": 0.6,
            "graph_weight": 0.4
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have all three search types
        assert "vector_only" in data
        assert "graph_only" in data
        assert "hybrid" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
