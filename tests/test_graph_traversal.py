"""
Graph Traversal Test Cases (TC-GRAPH-01 through TC-GRAPH-03)

Based on Devfolio Hackathon Test Case Markers.
Tests graph-only search functionality.
"""

import pytest
from conftest import (
    P0, P1, integration, edge_case,
    compute_graph_score
)


# ==============================================================================
# TC-GRAPH-01 (P0) — BFS / depth-limited traversal
# ==============================================================================

@P0
@integration
class TestGraphTraversalBFS:
    """
    TC-GRAPH-01: GET /search/graph?start_id=X&depth=2 returns reachable nodes up to depth.
    
    Steps: Build chain A->B->C->D; query depth=2 from A.
    Expected: Returns B and C (depth 1 and 2), not D.
    """
    
    def test_depth_limited_traversal(self, client):
        """Test BFS traversal respects depth limit."""
        # Create chain: A -> B -> C -> D
        node_a = client.post("/nodes", json={
            "text": "Node A - Start",
            "metadata": {"position": "start"}
        }).json()
        
        node_b = client.post("/nodes", json={
            "text": "Node B - Depth 1",
            "metadata": {"position": "middle"}
        }).json()
        
        node_c = client.post("/nodes", json={
            "text": "Node C - Depth 2",
            "metadata": {"position": "middle"}
        }).json()
        
        node_d = client.post("/nodes", json={
            "text": "Node D - Depth 3",
            "metadata": {"position": "end"}
        }).json()
        
        # Create edges forming chain
        client.post("/edges", json={
            "source_id": node_a["id"],
            "target_id": node_b["id"],
            "type": "connects_to",
            "weight": 1.0
        })
        
        client.post("/edges", json={
            "source_id": node_b["id"],
            "target_id": node_c["id"],
            "type": "connects_to",
            "weight": 1.0
        })
        
        client.post("/edges", json={
            "source_id": node_c["id"],
            "target_id": node_d["id"],
            "type": "connects_to",
            "weight": 1.0
        })
        
        # Query with depth=2 from A
        response = client.get("/search/graph", params={
            "start_id": node_a["id"],
            "depth": 2
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        results = data["results"]
        
        # Extract node IDs from results
        result_ids = [r["node_id"] for r in results]
        
        # B should be found (depth 1)
        assert node_b["id"] in result_ids, "Node B (depth 1) should be in results"
        
        # C should be found (depth 2)
        assert node_c["id"] in result_ids, "Node C (depth 2) should be in results"
        
        # D should NOT be found (depth 3 exceeds limit)
        assert node_d["id"] not in result_ids, \
            "Node D (depth 3) should NOT be in results when depth=2"
    
    def test_traversal_depth_1(self, client):
        """Test traversal with depth=1 returns only immediate neighbors."""
        # Create star topology: Center -> A, B, C
        center = client.post("/nodes", json={
            "text": "Center node",
            "metadata": {}
        }).json()
        
        neighbors = []
        for label in ["A", "B", "C"]:
            node = client.post("/nodes", json={
                "text": f"Neighbor {label}",
                "metadata": {"label": label}
            }).json()
            neighbors.append(node)
            
            client.post("/edges", json={
                "source_id": center["id"],
                "target_id": node["id"],
                "type": "connects",
                "weight": 1.0
            })
        
        # Create second-degree connection: A -> X
        node_x = client.post("/nodes", json={
            "text": "Second degree X",
            "metadata": {}
        }).json()
        
        client.post("/edges", json={
            "source_id": neighbors[0]["id"],
            "target_id": node_x["id"],
            "type": "connects",
            "weight": 1.0
        })
        
        # Query with depth=1
        response = client.get("/search/graph", params={
            "start_id": center["id"],
            "depth": 1
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        result_ids = [r["node_id"] for r in results]
        
        # All immediate neighbors should be found
        for neighbor in neighbors:
            assert neighbor["id"] in result_ids, \
                f"Immediate neighbor should be in depth=1 results"
        
        # X should NOT be found (depth 2)
        assert node_x["id"] not in result_ids, \
            "Second-degree node should NOT be in depth=1 results"


# ==============================================================================
# TC-GRAPH-02 (P1) — Multi-type relationships
# ==============================================================================

@P1
@integration
class TestGraphTraversalEdgeTypeFilter:
    """
    TC-GRAPH-02: Graph traversal respects relationship type filtering.
    
    Expected: When filtered by type=author_of, only edges of that type followed.
    """
    
    def test_filter_by_edge_type(self, client):
        """Test traversal filtered by specific edge type."""
        # Create nodes
        author = client.post("/nodes", json={
            "text": "Author node",
            "metadata": {"role": "author"}
        }).json()
        
        paper1 = client.post("/nodes", json={
            "text": "Paper 1 written by author",
            "metadata": {"type": "paper"}
        }).json()
        
        paper2 = client.post("/nodes", json={
            "text": "Paper 2 written by author",
            "metadata": {"type": "paper"}
        }).json()
        
        cited_paper = client.post("/nodes", json={
            "text": "Paper cited by author",
            "metadata": {"type": "paper"}
        }).json()
        
        # Create edges with different types
        client.post("/edges", json={
            "source_id": author["id"],
            "target_id": paper1["id"],
            "type": "author_of",
            "weight": 1.0
        })
        
        client.post("/edges", json={
            "source_id": author["id"],
            "target_id": paper2["id"],
            "type": "author_of",
            "weight": 1.0
        })
        
        client.post("/edges", json={
            "source_id": author["id"],
            "target_id": cited_paper["id"],
            "type": "cites",
            "weight": 1.0
        })
        
        # Query with edge_type filter
        response = client.get("/search/graph", params={
            "start_id": author["id"],
            "depth": 1,
            "edge_types": "author_of"
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        result_ids = [r["node_id"] for r in results]
        
        # Papers with author_of edges should be found
        assert paper1["id"] in result_ids, "Paper 1 (author_of) should be found"
        assert paper2["id"] in result_ids, "Paper 2 (author_of) should be found"
        
        # Cited paper should NOT be found (different edge type)
        assert cited_paper["id"] not in result_ids, \
            "Cited paper should NOT be found when filtering by author_of"


# ==============================================================================
# TC-GRAPH-03 (P1) — Cycle handling
# ==============================================================================

@P1
@edge_case
class TestGraphTraversalCycleHandling:
    """
    TC-GRAPH-03: Graph has cycles A->B->A; traversal must not infinite-loop.
    
    Expected: Nodes visited once; traversal terminates.
    """
    
    def test_cycle_does_not_cause_infinite_loop(self, client):
        """Test that cycles are handled without infinite loops."""
        # Create cycle: A -> B -> C -> A
        node_a = client.post("/nodes", json={
            "text": "Cycle Node A",
            "metadata": {"position": "A"}
        }).json()
        
        node_b = client.post("/nodes", json={
            "text": "Cycle Node B",
            "metadata": {"position": "B"}
        }).json()
        
        node_c = client.post("/nodes", json={
            "text": "Cycle Node C",
            "metadata": {"position": "C"}
        }).json()
        
        # Create cycle
        client.post("/edges", json={
            "source_id": node_a["id"],
            "target_id": node_b["id"],
            "type": "next",
            "weight": 1.0
        })
        
        client.post("/edges", json={
            "source_id": node_b["id"],
            "target_id": node_c["id"],
            "type": "next",
            "weight": 1.0
        })
        
        client.post("/edges", json={
            "source_id": node_c["id"],
            "target_id": node_a["id"],
            "type": "next",
            "weight": 1.0
        })
        
        # Query with depth=5 (more than cycle length)
        # This should terminate without hanging
        response = client.get("/search/graph", params={
            "start_id": node_a["id"],
            "depth": 5
        })
        
        # If we get here, we didn't infinite loop
        assert response.status_code == 200
        results = response.json()["results"]
        
        # Should find B and C (but each only once)
        result_ids = [r["node_id"] for r in results]
        
        # B and C should be present
        assert node_b["id"] in result_ids, "Node B should be found"
        assert node_c["id"] in result_ids, "Node C should be found"
        
        # Each node should appear at most once
        assert result_ids.count(node_b["id"]) == 1, \
            "Node B should appear exactly once (no duplicates from cycle)"
        assert result_ids.count(node_c["id"]) == 1, \
            "Node C should appear exactly once (no duplicates from cycle)"
    
    def test_self_loop_handled(self, client):
        """Test that self-referencing edges don't cause issues."""
        # Create node with self-loop potential
        node = client.post("/nodes", json={
            "text": "Self-referencing node",
            "metadata": {}
        }).json()
        
        # Some systems may not allow self-loops; if they do, test handling
        self_edge_response = client.post("/edges", json={
            "source_id": node["id"],
            "target_id": node["id"],
            "type": "self_ref",
            "weight": 1.0
        })
        
        # If self-loop was created, traversal should still work
        if self_edge_response.status_code == 201:
            response = client.get("/search/graph", params={
                "start_id": node["id"],
                "depth": 3
            })
            
            assert response.status_code == 200, \
                "Traversal should complete even with self-loops"


# ==============================================================================
# Additional Graph Search Tests
# ==============================================================================

@integration
class TestGraphSearchResponseFormat:
    """Test graph search response format and fields."""
    
    def test_response_contains_required_fields(self, client):
        """Verify response contains search_type=graph."""
        node = client.post("/nodes", json={
            "text": "Graph search test node",
            "metadata": {}
        }).json()
        
        response = client.get("/search/graph", params={
            "start_id": node["id"],
            "depth": 1
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "search_type" in data
        assert data["search_type"] == "graph"
    
    def test_results_include_depth_info(self, client):
        """Verify traversal results include hop/depth information."""
        # Create simple A -> B structure
        node_a = client.post("/nodes", json={
            "text": "Node A",
            "metadata": {}
        }).json()
        
        node_b = client.post("/nodes", json={
            "text": "Node B",
            "metadata": {}
        }).json()
        
        client.post("/edges", json={
            "source_id": node_a["id"],
            "target_id": node_b["id"],
            "type": "links",
            "weight": 1.0
        })
        
        response = client.get("/search/graph", params={
            "start_id": node_a["id"],
            "depth": 2
        })
        
        assert response.status_code == 200
        results = response.json()["results"]
        
        # Results should include depth/hop information
        for result in results:
            # Check for depth or hop field
            assert "depth" in result or "graph_score" in result, \
                "Results should include depth or graph_score"
    
    def test_nonexistent_start_node_returns_404(self, client):
        """Test that traversal from non-existent node returns 404."""
        response = client.get("/search/graph", params={
            "start_id": "nonexistent-node-id-xyz",
            "depth": 2
        })
        
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
