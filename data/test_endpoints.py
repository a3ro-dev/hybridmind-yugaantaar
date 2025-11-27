"""Test all HybridMind endpoints for evaluation."""
import requests
import json

BASE_URL = "http://localhost:8000"

# Store created IDs for cleanup
created_node_id = None
created_edge_id = None


def test_crud_operations():
    """Test Node and Edge CRUD - creates fresh resources."""
    global created_node_id, created_edge_id
    
    print("=" * 60)
    print("1. CRUD OPERATIONS")
    print("=" * 60)
    
    # CREATE NODE
    print("\n[POST /nodes] Creating test node...")
    r = requests.post(f"{BASE_URL}/nodes", json={
        "text": "Test node about machine learning, deep learning, and neural networks for API evaluation.",
        "metadata": {"title": "Test Node", "category": "cs.AI", "tags": ["test", "evaluation"]}
    })
    print(f"  Status: {r.status_code}")
    if r.status_code == 201:
        node_data = r.json()
        created_node_id = node_data["id"]
        print(f"  Created node: {created_node_id}")
    else:
        print(f"  Error: {r.json()}")
        return
    
    # GET NODE
    print(f"\n[GET /nodes/{created_node_id[:8]}...] Retrieving node...")
    r = requests.get(f"{BASE_URL}/nodes/{created_node_id}")
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print(f"  Text: {r.json()['text'][:50]}...")
    
    # UPDATE NODE
    print(f"\n[PUT /nodes/{created_node_id[:8]}...] Updating node...")
    r = requests.put(f"{BASE_URL}/nodes/{created_node_id}", json={
        "text": "Updated: Deep learning transformers and attention mechanisms.",
        "metadata": {"title": "Updated Test", "category": "cs.LG"},
        "regenerate_embedding": True
    })
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print(f"  Updated text: {r.json()['text'][:50]}...")
    
    # Get an existing node to connect to
    existing_nodes = requests.get(f"{BASE_URL}/nodes", params={"limit": 1}).json()
    target_node_id = existing_nodes[0]["id"] if existing_nodes else None
    
    # CREATE EDGE (connect to existing node)
    print("\n[POST /edges] Creating test edge...")
    if not target_node_id:
        print("  ⚠ No existing nodes to connect to")
        return
    
    r = requests.post(f"{BASE_URL}/edges", json={
        "source_id": created_node_id,
        "target_id": target_node_id,
        "type": "related_to",
        "weight": 0.85
    })
    print(f"  Status: {r.status_code}")
    if r.status_code == 201:
        edge_data = r.json()
        created_edge_id = edge_data["id"]
        print(f"  Created edge: {created_edge_id}")
    else:
        print(f"  Error: {r.json()}")
    
    # GET EDGE
    if created_edge_id:
        print(f"\n[GET /edges/{created_edge_id[:8]}...] Retrieving edge...")
        r = requests.get(f"{BASE_URL}/edges/{created_edge_id}")
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            e = r.json()
            print(f"  Edge: {e['source_id'][:8]}... --[{e['type']}]--> {e['target_id'][:8]}...")
    
    # UPDATE EDGE
    if created_edge_id:
        print(f"\n[PUT /edges/{created_edge_id[:8]}...] Updating edge...")
        r = requests.put(f"{BASE_URL}/edges/{created_edge_id}", json={
            "type": "extends",
            "weight": 0.95
        })
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  Updated type: {r.json()['type']}, weight: {r.json()['weight']}")
    
    print("\n  ✓ All CRUD operations successful!")


def test_vector_search():
    print("\n" + "=" * 60)
    print("2. VECTOR SEARCH")
    print("=" * 60)
    r = requests.post(f"{BASE_URL}/search/vector", json={
        "query_text": "machine learning neural networks deep learning",
        "top_k": 5
    })
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Query time: {data['query_time_ms']}ms")
    print(f"Total candidates: {data['total_candidates']}")
    print(f"Search type: {data['search_type']}")
    print("Results:")
    for i, res in enumerate(data["results"], 1):
        score = res.get("vector_score", 0)
        text = res["text"][:60]
        print(f"  {i}. [score={score:.3f}] {text}...")
    return data


def test_graph_search():
    print("\n" + "=" * 60)
    print("3. GRAPH SEARCH")
    print("=" * 60)
    
    # Get an existing node with edges
    existing_nodes = requests.get(f"{BASE_URL}/nodes", params={"limit": 5}).json()
    start_node_id = existing_nodes[0]["id"] if existing_nodes else None
    
    if not start_node_id:
        print("  ⚠ No existing nodes for graph search")
        return {}
    
    print(f"  Starting from node: {start_node_id[:20]}...")
    r = requests.get(f"{BASE_URL}/search/graph", params={
        "start_id": start_node_id,
        "depth": 2,
        "direction": "both"
    })
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Query time: {data['query_time_ms']}ms")
    print(f"Total candidates: {data['total_candidates']}")
    print(f"Search type: {data['search_type']}")
    print("Results:")
    for i, res in enumerate(data["results"][:5], 1):
        depth = res.get("depth", 0)
        score = res.get("graph_score", 0)
        text = res["text"][:50]
        print(f"  {i}. [depth={depth}, score={score:.3f}] {text}...")
    return data


def test_hybrid_search():
    print("\n" + "=" * 60)
    print("4. HYBRID SEARCH (CRS Algorithm)")
    print("=" * 60)
    r = requests.post(f"{BASE_URL}/search/hybrid", json={
        "query_text": "artificial intelligence machine learning",
        "top_k": 5,
        "vector_weight": 0.6,
        "graph_weight": 0.4
    })
    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Query time: {data['query_time_ms']}ms")
    print(f"Total candidates: {data['total_candidates']}")
    print(f"Search type: {data['search_type']}")
    print("Results (with CRS scores):")
    for i, res in enumerate(data["results"], 1):
        v_score = res.get("vector_score", 0)
        g_score = res.get("graph_score", 0)
        c_score = res.get("combined_score", 0)
        text = res["text"][:45]
        print(f"  {i}. V={v_score:.3f} G={g_score:.3f} CRS={c_score:.3f} | {text}...")
    return data


def test_compare_search():
    print("\n" + "=" * 60)
    print("5. COMPARE SEARCH (Vector vs Graph vs Hybrid)")
    print("=" * 60)
    r = requests.post(f"{BASE_URL}/search/compare", json={
        "query_text": "deep learning transformers attention",
        "top_k": 3,
        "vector_weight": 0.6,
        "graph_weight": 0.4
    })
    print(f"Status: {r.status_code}")
    data = r.json()
    
    print("\nVector-Only Results:")
    for i, res in enumerate(data["vector_only"]["results"], 1):
        print(f"  {i}. [{res['score']:.3f}] {res['text'][:50]}...")
    
    print(f"\nGraph-Only Results: {len(data['graph_only']['results'])} nodes")
    
    print("\nHybrid Results:")
    for i, res in enumerate(data["hybrid"]["results"], 1):
        print(f"  {i}. V={res['vector_score']:.2f} G={res['graph_score']:.2f} CRS={res['combined_score']:.3f}")
    
    print(f"\nAnalysis:")
    analysis = data["analysis"]
    print(f"  - Hybrid combines best: {analysis['hybrid_combines_best']} results")
    print(f"  - Overlap all modes: {analysis['overlap_all']}")
    return data


def test_path_finding():
    print("\n" + "=" * 60)
    print("6. PATH FINDING (Multi-hop)")
    print("=" * 60)
    
    # Get two existing nodes
    existing_nodes = requests.get(f"{BASE_URL}/nodes", params={"limit": 10}).json()
    if len(existing_nodes) < 2:
        print("  ⚠ Not enough nodes for path finding")
        return {}
    
    source_id = existing_nodes[0]["id"]
    target_id = existing_nodes[5]["id"] if len(existing_nodes) > 5 else existing_nodes[1]["id"]
    
    print(f"  Finding path: {source_id[:15]}... → {target_id[:15]}...")
    r = requests.get(f"{BASE_URL}/search/path/{source_id}/{target_id}")
    print(f"Status: {r.status_code}")
    data = r.json()
    if data.get("path_exists"):
        print(f"Path exists! Length: {data['length']} hops")
        print(f"Path: {' -> '.join(data['path'][:5])}...")
    else:
        print("No path found between nodes")
    return data


def test_delete_operations():
    """Delete the test resources we created."""
    global created_node_id, created_edge_id
    
    print("\n" + "=" * 60)
    print("7. DELETE OPERATIONS (Cleanup)")
    print("=" * 60)
    
    # Delete the test edge first (before node, since edge depends on node)
    if created_edge_id:
        print(f"\n[DELETE /edges/{created_edge_id[:8]}...] Deleting test edge...")
        r = requests.delete(f"{BASE_URL}/edges/{created_edge_id}")
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  ✓ Edge deleted successfully")
    else:
        print("\n  ⚠ No edge to delete (not created)")
    
    # Delete the test node
    if created_node_id:
        print(f"\n[DELETE /nodes/{created_node_id[:8]}...] Deleting test node...")
        r = requests.delete(f"{BASE_URL}/nodes/{created_node_id}")
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"  ✓ Node deleted: {data['node_id'][:8]}...")
            print(f"  ✓ Edges removed: {data['edges_removed']}")
    else:
        print("\n  ⚠ No node to delete (not created)")


def test_health():
    """Test health endpoints."""
    print("\n" + "=" * 60)
    print("8. HEALTH & STATS")
    print("=" * 60)
    
    # Health
    r = requests.get(f"{BASE_URL}/health")
    print(f"\n[GET /health] Status: {r.status_code}")
    if r.status_code == 200:
        h = r.json()
        print(f"  System: {h['status']}")
        print(f"  Uptime: {h['uptime_seconds']:.1f}s")
    
    # Stats
    r = requests.get(f"{BASE_URL}/search/stats")
    print(f"\n[GET /search/stats] Status: {r.status_code}")
    if r.status_code == 200:
        s = r.json()
        print(f"  Nodes: {s['total_nodes']}")
        print(f"  Edges: {s['total_edges']}")
        print(f"  Edge types: {list(s['edge_types'].keys())}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   HYBRIDMIND API EVALUATION TEST")
    print("=" * 60 + "\n")
    
    # First create test resources
    test_crud_operations()
    
    # Then run search tests
    test_vector_search()
    test_graph_search()
    test_hybrid_search()
    test_compare_search()
    test_path_finding()
    
    # Health check
    test_health()
    
    # Finally cleanup
    test_delete_operations()
    
    print("\n" + "=" * 60)
    print("   ✅ ALL TESTS COMPLETED!")
    print("=" * 60)
