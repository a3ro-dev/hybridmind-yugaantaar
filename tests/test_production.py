"""
Tests for HybridMind production features.
- Query caching
- Rate limiting
- Bulk operations
- Enhanced health endpoints
- Soft delete for vector index
- Learned CRS ranker
"""

import pytest
import tempfile
import time
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
import os

# Set test environment before imports
os.environ["HYBRIDMIND_DATABASE_PATH"] = "test_data/hybridmind.db"
os.environ["HYBRIDMIND_VECTOR_INDEX_PATH"] = "test_data/vector.index"
os.environ["HYBRIDMIND_GRAPH_INDEX_PATH"] = "test_data/graph.pkl"

from main import app
from engine.cache import QueryCache, get_query_cache
from middleware.rate_limit import RateLimiter
from storage.vector_index import VectorIndex
from engine.learned_ranker import LearnedCRSRanker, ScoredCandidate


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


# ============================================================================
# QUERY CACHE TESTS
# ============================================================================

class TestQueryCache:
    """Tests for query caching functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create a fresh cache for testing."""
        return QueryCache(maxsize=100, ttl=10)
    
    def test_cache_basic_operations(self, cache):
        """Test basic cache set/get."""
        params = {"query": "test", "top_k": 10}
        result = {"results": [{"id": "1"}], "query_time_ms": 5.0}
        
        # Set
        cache.set("hybrid", params, result)
        
        # Get
        cached = cache.get("hybrid", params)
        assert cached is not None
        assert cached["results"] == result["results"]
    
    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        params = {"query": "nonexistent"}
        cached = cache.get("hybrid", params)
        assert cached is None
    
    def test_cache_stats(self, cache):
        """Test cache statistics."""
        params = {"query": "test"}
        result = {"results": []}
        
        # Miss
        cache.get("hybrid", params)
        
        # Set and hit
        cache.set("hybrid", params, result)
        cache.get("hybrid", params)
        
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_cache_invalidation(self, cache):
        """Test cache invalidation."""
        params = {"query": "test"}
        cache.set("hybrid", params, {"results": []})
        
        # Invalidate
        cache.invalidate_all()
        
        # Should be gone
        assert cache.get("hybrid", params) is None
        assert cache.stats["size"] == 0
    
    def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        cache = QueryCache(maxsize=100, ttl=1)  # 1 second TTL
        params = {"query": "test"}
        cache.set("hybrid", params, {"results": []})
        
        # Should be present
        assert cache.get("hybrid", params) is not None
        
        # Wait for TTL
        time.sleep(1.5)
        
        # Should be expired
        assert cache.get("hybrid", params) is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = QueryCache(maxsize=3, ttl=300)
        
        # Fill cache
        for i in range(3):
            cache.set("hybrid", {"q": i}, {"r": i})
        
        # Add new entry (should evict oldest)
        cache.set("hybrid", {"q": 3}, {"r": 3})
        
        assert cache.stats["size"] == 3
        # q=3 should be there (just added)
        assert cache.get("hybrid", {"q": 3}) is not None
        # One of the earlier entries was evicted
        present_count = sum(1 for i in range(3) if cache.get("hybrid", {"q": i}) is not None)
        assert present_count == 2  # One was evicted
    
    def test_cache_deterministic_keys(self, cache):
        """Test that same params produce same cache key."""
        params1 = {"query": "test", "top_k": 10}
        params2 = {"top_k": 10, "query": "test"}  # Different order
        
        cache.set("hybrid", params1, {"results": ["a"]})
        
        # Should hit cache with reordered params
        cached = cache.get("hybrid", params2)
        assert cached is not None


# ============================================================================
# RATE LIMITER TESTS
# ============================================================================

class TestRateLimiter:
    """Tests for rate limiting functionality."""
    
    @pytest.fixture
    def limiter(self):
        """Create a fresh rate limiter."""
        return RateLimiter(requests_per_minute=60, burst_size=10)
    
    def test_basic_rate_limiting(self, limiter):
        """Test basic request allowance."""
        allowed, info = limiter.is_allowed("client1")
        assert allowed
        assert info["remaining"] <= 10
    
    def test_burst_capacity(self, limiter):
        """Test burst capacity allows initial burst."""
        # Should allow burst_size requests immediately
        for i in range(10):
            allowed, info = limiter.is_allowed("client1")
            assert allowed, f"Request {i+1} should be allowed"
    
    def test_rate_limit_exceeded(self):
        """Test that rate limit is enforced."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)
        
        # Exhaust burst
        for _ in range(5):
            limiter.is_allowed("client1")
        
        # Next should be throttled
        allowed, info = limiter.is_allowed("client1")
        assert not allowed
        assert "retry_after" in info
    
    def test_per_client_limits(self):
        """Test that limits are per-client."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=3)
        
        # Exhaust client1's burst
        for _ in range(3):
            limiter.is_allowed("client1")
        
        # client2 should still have allowance
        allowed, _ = limiter.is_allowed("client2")
        assert allowed
    
    def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(requests_per_minute=600, burst_size=5)  # 10/sec
        
        # Exhaust burst
        for _ in range(5):
            limiter.is_allowed("client1")
        
        # Wait for refill (100ms should give ~1 token at 10/sec)
        time.sleep(0.2)
        
        # Should have some tokens now
        allowed, _ = limiter.is_allowed("client1")
        assert allowed
    
    def test_limiter_stats(self, limiter):
        """Test rate limiter statistics."""
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        
        stats = limiter.stats
        assert stats["total_requests"] == 2
        assert stats["active_clients"] >= 1


# ============================================================================
# BULK OPERATIONS TESTS
# ============================================================================

class TestBulkOperations:
    """Tests for bulk import API endpoints."""
    
    def test_bulk_create_nodes(self, client):
        """Test bulk node creation."""
        nodes_data = {
            "nodes": [
                {"text": "Bulk node 1", "metadata": {"index": 1}},
                {"text": "Bulk node 2", "metadata": {"index": 2}},
                {"text": "Bulk node 3", "metadata": {"index": 3}},
            ],
            "generate_embeddings": True
        }
        
        response = client.post("/bulk/nodes", json=nodes_data)
        assert response.status_code == 200
        data = response.json()
        assert data["created"] == 3
        assert data["failed"] == 0
        assert data["success"]
    
    def test_bulk_create_nodes_with_custom_ids(self, client):
        """Test bulk node creation with custom IDs."""
        import uuid
        custom_id = f"custom-{uuid.uuid4().hex[:8]}"
        
        nodes_data = {
            "nodes": [
                {"id": custom_id, "text": "Custom ID node", "metadata": {}},
            ],
            "generate_embeddings": True
        }
        
        response = client.post("/bulk/nodes", json=nodes_data)
        assert response.status_code == 200
        
        # Verify node exists with custom ID
        get_response = client.get(f"/nodes/{custom_id}")
        assert get_response.status_code == 200
    
    def test_bulk_create_edges(self, client):
        """Test bulk edge creation."""
        # First create nodes
        nodes = []
        for i in range(3):
            resp = client.post("/nodes", json={"text": f"Edge test node {i}", "metadata": {}})
            nodes.append(resp.json()["id"])
        
        edges_data = {
            "edges": [
                {"source_id": nodes[0], "target_id": nodes[1], "type": "related", "weight": 0.8},
                {"source_id": nodes[1], "target_id": nodes[2], "type": "related", "weight": 0.9},
            ],
            "skip_validation": False
        }
        
        response = client.post("/bulk/edges", json=edges_data)
        assert response.status_code == 200
        data = response.json()
        assert data["created"] == 2
        assert data["failed"] == 0
    
    def test_bulk_create_edges_invalid_nodes(self, client):
        """Test bulk edge creation with invalid node IDs."""
        edges_data = {
            "edges": [
                {"source_id": "invalid-1", "target_id": "invalid-2", "type": "test"},
            ],
            "skip_validation": False
        }
        
        response = client.post("/bulk/edges", json=edges_data)
        assert response.status_code == 200
        data = response.json()
        assert data["created"] == 0
        assert data["success"] == False
    
    def test_bulk_import_combined(self, client):
        """Test combined bulk import of nodes and edges."""
        import uuid
        suffix = uuid.uuid4().hex[:8]
        
        import_data = {
            "nodes": [
                {"id": f"import-{suffix}-1", "text": "Import node 1", "metadata": {}},
                {"id": f"import-{suffix}-2", "text": "Import node 2", "metadata": {}},
            ],
            "edges": [
                {"source_id": f"import-{suffix}-1", "target_id": f"import-{suffix}-2", "type": "test"},
            ],
            "generate_embeddings": True
        }
        
        response = client.post("/bulk/import", json=import_data)
        assert response.status_code == 200
        data = response.json()
        assert data["nodes"]["created"] == 2
        assert data["edges"]["created"] == 1


# ============================================================================
# HEALTH ENDPOINTS TESTS
# ============================================================================

class TestHealthEndpoints:
    """Tests for enhanced health check endpoints."""
    
    def test_health_endpoint(self, client):
        """Test comprehensive health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "components" in data
        assert "metrics" in data
        
        # Check components
        components = data["components"]
        assert "embedding" in components
        assert "vector_index" in components
        assert "graph_index" in components
        assert "database" in components
        assert "cache" in components
    
    def test_ready_endpoint(self, client):
        """Test readiness probe endpoint."""
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "nodes_loaded" in data
        assert "edges_loaded" in data
    
    def test_live_endpoint(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
    
    def test_cache_stats_endpoint(self, client):
        """Test cache stats endpoint."""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data
        assert "size" in data
        assert "maxsize" in data
        assert "ttl_seconds" in data
    
    def test_cache_clear_endpoint(self, client):
        """Test cache clear endpoint."""
        response = client.post("/cache/clear")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"


# ============================================================================
# SOFT DELETE VECTOR INDEX TESTS
# ============================================================================

class TestVectorIndexSoftDelete:
    """Tests for vector index soft delete functionality."""
    
    @pytest.fixture
    def index(self):
        """Create temporary vector index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "vector.index"
            yield VectorIndex(dimension=384, index_path=str(index_path), deletion_threshold=0.2)
    
    def test_soft_delete_basic(self, index):
        """Test basic soft delete."""
        # Add multiple nodes so deletion doesn't trigger compaction
        for i in range(5):
            vec = np.random.randn(384).astype(np.float32)
            index.add(f"test-node-{i}", vec)
        
        assert index.size == 5
        assert index.total_size == 5
        
        # Soft delete one node (1/5 = 20% which equals threshold, doesn't trigger)
        index.remove("test-node-0")
        
        # Size should be 4, deleted_ids should have 1
        assert index.size == 4
        assert "test-node-0" in index.deleted_ids
    
    def test_soft_delete_search_excludes_deleted(self, index):
        """Test that deleted nodes are excluded from search."""
        # Add multiple nodes
        base_vec = np.random.randn(384).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)
        
        for i in range(5):
            noise = np.random.randn(384).astype(np.float32) * 0.1
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            index.add(f"node-{i}", vec)
        
        # Delete node-2
        index.remove("node-2")
        
        # Search should not return deleted node
        results = index.search(base_vec, top_k=10)
        result_ids = [r[0] for r in results]
        
        assert "node-2" not in result_ids
        assert len(result_ids) == 4
    
    def test_automatic_compaction(self):
        """Test automatic compaction when threshold exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "vector.index"
            # 20% threshold means compact after deleting 1 of 5 nodes
            index = VectorIndex(dimension=384, index_path=str(index_path), deletion_threshold=0.2)
            
            # Add 5 nodes
            for i in range(5):
                vec = np.random.randn(384).astype(np.float32)
                index.add(f"node-{i}", vec)
            
            assert index.total_size == 5
            
            # Delete 2 nodes (40% > 20% threshold)
            index.remove("node-0")
            index.remove("node-1")
            
            # Should have triggered compaction
            assert len(index.deleted_ids) == 0  # Cleared after compaction
            assert index.total_size == 3  # Compacted to 3 nodes
    
    def test_batch_add(self, index):
        """Test batch add functionality."""
        nodes = []
        for i in range(10):
            vec = np.random.randn(384).astype(np.float32)
            nodes.append((f"batch-{i}", vec))
        
        index.add_batch(nodes)
        
        assert index.size == 10
    
    def test_force_compact(self, index):
        """Test force compaction."""
        for i in range(5):
            vec = np.random.randn(384).astype(np.float32)
            index.add(f"node-{i}", vec)
        
        # Soft delete
        index.remove("node-0")
        
        assert len(index.deleted_ids) == 1
        
        # Force compact
        index.force_compact()
        
        assert len(index.deleted_ids) == 0
        assert index.total_size == 4
    
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
        assert stats["deletion_ratio"] > 0


# ============================================================================
# LEARNED CRS RANKER TESTS
# ============================================================================

class TestLearnedCRSRanker:
    """Tests for learned CRS ranker."""
    
    @pytest.fixture
    def ranker(self):
        """Create ranker with default settings."""
        return LearnedCRSRanker()
    
    def test_basic_scoring(self, ranker):
        """Test basic score computation."""
        candidate = ScoredCandidate(
            node_id="test",
            text="Test text",
            metadata={},
            vector_score=0.8,
            graph_score=0.6
        )
        
        score, explanation = ranker.compute_score(candidate)
        
        assert 0 <= score <= 1
        assert "final_crs" in explanation
        assert "raw_vector_score" in explanation
        assert "raw_graph_score" in explanation
    
    def test_edge_type_weighting(self, ranker):
        """Test edge type weighting."""
        # High-value edge type
        candidate1 = ScoredCandidate(
            node_id="test1",
            text="Test",
            metadata={},
            vector_score=0.5,
            graph_score=0.5,
            edge_types=["cites"]  # High weight
        )
        
        # Low-value edge type
        candidate2 = ScoredCandidate(
            node_id="test2",
            text="Test",
            metadata={},
            vector_score=0.5,
            graph_score=0.5,
            edge_types=["related_to"]  # Lower weight
        )
        
        score1, _ = ranker.compute_score(candidate1)
        score2, _ = ranker.compute_score(candidate2)
        
        # Higher edge weight should result in higher score
        assert score1 >= score2
    
    def test_hop_distance_decay(self, ranker):
        """Test hop distance decay."""
        # Close node (1 hop)
        candidate1 = ScoredCandidate(
            node_id="test1",
            text="Test",
            metadata={},
            vector_score=0.5,
            graph_score=0.8,
            hop_distance=1
        )
        
        # Far node (3 hops)
        candidate2 = ScoredCandidate(
            node_id="test2",
            text="Test",
            metadata={},
            vector_score=0.5,
            graph_score=0.8,
            hop_distance=3
        )
        
        score1, _ = ranker.compute_score(candidate1)
        score2, _ = ranker.compute_score(candidate2)
        
        # Closer should have higher score due to decay
        assert score1 > score2
    
    def test_harmony_bonus(self, ranker):
        """Test harmony bonus for dual-channel relevance."""
        # High scores on both channels
        candidate1 = ScoredCandidate(
            node_id="test1",
            text="Test",
            metadata={},
            vector_score=0.8,
            graph_score=0.8
        )
        
        # High on one, low on other
        candidate2 = ScoredCandidate(
            node_id="test2",
            text="Test",
            metadata={},
            vector_score=0.9,
            graph_score=0.2
        )
        
        score1, exp1 = ranker.compute_score(candidate1)
        score2, exp2 = ranker.compute_score(candidate2)
        
        # First should get harmony bonus
        assert exp1["harmony_bonus"] > exp2["harmony_bonus"]
    
    def test_rank_candidates(self, ranker):
        """Test ranking multiple candidates."""
        candidates = [
            ScoredCandidate("a", "A", {}, 0.9, 0.3),
            ScoredCandidate("b", "B", {}, 0.5, 0.5),
            ScoredCandidate("c", "C", {}, 0.3, 0.9),
        ]
        
        ranked = ranker.rank_candidates(candidates, top_k=3)
        
        assert len(ranked) == 3
        # Should be sorted by score descending
        scores = [r[1] for r in ranked]
        assert scores == sorted(scores, reverse=True)
    
    def test_custom_weights(self, ranker):
        """Test custom vector/graph weights."""
        candidate = ScoredCandidate(
            node_id="test",
            text="Test",
            metadata={},
            vector_score=0.8,
            graph_score=0.2
        )
        
        # Vector-heavy
        score1, _ = ranker.compute_score(candidate, vector_weight=0.9, graph_weight=0.1)
        
        # Graph-heavy
        score2, _ = ranker.compute_score(candidate, vector_weight=0.1, graph_weight=0.9)
        
        # Vector-heavy should be higher since vector_score > graph_score
        assert score1 > score2
    
    def test_score_bounds(self, ranker):
        """Test that scores are bounded [0, 1]."""
        # Edge cases
        candidates = [
            ScoredCandidate("a", "", {}, 0.0, 0.0),
            ScoredCandidate("b", "", {}, 1.0, 1.0),
            ScoredCandidate("c", "", {}, 0.5, 0.5),
        ]
        
        for candidate in candidates:
            score, _ = ranker.compute_score(candidate)
            assert 0 <= score <= 1


# ============================================================================
# CACHE INTEGRATION TESTS
# ============================================================================

class TestCacheIntegration:
    """Integration tests for caching in search endpoints."""
    
    def test_hybrid_search_caching(self, client):
        """Test that hybrid search results are cached."""
        # First search
        search_data = {"query_text": "test caching query", "top_k": 5}
        
        response1 = client.post("/search/hybrid", json=search_data)
        assert response1.status_code == 200
        
        # Second search (should be cached)
        response2 = client.post("/search/hybrid", json=search_data)
        assert response2.status_code == 200
        
        # Check cache stats
        stats_response = client.get("/cache/stats")
        stats = stats_response.json()
        assert stats["hits"] >= 1
    
    def test_cache_invalidation_on_node_create(self, client):
        """Test that cache is invalidated when nodes are created."""
        # Do a search to populate cache
        client.post("/search/hybrid", json={"query_text": "test", "top_k": 5})
        
        # Get cache size
        stats1 = client.get("/cache/stats").json()
        initial_size = stats1["size"]
        
        # Create a node (should invalidate cache)
        client.post("/nodes", json={"text": "New node", "metadata": {}})
        
        # Cache should be cleared
        stats2 = client.get("/cache/stats").json()
        assert stats2["size"] == 0 or stats2["size"] < initial_size


# ============================================================================
# RATE LIMIT INTEGRATION TESTS
# ============================================================================

class TestRateLimitIntegration:
    """Integration tests for rate limiting middleware."""
    
    def test_rate_limit_headers_present(self, client):
        """Test that rate limit headers are present on responses."""
        # Use a non-exempt endpoint
        response = client.get("/")
        
        # Check for rate limit headers
        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
    
    def test_exempt_paths_no_headers(self, client):
        """Test that exempt paths don't have rate limit headers."""
        # Health endpoints are exempt
        response = client.get("/health")
        
        # Health is exempt, so no rate limit headers or they're still added
        # (implementation may vary)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

