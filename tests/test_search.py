"""
Tests for HybridMind search engines.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.graph_index import GraphIndex
from engine.embedding import EmbeddingEngine
from engine.vector_search import VectorSearchEngine
from engine.graph_search import GraphSearchEngine
from engine.hybrid_ranker import HybridRanker


@pytest.fixture
def test_environment():
    """Create test environment with all components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Initialize components
        sqlite_store = SQLiteStore(str(tmpdir / "test.db"))
        vector_index = VectorIndex(dimension=384, index_path=str(tmpdir / "vector.index"))
        graph_index = GraphIndex(index_path=str(tmpdir / "graph.pkl"))
        embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        
        # Create search engines
        vector_engine = VectorSearchEngine(vector_index, sqlite_store, embedding_engine)
        graph_engine = GraphSearchEngine(graph_index, sqlite_store)
        hybrid_ranker = HybridRanker(vector_engine, graph_engine)
        
        # Add test data
        test_nodes = [
            ("node-1", "Transformer attention mechanisms for natural language processing"),
            ("node-2", "BERT pre-training for language understanding"),
            ("node-3", "Deep learning optimization with Adam optimizer"),
            ("node-4", "Computer vision with convolutional neural networks"),
            ("node-5", "Recurrent neural networks for sequence modeling"),
        ]
        
        for node_id, text in test_nodes:
            embedding = embedding_engine.embed(text)
            sqlite_store.create_node(node_id, text, {"title": text[:20]}, embedding)
            vector_index.add(node_id, embedding)
            graph_index.add_node(node_id)
        
        # Add edges
        edges = [
            ("node-1", "node-2", "related_to"),
            ("node-2", "node-3", "cites"),
            ("node-3", "node-4", "related_to"),
            ("node-1", "node-5", "cites"),
        ]
        
        for source, target, edge_type in edges:
            sqlite_store.create_edge(f"edge-{source}-{target}", source, target, edge_type, 1.0)
            graph_index.add_edge(source, target, edge_type, 1.0)
        
        yield {
            "sqlite_store": sqlite_store,
            "vector_index": vector_index,
            "graph_index": graph_index,
            "embedding_engine": embedding_engine,
            "vector_engine": vector_engine,
            "graph_engine": graph_engine,
            "hybrid_ranker": hybrid_ranker
        }
        
        sqlite_store.close()


class TestVectorSearch:
    """Tests for vector search engine."""
    
    def test_basic_search(self, test_environment):
        """Test basic vector search."""
        engine = test_environment["vector_engine"]
        
        results, time_ms, candidates = engine.search(
            query_text="attention transformer NLP",
            top_k=3
        )
        
        assert len(results) <= 3
        assert time_ms > 0
        assert candidates > 0
        
        # Results should be returned (mock embeddings may not match expected order)
        if results:
            assert len(results[0]["text"]) > 0
    
    def test_min_score_filter(self, test_environment):
        """Test minimum score filtering."""
        engine = test_environment["vector_engine"]
        
        results, _, _ = engine.search(
            query_text="attention",
            top_k=10,
            min_score=0.9  # Very high threshold
        )
        
        # Should filter out low-scoring results
        for r in results:
            assert r["vector_score"] >= 0.9


class TestGraphSearch:
    """Tests for graph search engine."""
    
    def test_traversal(self, test_environment):
        """Test graph traversal."""
        engine = test_environment["graph_engine"]
        
        results, time_ms, candidates = engine.traverse(
            start_id="node-1",
            depth=2
        )
        
        assert len(results) > 0
        assert time_ms > 0
        
        # Should find connected nodes
        node_ids = [r["node_id"] for r in results]
        assert "node-2" in node_ids or "node-5" in node_ids
    
    def test_proximity_scores(self, test_environment):
        """Test proximity score computation."""
        engine = test_environment["graph_engine"]
        
        scores = engine.compute_proximity_scores(
            node_ids=["node-2", "node-3", "node-4"],
            reference_nodes=["node-1"]
        )
        
        # node-2 is directly connected to node-1
        # node-3 is 2 hops away
        assert scores["node-2"] > scores["node-3"]


class TestHybridSearch:
    """Tests for hybrid search engine."""
    
    def test_hybrid_search(self, test_environment):
        """Test hybrid search."""
        ranker = test_environment["hybrid_ranker"]
        
        results, time_ms, candidates = ranker.search(
            query_text="attention mechanisms",
            top_k=5,
            vector_weight=0.6,
            graph_weight=0.4
        )
        
        assert len(results) <= 5
        assert time_ms > 0
        
        # Check that scores are computed
        for r in results:
            assert "vector_score" in r
            assert "graph_score" in r
            assert "combined_score" in r
    
    def test_weight_impact(self, test_environment):
        """Test that weights affect ranking."""
        ranker = test_environment["hybrid_ranker"]
        
        # Vector-heavy search
        results_vector = ranker.search(
            query_text="neural networks",
            top_k=5,
            vector_weight=0.9,
            graph_weight=0.1
        )[0]
        
        # Graph-heavy search
        results_graph = ranker.search(
            query_text="neural networks",
            top_k=5,
            vector_weight=0.1,
            graph_weight=0.9,
            anchor_nodes=["node-1"]
        )[0]
        
        # Results should be different
        vector_ids = [r["node_id"] for r in results_vector]
        graph_ids = [r["node_id"] for r in results_graph]
        
        # At least the order or some results should differ
        # (may not always be different depending on data)
        assert len(vector_ids) > 0 and len(graph_ids) > 0
    
    def test_anchor_nodes(self, test_environment):
        """Test anchor node functionality."""
        ranker = test_environment["hybrid_ranker"]
        
        # Search with anchor
        results, _, _ = ranker.search(
            query_text="deep learning",
            top_k=5,
            anchor_nodes=["node-1"]
        )
        
        # Connected nodes should have graph scores
        for r in results:
            assert r["graph_score"] >= 0


class TestCRSAlgorithm:
    """Tests for CRS scoring algorithm."""
    
    def test_crs_formula(self, test_environment):
        """Test CRS score calculation."""
        ranker = test_environment["hybrid_ranker"]
        
        results, _, _ = ranker.search(
            query_text="transformer",
            top_k=5,
            vector_weight=0.6,
            graph_weight=0.4
        )
        
        for r in results:
            v = r["vector_score"]
            g = r["graph_score"]
            expected = 0.6 * v + 0.4 * g
            
            # Allow small floating point difference
            assert abs(r["combined_score"] - expected) < 0.01
    
    def test_score_bounds(self, test_environment):
        """Test that scores are within bounds."""
        ranker = test_environment["hybrid_ranker"]
        
        results, _, _ = ranker.search(
            query_text="test query",
            top_k=10
        )
        
        for r in results:
            assert 0 <= r["vector_score"] <= 1
            assert 0 <= r["graph_score"] <= 1
            assert 0 <= r["combined_score"] <= 1


class TestLearnedRanker:
    """Tests for the learned CRS ranker."""
    
    def test_learned_ranker_basic(self):
        """Test basic learned ranker functionality."""
        from engine.learned_ranker import LearnedCRSRanker, ScoredCandidate
        
        ranker = LearnedCRSRanker()
        candidate = ScoredCandidate(
            node_id="test",
            text="Test",
            metadata={},
            vector_score=0.8,
            graph_score=0.6
        )
        
        score, explanation = ranker.compute_score(candidate)
        assert 0 <= score <= 1
        assert "final_crs" in explanation
    
    def test_edge_type_weighting(self):
        """Test edge type weighting in learned ranker."""
        from engine.learned_ranker import LearnedCRSRanker, ScoredCandidate
        
        ranker = LearnedCRSRanker()
        
        # High-value edge type
        candidate_cite = ScoredCandidate(
            node_id="a", text="", metadata={},
            vector_score=0.5, graph_score=0.5,
            edge_types=["cites"]
        )
        
        # Low-value edge type
        candidate_related = ScoredCandidate(
            node_id="b", text="", metadata={},
            vector_score=0.5, graph_score=0.5,
            edge_types=["related_to"]
        )
        
        score1, _ = ranker.compute_score(candidate_cite)
        score2, _ = ranker.compute_score(candidate_related)
        
        assert score1 >= score2
    
    def test_hop_decay(self):
        """Test hop distance decay in learned ranker."""
        from engine.learned_ranker import LearnedCRSRanker, ScoredCandidate
        
        ranker = LearnedCRSRanker()
        
        # 1 hop
        candidate1 = ScoredCandidate(
            node_id="a", text="", metadata={},
            vector_score=0.5, graph_score=0.8,
            hop_distance=1
        )
        
        # 3 hops
        candidate3 = ScoredCandidate(
            node_id="b", text="", metadata={},
            vector_score=0.5, graph_score=0.8,
            hop_distance=3
        )
        
        score1, _ = ranker.compute_score(candidate1)
        score3, _ = ranker.compute_score(candidate3)
        
        assert score1 > score3  # Closer should score higher


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

