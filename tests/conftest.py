"""
Shared fixtures and test configuration for HybridMind test suite.

Based on Devfolio Hackathon Test Case Markers.
"""

import pytest
import os
import sys
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment before imports
os.environ["HYBRIDMIND_DATABASE_PATH"] = "test_data/hybridmind_test.db"
os.environ["HYBRIDMIND_VECTOR_INDEX_PATH"] = "test_data/vector_test.index"
os.environ["HYBRIDMIND_GRAPH_INDEX_PATH"] = "test_data/graph_test.pkl"

from fastapi.testclient import TestClient
from main import app


# ==============================================================================
# CANONICAL MOCK EMBEDDINGS (6-dimensional for testing)
# ==============================================================================

MOCK_EMBEDDINGS = {
    "doc1": np.array([0.90, 0.10, 0.00, 0.00, 0.00, 0.00], dtype=np.float32),  # Redis caching strategies
    "doc2": np.array([0.70, 0.10, 0.60, 0.00, 0.00, 0.00], dtype=np.float32),  # RedisGraph module
    "doc3": np.array([0.10, 0.05, 0.00, 0.90, 0.00, 0.00], dtype=np.float32),  # Distributed systems
    "doc4": np.array([0.80, 0.15, 0.00, 0.00, 0.00, 0.00], dtype=np.float32),  # Cache invalidation
    "doc5": np.array([0.05, 0.00, 0.90, 0.10, 0.00, 0.00], dtype=np.float32),  # Graph algorithms
    "doc6": np.array([0.60, 0.05, 0.50, 0.00, 0.10, 0.00], dtype=np.float32),  # README: Redis+Graph
}

# Query embedding for "redis caching"
QUERY_EMBEDDING_REDIS_CACHING = np.array([0.88, 0.12, 0.02, 0.00, 0.00, 0.00], dtype=np.float32)


# ==============================================================================
# CANONICAL DOCUMENT TEXTS
# ==============================================================================

CANONICAL_DOCUMENTS = {
    "doc1": {
        "title": "Redis caching strategies",
        "text": """Redis became the default choice for caching mostly because people like avoiding slow databases.
There are the usual headaches: eviction policies like LRU vs LFU, memory pressure, and when
someone forgets to set TTLs and wonders why servers fall over. A funny incident last month:
our checkout service kept missing prices because a stale cache key survived a deploy.""",
        "metadata": {"type": "article", "tags": ["cache", "redis"], "author": "alice"}
    },
    "doc2": {
        "title": "RedisGraph module",
        "text": """The RedisGraph module promises a weird marriage: pretend your cache is also a graph database.
Honestly, it works better than expected. You can store relationships like user -> viewed -> product
and then still query it with cypher-like syntax. Someone even built a PageRank demo over it.""",
        "metadata": {"type": "article", "tags": ["redis", "graph"]}
    },
    "doc3": {
        "title": "Distributed systems",
        "text": """Distributed systems are basically long-distance relationships. Nodes drift apart, messages get lost,
and during network partitions everyone blames everyone else. Leader election decides who gets
boss privileges until the next heartbeat timeout. Caching across a cluster is especially fun because
one stale node ruins the whole party.""",
        "metadata": {"type": "article", "tags": ["distributed", "systems"]}
    },
    "doc4": {
        "title": "Cache invalidation note",
        "text": """A short note on cache invalidation: you think you understand it until your application grows. Patterns
like write-through, write-behind, and cache-aside all behave differently under load. Versioned keys
help, but someone will always ship code that forgets to update them. The universe trends toward chaos.""",
        "metadata": {"type": "note", "tags": ["cache"]}
    },
    "doc5": {
        "title": "Graph algorithms",
        "text": """Graph algorithms show up in real life more than people notice. Social feeds rely on BFS for exploring
connections, recommendations rely on random walks, and PageRank still refuses to die. Even your
team's on-call rotation effectively forms a directed cycle, complete with its own failure modes.""",
        "metadata": {"type": "article", "tags": ["graph"]}
    },
    "doc6": {
        "title": "README: Redis+Graph",
        "text": """README draft: to combine Redis with a graph database, you start by defining nodes for each entity,
like articles, users, or configuration snippets. Then you create edges describing interactions: mentions,
references, imports, or even blame (use sparingly). The magic happens when semantic search embeddings
overlay this structure and suddenly the system feels smarter than it is.""",
        "metadata": {"type": "readme", "tags": ["redis", "graph"]}
    }
}


# ==============================================================================
# CANONICAL EDGES
# ==============================================================================

CANONICAL_EDGES = [
    {"source": "doc1", "target": "doc4", "type": "related_to", "weight": 0.8},
    {"source": "doc2", "target": "doc6", "type": "mentions", "weight": 0.9},
    {"source": "doc6", "target": "doc1", "type": "references", "weight": 0.6},
    {"source": "doc3", "target": "doc5", "type": "related_to", "weight": 0.5},
    {"source": "doc2", "target": "doc5", "type": "example_of", "weight": 0.3},
]


# ==============================================================================
# EXPECTED SCORES (Pre-calculated for canonical dataset)
# ==============================================================================

# Expected vector scores for "redis caching" query
EXPECTED_VECTOR_SCORES = {
    "doc1": 0.99943737,
    "doc4": 0.99712011,
    "doc2": 0.77237251,
    "doc6": 0.66474701,
    "doc5": 0.02237546,
}

# Graph proximity: graph_score = 1 / (1 + hops)
# hop=0 → 1.0, hop=1 → 0.5, hop=2 → 0.333, unreachable → 0.0


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def client():
    """Create test client for API tests."""
    return TestClient(app)


@pytest.fixture
def clean_client():
    """Create a fresh test client and clean up test data after use."""
    client = TestClient(app)
    yield client
    # Cleanup can be done here if needed


@pytest.fixture
def mock_embedding_6d():
    """Generate a random 6-dimensional mock embedding."""
    def _generate():
        vec = np.random.randn(6).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()
    return _generate


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_graph_score(hops: int) -> float:
    """Compute graph proximity score: 1 / (1 + hops)."""
    if hops < 0:
        return 0.0
    return 1.0 / (1.0 + hops)


def compute_hybrid_score(vector_score: float, graph_score: float, 
                         vector_weight: float = 0.6, graph_weight: float = 0.4) -> float:
    """Compute hybrid CRS score."""
    return vector_weight * vector_score + graph_weight * graph_score


# ==============================================================================
# TEST MARKERS
# ==============================================================================

# Priority markers for test categorization
P0 = pytest.mark.priority_p0  # Blocking
P1 = pytest.mark.priority_p1  # Important
P2 = pytest.mark.priority_p2  # Nice-to-have

# Scope markers
integration = pytest.mark.integration
system = pytest.mark.system
edge_case = pytest.mark.edge_case
performance = pytest.mark.performance
