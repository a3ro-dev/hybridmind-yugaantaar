"""
Comparison Engine - Query all three databases and compare results.
HybridMind vs Neo4j vs ChromaDB
"""

import sys
import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Ensure hybridmind is in path
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_parent)
sys.path.insert(0, _root)
sys.path.insert(0, _parent)


@dataclass
class SearchResult:
    """Standardized search result."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Results from a single database."""
    database: str
    latency_ms: float
    results: List[SearchResult]
    success: bool
    error: Optional[str] = None


class ComparisonEngine:
    """Engine to compare HybridMind, Neo4j, and ChromaDB."""
    
    def __init__(self):
        self._hybridmind = None
        self._neo4j_driver = None
        self._chromadb = None
        self._embedding_engine = None
    
    def _get_hybridmind(self):
        """Lazy load HybridMind."""
        if self._hybridmind is None:
            from hybridmind.api.dependencies import get_db_manager
            self._hybridmind = get_db_manager()
            self._embedding_engine = self._hybridmind.embedding_engine
        return self._hybridmind
    
    def _get_neo4j(self):
        """Lazy load Neo4j connection."""
        if self._neo4j_driver is None:
            try:
                from neo4j import GraphDatabase
                self._neo4j_driver = GraphDatabase.driver(
                    "bolt://localhost:7687",
                    auth=("neo4j", "password")
                )
                self._neo4j_driver.verify_connectivity()
            except Exception as e:
                self._neo4j_driver = "error"
                self._neo4j_error = str(e)
        return self._neo4j_driver
    
    def _get_chromadb(self):
        """Lazy load ChromaDB."""
        if self._chromadb is None:
            try:
                import chromadb
                # Use absolute path relative to this file
                chromadb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chromadb")
                client = chromadb.PersistentClient(path=chromadb_path)
                self._chromadb = client.get_collection("arxiv_papers")
            except Exception as e:
                self._chromadb = "error"
                self._chromadb_error = str(e)
        return self._chromadb
    
    def search_hybridmind(self, query: str, top_k: int = 10, 
                          mode: str = "hybrid") -> BenchmarkResult:
        """Search HybridMind (our hybrid database)."""
        try:
            db = self._get_hybridmind()
            
            start = time.perf_counter()
            
            if mode == "vector":
                results, latency, _ = db.vector_engine.search(query, top_k)
                score_key = "vector_score"
            elif mode == "graph":
                # For graph, we need a start node - use vector search first
                vec_results, _, _ = db.vector_engine.search(query, 1)
                if vec_results:
                    start_id = vec_results[0]["node_id"]
                    results, latency, _ = db.graph_engine.traverse(start_id, depth=2)
                    score_key = "graph_score"
                else:
                    results = []
                    score_key = "graph_score"
            else:  # hybrid
                results, latency, _ = db.hybrid_ranker.search(query, top_k)
                score_key = "combined_score"
            
            elapsed = (time.perf_counter() - start) * 1000
            
            search_results = [
                SearchResult(
                    id=r["node_id"],
                    text=r.get("text", "")[:200],
                    score=r.get(score_key, r.get("score", 0)),
                    metadata=r.get("metadata", {})
                )
                for r in results[:top_k]
            ]
            
            return BenchmarkResult(
                database=f"HybridMind ({mode})",
                latency_ms=elapsed,
                results=search_results,
                success=True
            )
        
        except Exception as e:
            return BenchmarkResult(
                database=f"HybridMind ({mode})",
                latency_ms=0,
                results=[],
                success=False,
                error=str(e)
            )
    
    def search_neo4j(self, query: str, top_k: int = 10) -> BenchmarkResult:
        """Search Neo4j (graph-only, using text matching)."""
        try:
            driver = self._get_neo4j()
            
            if driver == "error":
                return BenchmarkResult(
                    database="Neo4j (graph)",
                    latency_ms=0,
                    results=[],
                    success=False,
                    error=getattr(self, '_neo4j_error', 'Connection failed')
                )
            
            # Extract keywords from query
            keywords = query.lower().split()[:3]
            
            start = time.perf_counter()
            
            with driver.session() as session:
                # Search by text content and get connected nodes
                result = session.run("""
                    MATCH (p:Paper)
                    WHERE ANY(word IN $keywords WHERE toLower(p.text) CONTAINS word)
                    OPTIONAL MATCH (p)-[r:RELATES]-(connected:Paper)
                    WITH p, COUNT(DISTINCT connected) as connections
                    RETURN p.id as id, p.title as title, p.text as text, 
                           p.category as category, connections
                    ORDER BY connections DESC
                    LIMIT $limit
                """, keywords=keywords, limit=top_k)
                
                records = list(result)
            
            elapsed = (time.perf_counter() - start) * 1000
            
            search_results = [
                SearchResult(
                    id=r["id"],
                    text=r["text"][:200] if r["text"] else "",
                    score=r["connections"] / 10.0,  # Normalize
                    metadata={"title": r["title"], "category": r["category"]}
                )
                for r in records
            ]
            
            return BenchmarkResult(
                database="Neo4j (graph)",
                latency_ms=elapsed,
                results=search_results,
                success=True
            )
        
        except Exception as e:
            return BenchmarkResult(
                database="Neo4j (graph)",
                latency_ms=0,
                results=[],
                success=False,
                error=str(e)
            )
    
    def search_chromadb(self, query: str, top_k: int = 10) -> BenchmarkResult:
        """Search ChromaDB (vector-only)."""
        try:
            collection = self._get_chromadb()
            
            if collection == "error":
                return BenchmarkResult(
                    database="ChromaDB (vector)",
                    latency_ms=0,
                    results=[],
                    success=False,
                    error=getattr(self, '_chromadb_error', 'Collection not found')
                )
            
            # Get embedding for query
            if self._embedding_engine is None:
                self._get_hybridmind()
            
            query_embedding = self._embedding_engine.embed(query).tolist()
            
            start = time.perf_counter()
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            elapsed = (time.perf_counter() - start) * 1000
            
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    # Convert L2 distance to similarity score (1 / (1 + distance))
                    score = 1.0 / (1.0 + distance)
                    
                    search_results.append(SearchResult(
                        id=doc_id,
                        text=results["documents"][0][i][:200] if results["documents"] else "",
                        score=round(score, 3),
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {}
                    ))
            
            return BenchmarkResult(
                database="ChromaDB (vector)",
                latency_ms=elapsed,
                results=search_results,
                success=True
            )
        
        except Exception as e:
            return BenchmarkResult(
                database="ChromaDB (vector)",
                latency_ms=0,
                results=[],
                success=False,
                error=str(e)
            )
    
    def compare_all(self, query: str, top_k: int = 10) -> Dict[str, BenchmarkResult]:
        """Run search on all three databases and compare."""
        return {
            "hybridmind": self.search_hybridmind(query, top_k, mode="hybrid"),
            "hybridmind_vector": self.search_hybridmind(query, top_k, mode="vector"),
            "neo4j": self.search_neo4j(query, top_k),
            "chromadb": self.search_chromadb(query, top_k)
        }
    
    def benchmark(self, queries: List[str], top_k: int = 10, 
                  iterations: int = 3) -> Dict[str, Dict[str, float]]:
        """Run benchmark across all databases."""
        results = {
            "hybridmind_hybrid": [],
            "hybridmind_vector": [],
            "neo4j": [],
            "chromadb": []
        }
        
        for _ in range(iterations):
            for query in queries:
                # HybridMind Hybrid
                r = self.search_hybridmind(query, top_k, mode="hybrid")
                if r.success:
                    results["hybridmind_hybrid"].append(r.latency_ms)
                
                # HybridMind Vector
                r = self.search_hybridmind(query, top_k, mode="vector")
                if r.success:
                    results["hybridmind_vector"].append(r.latency_ms)
                
                # Neo4j
                r = self.search_neo4j(query, top_k)
                if r.success:
                    results["neo4j"].append(r.latency_ms)
                
                # ChromaDB
                r = self.search_chromadb(query, top_k)
                if r.success:
                    results["chromadb"].append(r.latency_ms)
        
        # Calculate stats
        stats = {}
        for db, latencies in results.items():
            if latencies:
                stats[db] = {
                    "mean": sum(latencies) / len(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "count": len(latencies)
                }
            else:
                stats[db] = {"mean": 0, "min": 0, "max": 0, "count": 0}
        
        return stats


# Singleton instance
_comparison_engine = None

def get_comparison_engine() -> ComparisonEngine:
    """Get singleton comparison engine."""
    global _comparison_engine
    if _comparison_engine is None:
        _comparison_engine = ComparisonEngine()
    return _comparison_engine

