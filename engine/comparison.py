"""
Comparison Engine for HybridMind vs Neo4j vs ChromaDB.
Provides unified interface to query all three systems for benchmarking.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result from a single database system."""
    system: str
    results: List[Dict[str, Any]]
    latency_ms: float
    total_candidates: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMetrics:
    """Performance metrics for comparison."""
    latency_ms: float
    throughput_qps: float
    recall_at_k: float
    precision_at_k: float
    overlap_with_ground_truth: int
    unique_results: int


class Neo4jClient:
    """
    Neo4j graph database client for comparison.
    Pure graph-based retrieval using Cypher queries.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._available = None
    
    @property
    def driver(self):
        """Lazy load Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password)
                )
                self._driver.verify_connectivity()
                self._available = True
                logger.info(f"Neo4j connected at {self.uri}")
            except ImportError:
                logger.warning("neo4j package not installed")
                self._available = False
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}")
                self._available = False
        return self._driver
    
    @property
    def is_available(self) -> bool:
        """Check if Neo4j is available."""
        if self._available is None:
            _ = self.driver  # Trigger connection attempt
        return self._available or False
    
    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10
    ) -> ComparisonResult:
        """
        Full-text search in Neo4j (no vector similarity).
        Uses text index or CONTAINS for matching.
        """
        if not self.is_available:
            return ComparisonResult(
                system="neo4j",
                results=[],
                latency_ms=0,
                total_candidates=0,
                error="Neo4j not available"
            )
        
        start = time.perf_counter()
        
        try:
            with self.driver.session() as session:
                # Use full-text search or basic text matching
                # Searching by keywords in the text
                keywords = query_text.lower().split()[:5]  # Top 5 keywords
                
                # Build search condition
                conditions = " OR ".join([
                    f"toLower(p.text) CONTAINS '{kw}'" 
                    for kw in keywords
                ])
                
                result = session.run(f"""
                    MATCH (p:Paper)
                    WHERE {conditions}
                    WITH p, 
                         reduce(score = 0, kw IN {keywords} | 
                            score + CASE WHEN toLower(p.text) CONTAINS kw THEN 1 ELSE 0 END
                         ) AS match_score
                    WHERE match_score > 0
                    RETURN p.id AS id, p.title AS title, p.text AS text, 
                           p.category AS category, match_score
                    ORDER BY match_score DESC
                    LIMIT $top_k
                """, top_k=top_k, keywords=keywords)
                
                results = []
                for record in result:
                    results.append({
                        "node_id": record["id"],
                        "text": record["text"][:500] if record["text"] else "",
                        "metadata": {
                            "title": record["title"],
                            "category": record["category"]
                        },
                        "score": record["match_score"] / len(keywords) if keywords else 0
                    })
                
                latency = (time.perf_counter() - start) * 1000
                
                return ComparisonResult(
                    system="neo4j",
                    results=results,
                    latency_ms=latency,
                    total_candidates=len(results),
                    metadata={"search_type": "full_text", "keywords": keywords}
                )
                
        except Exception as e:
            logger.error(f"Neo4j search error: {e}")
            return ComparisonResult(
                system="neo4j",
                results=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                total_candidates=0,
                error=str(e)
            )
    
    def search_by_graph(
        self,
        start_id: str,
        depth: int = 2,
        top_k: int = 10
    ) -> ComparisonResult:
        """
        Graph traversal search starting from a node.
        Returns connected papers within specified depth.
        """
        if not self.is_available:
            return ComparisonResult(
                system="neo4j",
                results=[],
                latency_ms=0,
                total_candidates=0,
                error="Neo4j not available"
            )
        
        start = time.perf_counter()
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH path = (start:Paper {id: $start_id})-[*1..%d]-(connected:Paper)
                    WHERE start <> connected
                    WITH connected, min(length(path)) AS distance
                    RETURN connected.id AS id, connected.title AS title, 
                           connected.text AS text, connected.category AS category,
                           distance
                    ORDER BY distance ASC
                    LIMIT $top_k
                """ % depth, start_id=start_id, top_k=top_k)
                
                results = []
                for record in result:
                    distance = record["distance"]
                    score = 1.0 / (1 + distance)  # Inverse distance score
                    results.append({
                        "node_id": record["id"],
                        "text": record["text"][:500] if record["text"] else "",
                        "metadata": {
                            "title": record["title"],
                            "category": record["category"]
                        },
                        "score": score,
                        "depth": distance
                    })
                
                latency = (time.perf_counter() - start) * 1000
                
                return ComparisonResult(
                    system="neo4j",
                    results=results,
                    latency_ms=latency,
                    total_candidates=len(results),
                    metadata={"search_type": "graph_traversal", "depth": depth}
                )
                
        except Exception as e:
            logger.error(f"Neo4j graph search error: {e}")
            return ComparisonResult(
                system="neo4j",
                results=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                total_candidates=0,
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Neo4j database statistics."""
        if not self.is_available:
            return {"available": False}
        
        try:
            with self.driver.session() as session:
                node_count = session.run(
                    "MATCH (n:Paper) RETURN count(n) AS count"
                ).single()["count"]
                
                edge_count = session.run(
                    "MATCH ()-[r]->() RETURN count(r) AS count"
                ).single()["count"]
                
                return {
                    "available": True,
                    "nodes": node_count,
                    "edges": edge_count,
                    "uri": self.uri
                }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def close(self):
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None


class ChromaDBClient:
    """
    ChromaDB vector database client for comparison.
    Pure vector-based retrieval using embeddings.
    """
    
    def __init__(
        self,
        persist_path: str = "data/chromadb",
        collection_name: str = "arxiv_papers"
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._available = None
        self._embedding_engine = None
    
    def _get_embedding_engine(self):
        """Get HybridMind's embedding engine for consistent embeddings."""
        if self._embedding_engine is None:
            try:
                from api.dependencies import get_db_manager
                self._embedding_engine = get_db_manager().embedding_engine
            except Exception as e:
                logger.warning(f"Could not get embedding engine: {e}")
        return self._embedding_engine
    
    @property
    def client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.persist_path)
                self._available = True
                logger.info(f"ChromaDB connected at {self.persist_path}")
            except ImportError:
                logger.warning("chromadb package not installed")
                self._available = False
            except Exception as e:
                logger.warning(f"ChromaDB connection failed: {e}")
                self._available = False
        return self._client
    
    @property
    def collection(self):
        """Get the papers collection."""
        if self._collection is None and self.client:
            try:
                self._collection = self.client.get_collection(self.collection_name)
            except Exception as e:
                logger.warning(f"ChromaDB collection not found: {e}")
                self._available = False
        return self._collection
    
    @property
    def is_available(self) -> bool:
        """Check if ChromaDB is available."""
        if self._available is None:
            _ = self.collection  # Trigger connection attempt
        return self._available or False
    
    def search(
        self,
        query_text: str,
        top_k: int = 10
    ) -> ComparisonResult:
        """
        Vector similarity search using query embedding.
        """
        if not self.is_available:
            return ComparisonResult(
                system="chromadb",
                results=[],
                latency_ms=0,
                total_candidates=0,
                error="ChromaDB not available"
            )
        
        start = time.perf_counter()
        
        try:
            # Generate embedding using HybridMind's engine for consistency
            embedding_engine = self._get_embedding_engine()
            if embedding_engine:
                query_embedding = embedding_engine.embed(query_text).tolist()
            else:
                # Fall back to ChromaDB's default embedding
                query_embedding = None
            
            if query_embedding:
                result = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                # Use ChromaDB's built-in embedding
                result = self.collection.query(
                    query_texts=[query_text],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            
            results = []
            if result["ids"] and result["ids"][0]:
                for i, node_id in enumerate(result["ids"][0]):
                    # Convert L2 distance to similarity score
                    distance = result["distances"][0][i] if result["distances"] else 0
                    # Normalize: smaller distance = higher score
                    score = 1.0 / (1 + distance)
                    
                    results.append({
                        "node_id": node_id,
                        "text": result["documents"][0][i][:500] if result["documents"] else "",
                        "metadata": result["metadatas"][0][i] if result["metadatas"] else {},
                        "score": score,
                        "distance": distance
                    })
            
            latency = (time.perf_counter() - start) * 1000
            
            return ComparisonResult(
                system="chromadb",
                results=results,
                latency_ms=latency,
                total_candidates=len(results),
                metadata={"search_type": "vector_similarity"}
            )
            
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            return ComparisonResult(
                system="chromadb",
                results=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                total_candidates=0,
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        if not self.is_available:
            return {"available": False}
        
        try:
            count = self.collection.count()
            return {
                "available": True,
                "documents": count,
                "collection": self.collection_name,
                "path": self.persist_path
            }
        except Exception as e:
            return {"available": False, "error": str(e)}


class ComparisonEngine:
    """
    Unified comparison engine for benchmarking HybridMind vs Neo4j vs ChromaDB.
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        chromadb_path: str = "data/chromadb"
    ):
        self.neo4j = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
        self.chromadb = ChromaDBClient(chromadb_path)
        self._hybridmind = None
    
    @property
    def hybridmind(self):
        """Get HybridMind components lazily."""
        if self._hybridmind is None:
            try:
                from api.dependencies import get_db_manager
                db = get_db_manager()
                self._hybridmind = {
                    "vector_engine": db.vector_engine,
                    "graph_engine": db.graph_engine,
                    "hybrid_ranker": db.hybrid_ranker,
                    "embedding_engine": db.embedding_engine
                }
            except Exception as e:
                logger.error(f"Failed to get HybridMind components: {e}")
                self._hybridmind = {}
        return self._hybridmind
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get availability status of all systems."""
        status = {
            "hybridmind": {
                "available": bool(self.hybridmind),
                "type": "hybrid (vector + graph)"
            },
            "neo4j": {
                "available": self.neo4j.is_available,
                "type": "graph-only",
                **self.neo4j.get_stats()
            },
            "chromadb": {
                "available": self.chromadb.is_available,
                "type": "vector-only",
                **self.chromadb.get_stats()
            }
        }
        
        # Add HybridMind stats
        if self.hybridmind:
            try:
                from api.dependencies import get_db_manager
                stats = get_db_manager().get_stats()
                status["hybridmind"].update({
                    "nodes": stats["total_nodes"],
                    "edges": stats["total_edges"],
                    "vector_index_size": stats["vector_index_size"]
                })
            except:
                pass
        
        return status
    
    def search_hybridmind(
        self,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ) -> ComparisonResult:
        """Search using HybridMind's hybrid algorithm."""
        if not self.hybridmind or "hybrid_ranker" not in self.hybridmind:
            return ComparisonResult(
                system="hybridmind",
                results=[],
                latency_ms=0,
                total_candidates=0,
                error="HybridMind not available"
            )
        
        start = time.perf_counter()
        
        try:
            results, query_time_ms, total = self.hybridmind["hybrid_ranker"].search(
                query_text=query_text,
                top_k=top_k,
                vector_weight=vector_weight,
                graph_weight=graph_weight
            )
            
            latency = (time.perf_counter() - start) * 1000
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "node_id": r["node_id"],
                    "text": r["text"][:500] if r["text"] else "",
                    "metadata": r["metadata"],
                    "score": r.get("combined_score", 0),
                    "vector_score": r.get("vector_score", 0),
                    "graph_score": r.get("graph_score", 0)
                })
            
            return ComparisonResult(
                system="hybridmind",
                results=formatted_results,
                latency_ms=latency,
                total_candidates=total,
                metadata={
                    "search_type": "hybrid",
                    "vector_weight": vector_weight,
                    "graph_weight": graph_weight,
                    "internal_time_ms": query_time_ms
                }
            )
            
        except Exception as e:
            logger.error(f"HybridMind search error: {e}")
            return ComparisonResult(
                system="hybridmind",
                results=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                total_candidates=0,
                error=str(e)
            )
    
    def search_hybridmind_vector_only(
        self,
        query_text: str,
        top_k: int = 10
    ) -> ComparisonResult:
        """Search using only HybridMind's vector engine."""
        if not self.hybridmind or "vector_engine" not in self.hybridmind:
            return ComparisonResult(
                system="hybridmind_vector",
                results=[],
                latency_ms=0,
                total_candidates=0,
                error="HybridMind vector engine not available"
            )
        
        start = time.perf_counter()
        
        try:
            results, query_time_ms, total = self.hybridmind["vector_engine"].search(
                query_text=query_text,
                top_k=top_k
            )
            
            latency = (time.perf_counter() - start) * 1000
            
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "node_id": r["node_id"],
                    "text": r["text"][:500] if r["text"] else "",
                    "metadata": r["metadata"],
                    "score": r.get("vector_score", 0)
                })
            
            return ComparisonResult(
                system="hybridmind_vector",
                results=formatted_results,
                latency_ms=latency,
                total_candidates=total,
                metadata={"search_type": "vector_only"}
            )
            
        except Exception as e:
            return ComparisonResult(
                system="hybridmind_vector",
                results=[],
                latency_ms=(time.perf_counter() - start) * 1000,
                total_candidates=0,
                error=str(e)
            )
    
    def compare_all(
        self,
        query_text: str,
        top_k: int = 10,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4
    ) -> Dict[str, Any]:
        """
        Run the same query across all three systems and compare results.
        
        Returns comprehensive comparison including:
        - Results from each system
        - Latency comparison
        - Overlap analysis
        - Unique results per system
        """
        start_total = time.perf_counter()
        
        # Query all systems
        hybridmind_result = self.search_hybridmind(
            query_text, top_k, vector_weight, graph_weight
        )
        neo4j_result = self.neo4j.search_by_text(query_text, top_k)
        chromadb_result = self.chromadb.search(query_text, top_k)
        
        total_time = (time.perf_counter() - start_total) * 1000
        
        # Analyze results
        hm_ids = set(r["node_id"] for r in hybridmind_result.results)
        neo_ids = set(r["node_id"] for r in neo4j_result.results)
        chroma_ids = set(r["node_id"] for r in chromadb_result.results)
        
        # Calculate overlaps
        all_ids = hm_ids | neo_ids | chroma_ids
        common_all = hm_ids & neo_ids & chroma_ids
        hm_neo_overlap = hm_ids & neo_ids
        hm_chroma_overlap = hm_ids & chroma_ids
        neo_chroma_overlap = neo_ids & chroma_ids
        
        # Unique results
        hm_unique = hm_ids - neo_ids - chroma_ids
        neo_unique = neo_ids - hm_ids - chroma_ids
        chroma_unique = chroma_ids - hm_ids - neo_ids
        
        return {
            "query": query_text,
            "top_k": top_k,
            "total_time_ms": total_time,
            "results": {
                "hybridmind": {
                    "items": hybridmind_result.results,
                    "latency_ms": hybridmind_result.latency_ms,
                    "count": len(hybridmind_result.results),
                    "error": hybridmind_result.error,
                    "metadata": hybridmind_result.metadata
                },
                "neo4j": {
                    "items": neo4j_result.results,
                    "latency_ms": neo4j_result.latency_ms,
                    "count": len(neo4j_result.results),
                    "error": neo4j_result.error,
                    "metadata": neo4j_result.metadata
                },
                "chromadb": {
                    "items": chromadb_result.results,
                    "latency_ms": chromadb_result.latency_ms,
                    "count": len(chromadb_result.results),
                    "error": chromadb_result.error,
                    "metadata": chromadb_result.metadata
                }
            },
            "analysis": {
                "total_unique_results": len(all_ids),
                "common_to_all": len(common_all),
                "overlaps": {
                    "hybridmind_neo4j": len(hm_neo_overlap),
                    "hybridmind_chromadb": len(hm_chroma_overlap),
                    "neo4j_chromadb": len(neo_chroma_overlap)
                },
                "unique_per_system": {
                    "hybridmind": len(hm_unique),
                    "neo4j": len(neo_unique),
                    "chromadb": len(chroma_unique)
                },
                "latency_comparison": {
                    "hybridmind_ms": hybridmind_result.latency_ms,
                    "neo4j_ms": neo4j_result.latency_ms,
                    "chromadb_ms": chromadb_result.latency_ms,
                    "fastest": min(
                        [
                            ("hybridmind", hybridmind_result.latency_ms),
                            ("neo4j", neo4j_result.latency_ms if neo4j_result.latency_ms > 0 else float('inf')),
                            ("chromadb", chromadb_result.latency_ms if chromadb_result.latency_ms > 0 else float('inf'))
                        ],
                        key=lambda x: x[1]
                    )[0]
                }
            },
            "system_status": {
                "hybridmind": hybridmind_result.error is None,
                "neo4j": neo4j_result.error is None,
                "chromadb": chromadb_result.error is None
            }
        }
    
    def run_benchmark(
        self,
        queries: List[str],
        top_k: int = 10,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark across all systems.
        
        Args:
            queries: List of test queries
            top_k: Number of results per query
            iterations: Number of times to run each query (for averaging)
        
        Returns:
            Detailed benchmark results with statistics
        """
        results = {
            "hybridmind": {"latencies": [], "result_counts": []},
            "neo4j": {"latencies": [], "result_counts": []},
            "chromadb": {"latencies": [], "result_counts": []}
        }
        
        # Warm-up run
        if queries:
            _ = self.compare_all(queries[0], top_k)
        
        # Run benchmark
        for query in queries:
            for _ in range(iterations):
                comparison = self.compare_all(query, top_k)
                
                for system in ["hybridmind", "neo4j", "chromadb"]:
                    if not comparison["results"][system]["error"]:
                        results[system]["latencies"].append(
                            comparison["results"][system]["latency_ms"]
                        )
                        results[system]["result_counts"].append(
                            comparison["results"][system]["count"]
                        )
        
        # Calculate statistics
        stats = {}
        for system, data in results.items():
            if data["latencies"]:
                latencies = np.array(data["latencies"])
                stats[system] = {
                    "avg_latency_ms": float(np.mean(latencies)),
                    "p50_latency_ms": float(np.percentile(latencies, 50)),
                    "p95_latency_ms": float(np.percentile(latencies, 95)),
                    "p99_latency_ms": float(np.percentile(latencies, 99)),
                    "min_latency_ms": float(np.min(latencies)),
                    "max_latency_ms": float(np.max(latencies)),
                    "std_latency_ms": float(np.std(latencies)),
                    "avg_results": float(np.mean(data["result_counts"])),
                    "total_queries": len(data["latencies"]),
                    "throughput_qps": 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
                }
            else:
                stats[system] = {
                    "error": "No successful queries",
                    "total_queries": 0
                }
        
        return {
            "benchmark_config": {
                "queries_count": len(queries),
                "iterations_per_query": iterations,
                "top_k": top_k
            },
            "statistics": stats,
            "winner": {
                "lowest_latency": min(
                    [(k, v.get("avg_latency_ms", float('inf'))) for k, v in stats.items()],
                    key=lambda x: x[1]
                )[0] if stats else None,
                "highest_throughput": max(
                    [(k, v.get("throughput_qps", 0)) for k, v in stats.items()],
                    key=lambda x: x[1]
                )[0] if stats else None
            }
        }
    
    def close(self):
        """Clean up connections."""
        self.neo4j.close()


# Singleton instance
_comparison_engine: Optional[ComparisonEngine] = None


def get_comparison_engine(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    chromadb_path: str = "data/chromadb"
) -> ComparisonEngine:
    """Get the comparison engine singleton."""
    global _comparison_engine
    if _comparison_engine is None:
        _comparison_engine = ComparisonEngine(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            chromadb_path=chromadb_path
        )
    return _comparison_engine

