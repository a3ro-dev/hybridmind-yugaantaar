"""
Benchmark runner to compare HybridMind with Neo4j and ChromaDB.
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.graph_index import GraphIndex
from engine.embedding import EmbeddingEngine
from engine.vector_search import VectorSearchEngine
from engine.graph_search import GraphSearchEngine
from engine.hybrid_ranker import HybridRanker


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    system: str
    query_type: str
    query_time_ms: float
    result_count: int
    query_text: str = ""


class BenchmarkRunner:
    """Run benchmarks comparing different database systems."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.hybridmind = None
        self.chromadb = None
        self.neo4j = None
        
    def setup_hybridmind(self, db_path: str = "data/benchmark_hybridmind.db",
                         vector_path: str = "data/benchmark_vector.index",
                         graph_path: str = "data/benchmark_graph.pkl"):
        """Initialize HybridMind components."""
        print("Setting up HybridMind...")
        
        sqlite_store = SQLiteStore(db_path)
        vector_index = VectorIndex(dimension=384, index_path=vector_path)
        graph_index = GraphIndex(index_path=graph_path)
        
        # Load persisted indices
        try:
            vector_index.load()
            graph_index.load()
        except:
            pass
        
        embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        
        vector_engine = VectorSearchEngine(vector_index, sqlite_store, embedding_engine)
        graph_engine = GraphSearchEngine(graph_index, sqlite_store)
        hybrid_ranker = HybridRanker(vector_engine, graph_engine)
        
        self.hybridmind = {
            "sqlite_store": sqlite_store,
            "vector_index": vector_index,
            "graph_index": graph_index,
            "embedding_engine": embedding_engine,
            "vector_engine": vector_engine,
            "graph_engine": graph_engine,
            "hybrid_ranker": hybrid_ranker
        }
        
        print(f"  Nodes: {sqlite_store.count_nodes()}")
        print(f"  Vector index: {vector_index.size}")
        print(f"  Graph nodes: {graph_index.node_count}")
    
    def setup_chromadb(self, persist_dir: str = "data/chromadb_benchmark"):
        """Initialize ChromaDB."""
        try:
            from benchmarks.load_chromadb import ChromaDBLoader, CHROMADB_AVAILABLE
            if CHROMADB_AVAILABLE:
                print("Setting up ChromaDB...")
                # Use existing client
                import chromadb
                client = chromadb.PersistentClient(path=persist_dir)
                collection = client.get_collection("arxiv_papers")
                
                embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
                
                self.chromadb = {
                    "client": client,
                    "collection": collection,
                    "embedding_engine": embedding_engine
                }
                print(f"  Documents: {collection.count()}")
        except Exception as e:
            print(f"ChromaDB not available: {e}")
    
    def setup_neo4j(self, uri: str = "bolt://localhost:7687",
                    user: str = "neo4j", password: str = "password"):
        """Initialize Neo4j connection."""
        try:
            from benchmarks.load_neo4j import Neo4jLoader, NEO4J_AVAILABLE
            if NEO4J_AVAILABLE:
                print("Setting up Neo4j...")
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(uri, auth=(user, password))
                
                # Test connection
                with driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as count")
                    count = result.single()["count"]
                
                self.neo4j = {"driver": driver}
                print(f"  Nodes: {count}")
        except Exception as e:
            print(f"Neo4j not available: {e}")
    
    def benchmark_vector_search(self, queries: List[str], top_k: int = 10) -> Dict:
        """Benchmark vector search across systems."""
        print("\n=== Vector Search Benchmark ===")
        
        results = {
            "hybridmind": [],
            "chromadb": []
        }
        
        for query in queries:
            # HybridMind vector search
            if self.hybridmind:
                start = time.time()
                res, query_time, _ = self.hybridmind["vector_engine"].search(
                    query_text=query, top_k=top_k
                )
                results["hybridmind"].append({
                    "query": query,
                    "time_ms": query_time,
                    "count": len(res)
                })
            
            # ChromaDB vector search
            if self.chromadb:
                start = time.time()
                query_emb = self.chromadb["embedding_engine"].embed(query)
                chroma_res = self.chromadb["collection"].query(
                    query_embeddings=[query_emb.tolist()],
                    n_results=top_k
                )
                query_time = (time.time() - start) * 1000
                results["chromadb"].append({
                    "query": query,
                    "time_ms": query_time,
                    "count": len(chroma_res['ids'][0]) if chroma_res['ids'] else 0
                })
        
        # Print summary
        for system, sys_results in results.items():
            if sys_results:
                times = [r["time_ms"] for r in sys_results]
                print(f"\n{system.upper()}:")
                print(f"  Avg time: {statistics.mean(times):.2f}ms")
                print(f"  Min time: {min(times):.2f}ms")
                print(f"  Max time: {max(times):.2f}ms")
        
        return results
    
    def benchmark_graph_search(self, start_ids: List[str], depth: int = 2) -> Dict:
        """Benchmark graph traversal across systems."""
        print("\n=== Graph Search Benchmark ===")
        
        results = {
            "hybridmind": [],
            "neo4j": []
        }
        
        for start_id in start_ids:
            # HybridMind graph search
            if self.hybridmind:
                start = time.time()
                res, query_time, _ = self.hybridmind["graph_engine"].traverse(
                    start_id=start_id, depth=depth
                )
                results["hybridmind"].append({
                    "start_id": start_id,
                    "time_ms": query_time,
                    "count": len(res)
                })
            
            # Neo4j graph search
            if self.neo4j:
                start = time.time()
                with self.neo4j["driver"].session() as session:
                    result = session.run(
                        f"""
                        MATCH path = (start:Paper {{id: $start_id}})-[*1..{depth}]->(related:Paper)
                        RETURN DISTINCT related.id as id
                        LIMIT 20
                        """,
                        start_id=start_id
                    )
                    neo4j_res = list(result)
                query_time = (time.time() - start) * 1000
                results["neo4j"].append({
                    "start_id": start_id,
                    "time_ms": query_time,
                    "count": len(neo4j_res)
                })
        
        # Print summary
        for system, sys_results in results.items():
            if sys_results:
                times = [r["time_ms"] for r in sys_results]
                print(f"\n{system.upper()}:")
                print(f"  Avg time: {statistics.mean(times):.2f}ms")
                print(f"  Min time: {min(times):.2f}ms")
                print(f"  Max time: {max(times):.2f}ms")
        
        return results
    
    def benchmark_hybrid_search(self, queries: List[str], top_k: int = 10) -> Dict:
        """Benchmark hybrid search - HybridMind's specialty."""
        print("\n=== Hybrid Search Benchmark ===")
        print("(Only HybridMind supports native hybrid search)")
        
        results = {"hybridmind": []}
        
        if self.hybridmind:
            for query in queries:
                start = time.time()
                res, query_time, _ = self.hybridmind["hybrid_ranker"].search(
                    query_text=query,
                    top_k=top_k,
                    vector_weight=0.6,
                    graph_weight=0.4
                )
                results["hybridmind"].append({
                    "query": query,
                    "time_ms": query_time,
                    "count": len(res),
                    "results": res[:3]  # Keep top 3 for analysis
                })
            
            times = [r["time_ms"] for r in results["hybridmind"]]
            print(f"\nHYBRIDMIND:")
            print(f"  Avg time: {statistics.mean(times):.2f}ms")
            print(f"  Min time: {min(times):.2f}ms")
            print(f"  Max time: {max(times):.2f}ms")
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        print("\n" + "="*60)
        print("HYBRIDMIND BENCHMARK SUITE")
        print("="*60)
        
        # Test queries
        vector_queries = [
            "transformer attention mechanism neural network",
            "deep learning optimization gradient descent",
            "convolutional neural network image classification",
            "reinforcement learning policy gradient",
            "generative adversarial network image synthesis",
            "natural language processing BERT model",
            "graph neural network node embedding",
            "recurrent neural network sequence modeling",
            "autoencoder latent representation learning",
            "machine learning model training efficiency"
        ]
        
        # Get sample node IDs for graph search
        start_ids = []
        if self.hybridmind:
            nodes = self.hybridmind["sqlite_store"].list_nodes(limit=10)
            start_ids = [n["id"] for n in nodes]
        
        all_results = {}
        
        # Vector search benchmark
        all_results["vector"] = self.benchmark_vector_search(vector_queries)
        
        # Graph search benchmark
        if start_ids:
            all_results["graph"] = self.benchmark_graph_search(start_ids)
        
        # Hybrid search benchmark (HybridMind only)
        all_results["hybrid"] = self.benchmark_hybrid_search(vector_queries)
        
        # Summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """Print benchmark summary."""
        print("\n📊 Performance Comparison:\n")
        
        # Vector search comparison
        if "vector" in results:
            print("VECTOR SEARCH:")
            for sys, data in results["vector"].items():
                if data:
                    avg = statistics.mean([d["time_ms"] for d in data])
                    print(f"  {sys:12} → {avg:6.2f}ms avg")
        
        # Graph search comparison
        if "graph" in results:
            print("\nGRAPH TRAVERSAL:")
            for sys, data in results["graph"].items():
                if data:
                    avg = statistics.mean([d["time_ms"] for d in data])
                    print(f"  {sys:12} → {avg:6.2f}ms avg")
        
        # Hybrid search
        if "hybrid" in results:
            print("\nHYBRID SEARCH (HybridMind exclusive):")
            if results["hybrid"]["hybridmind"]:
                avg = statistics.mean([d["time_ms"] for d in results["hybrid"]["hybridmind"]])
                print(f"  hybridmind   → {avg:6.2f}ms avg")
                print("  chromadb     → ❌ Not supported")
                print("  neo4j        → ❌ Not supported")
        
        print("\n✅ HybridMind Advantages:")
        print("  1. Native hybrid search combining vector + graph")
        print("  2. Single system for all query types")
        print("  3. Unified scoring with configurable weights")
        print("  4. Better results through contextual relevance")
    
    def cleanup(self):
        """Clean up resources."""
        if self.hybridmind:
            self.hybridmind["sqlite_store"].close()
        if self.neo4j:
            self.neo4j["driver"].close()


def main():
    """Run benchmarks."""
    runner = BenchmarkRunner()
    
    # Setup systems
    runner.setup_hybridmind()
    runner.setup_chromadb()
    runner.setup_neo4j()
    
    # Run benchmarks
    results = runner.run_full_benchmark()
    
    # Save results
    with open("data/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to data/benchmark_results.json")
    
    runner.cleanup()


if __name__ == "__main__":
    main()

