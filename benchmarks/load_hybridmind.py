"""
Load ArXiv dataset into HybridMind database.
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybridmind.storage.sqlite_store import SQLiteStore
from hybridmind.storage.vector_index import VectorIndex
from hybridmind.storage.graph_index import GraphIndex
from hybridmind.engine.embedding import EmbeddingEngine


class HybridMindLoader:
    """Load data into HybridMind database."""
    
    def __init__(self, db_path: str = "data/benchmark_hybridmind.db",
                 vector_path: str = "data/benchmark_vector.index",
                 graph_path: str = "data/benchmark_graph.pkl"):
        self.db_path = db_path
        self.vector_path = vector_path
        self.graph_path = graph_path
        
        # Initialize components
        self.sqlite_store = SQLiteStore(db_path)
        self.vector_index = VectorIndex(dimension=384, index_path=vector_path)
        self.graph_index = GraphIndex(index_path=graph_path)
        self.embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        
    def load_papers(self, papers: List[Dict], batch_size: int = 50) -> Dict:
        """Load papers into HybridMind."""
        stats = {
            "total_papers": len(papers),
            "loaded": 0,
            "embedding_time": 0,
            "storage_time": 0,
            "total_time": 0
        }
        
        start_total = time.time()
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            texts = [p['text'] for p in batch]
            
            # Generate embeddings
            start_embed = time.time()
            embeddings = self.embedding_engine.embed_batch(texts)
            stats["embedding_time"] += time.time() - start_embed
            
            # Store nodes
            start_store = time.time()
            for j, paper in enumerate(batch):
                node_id = paper['id']
                
                # Store in SQLite
                self.sqlite_store.create_node(
                    node_id=node_id,
                    text=paper['text'],
                    metadata={
                        "title": paper.get('title', ''),
                        "source": "arxiv"
                    },
                    embedding=embeddings[j]
                )
                
                # Add to vector index
                self.vector_index.add(node_id, embeddings[j])
                
                # Add to graph index
                self.graph_index.add_node(node_id)
                
                stats["loaded"] += 1
            
            stats["storage_time"] += time.time() - start_store
            
            if (i + batch_size) % 100 == 0:
                print(f"  Loaded {stats['loaded']}/{len(papers)} papers...")
        
        stats["total_time"] = time.time() - start_total
        return stats
    
    def load_edges(self, edges: List[Tuple[str, str, str]]) -> Dict:
        """Load edges into HybridMind."""
        stats = {
            "total_edges": len(edges),
            "loaded": 0,
            "time": 0
        }
        
        start = time.time()
        
        for i, (source, target, edge_type) in enumerate(edges):
            edge_id = f"edge-{i}"
            
            try:
                # Store in SQLite
                self.sqlite_store.create_edge(
                    edge_id=edge_id,
                    source_id=source,
                    target_id=target,
                    edge_type=edge_type,
                    weight=1.0
                )
                
                # Add to graph index
                self.graph_index.add_edge(source, target, edge_type, 1.0, edge_id)
                
                stats["loaded"] += 1
            except Exception as e:
                pass  # Skip if nodes don't exist
        
        stats["time"] = time.time() - start
        return stats
    
    def save(self):
        """Persist indices to disk."""
        self.vector_index.save()
        self.graph_index.save()
        print("HybridMind indices saved.")
    
    def close(self):
        """Clean up resources."""
        self.sqlite_store.close()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            "nodes": self.sqlite_store.count_nodes(),
            "edges": self.sqlite_store.count_edges(),
            "vector_index_size": self.vector_index.size,
            "graph_nodes": self.graph_index.node_count,
            "graph_edges": self.graph_index.edge_count
        }


def load_from_files(papers_file: str = "data/arxiv_papers.json",
                    edges_file: str = "data/arxiv_edges.json"):
    """Load dataset from JSON files into HybridMind."""
    
    print("Loading ArXiv dataset into HybridMind...")
    
    # Load data
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    with open(edges_file, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)
    edges = [(e['source'], e['target'], e['type']) for e in edges_data]
    
    # Initialize loader
    loader = HybridMindLoader()
    
    # Load papers
    print(f"\nLoading {len(papers)} papers...")
    paper_stats = loader.load_papers(papers)
    print(f"  Papers loaded in {paper_stats['total_time']:.2f}s")
    print(f"  Embedding time: {paper_stats['embedding_time']:.2f}s")
    print(f"  Storage time: {paper_stats['storage_time']:.2f}s")
    
    # Load edges
    print(f"\nLoading {len(edges)} edges...")
    edge_stats = loader.load_edges(edges)
    print(f"  Edges loaded in {edge_stats['time']:.2f}s")
    
    # Save and get stats
    loader.save()
    stats = loader.get_stats()
    print(f"\nHybridMind database ready:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Vector index: {stats['vector_index_size']}")
    
    loader.close()
    return stats


if __name__ == "__main__":
    load_from_files()

