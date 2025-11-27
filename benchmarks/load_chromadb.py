"""
Load ArXiv dataset into ChromaDB vector database for comparison.
ChromaDB is a popular open-source vector database.
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not installed. Run: pip install chromadb")

from engine.embedding import EmbeddingEngine


class ChromaDBLoader:
    """Load data into ChromaDB vector database."""
    
    def __init__(self, persist_dir: str = "data/chromadb_benchmark"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available")
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Delete existing collection if exists
        try:
            self.client.delete_collection("arxiv_papers")
        except:
            pass
        
        # Create collection
        self.collection = self.client.create_collection(
            name="arxiv_papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding engine (same as HybridMind for fair comparison)
        self.embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    
    def load_papers(self, papers: List[Dict], batch_size: int = 100) -> Dict:
        """Load papers into ChromaDB."""
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
            
            ids = [p['id'] for p in batch]
            texts = [p['text'] for p in batch]
            metadatas = [{"title": p.get('title', '')} for p in batch]
            
            # Generate embeddings
            start_embed = time.time()
            embeddings = self.embedding_engine.embed_batch(texts)
            stats["embedding_time"] += time.time() - start_embed
            
            # Store in ChromaDB
            start_store = time.time()
            self.collection.add(
                ids=ids,
                embeddings=[emb.tolist() for emb in embeddings],
                documents=texts,
                metadatas=metadatas
            )
            stats["storage_time"] += time.time() - start_store
            
            stats["loaded"] += len(batch)
            
            if (i + batch_size) % 200 == 0:
                print(f"  Loaded {stats['loaded']}/{len(papers)} papers...")
        
        stats["total_time"] = time.time() - start_total
        return stats
    
    def search_vector(self, query_text: str, top_k: int = 10) -> Tuple[List[Dict], float]:
        """Vector similarity search - what ChromaDB does best."""
        start = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_engine.embed(query_text)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        query_time = (time.time() - start) * 1000  # ms
        
        # Format results
        formatted = []
        if results['ids'] and results['ids'][0]:
            for i, id_ in enumerate(results['ids'][0]):
                # ChromaDB returns distances (lower is better), convert to similarity
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance  # For cosine distance
                
                formatted.append({
                    "id": id_,
                    "text": results['documents'][0][i] if results['documents'] else "",
                    "score": similarity,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                })
        
        return formatted, query_time
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            "count": self.collection.count()
        }


def load_from_files(papers_file: str = "data/arxiv_papers.json") -> Dict:
    """Load dataset from JSON file into ChromaDB."""
    
    if not CHROMADB_AVAILABLE:
        print("ChromaDB not installed")
        return None
    
    print("Loading ArXiv dataset into ChromaDB...")
    
    # Load data
    with open(papers_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    # Initialize loader
    loader = ChromaDBLoader()
    
    # Load papers
    print(f"\nLoading {len(papers)} papers...")
    paper_stats = loader.load_papers(papers)
    print(f"  Papers loaded in {paper_stats['total_time']:.2f}s")
    print(f"  Embedding time: {paper_stats['embedding_time']:.2f}s")
    print(f"  Storage time: {paper_stats['storage_time']:.2f}s")
    
    # Get stats
    stats = loader.get_stats()
    print(f"\nChromaDB database ready:")
    print(f"  Documents: {stats['count']}")
    
    return stats


if __name__ == "__main__":
    load_from_files()

