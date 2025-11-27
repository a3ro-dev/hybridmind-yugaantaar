"""
Demo data loader for HybridMind.
Loads ArXiv papers from Hugging Face datasets.
"""

import sys
import os
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.dependencies import get_db_manager


def load_arxiv_from_huggingface(num_papers: int = 200):
    """Load ArXiv papers from Hugging Face datasets."""
    print("Loading ArXiv dataset from Hugging Face...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f"{sys.executable} -m pip install datasets --quiet")
        from datasets import load_dataset
    
    # Load ArXiv dataset - CS papers
    dataset = load_dataset(
        "ccdv/arxiv-classification",
        split="train",
        trust_remote_code=True
    )
    
    # Filter for CS papers and limit
    papers = []
    seen_titles = set()
    
    for item in dataset:
        if len(papers) >= num_papers:
            break
        
        text = item.get("text", "") or item.get("abstract", "")
        title = text[:100].split(".")[0] if text else "Untitled"
        
        # Skip duplicates
        if title in seen_titles or len(text) < 50:
            continue
        seen_titles.add(title)
        
        # Extract category
        label = item.get("label", 0)
        categories = [
            "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", 
            "cs.IR", "cs.RO", "cs.SE", "cs.DS", "cs.CR"
        ]
        category = categories[label % len(categories)]
        
        papers.append({
            "id": f"arxiv-{len(papers):04d}",
            "text": text[:1000],  # Limit text length
            "metadata": {
                "title": title[:200],
                "category": category,
                "source": "arxiv",
                "tags": [category.split(".")[-1], "research", "paper"]
            }
        })
    
    print(f"Loaded {len(papers)} papers from ArXiv dataset")
    return papers


def generate_citation_edges(papers: list, edge_ratio: float = 2.5):
    """Generate citation relationships between papers."""
    edges = []
    num_edges = int(len(papers) * edge_ratio)
    
    # Group papers by category for more realistic citations
    by_category = {}
    for p in papers:
        cat = p["metadata"].get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p["id"])
    
    edge_types = ["cites", "extends", "related_to", "same_topic", "uses_method"]
    
    for _ in range(num_edges):
        # 70% chance: cite within same category
        if random.random() < 0.7:
            cat = random.choice(list(by_category.keys()))
            if len(by_category[cat]) >= 2:
                source, target = random.sample(by_category[cat], 2)
            else:
                source = random.choice([p["id"] for p in papers])
                target = random.choice([p["id"] for p in papers])
        else:
            # Cross-category citation
            source = random.choice([p["id"] for p in papers])
            target = random.choice([p["id"] for p in papers])
        
        if source != target:
            edge_type = random.choice(edge_types)
            weight = round(random.uniform(0.5, 1.0), 2)
            
            edge_id = f"edge-{source}-{target}-{edge_type}"
            edges.append({
                "id": edge_id,
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": weight
            })
    
    # Remove duplicates
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)
    
    print(f"Generated {len(unique_edges)} citation edges")
    return unique_edges


def load_demo_data(num_papers: int = 150, clear_existing: bool = False):
    """Load demo dataset into HybridMind."""
    print("=" * 50)
    print("HybridMind Data Loader")
    print("=" * 50)
    
    # Initialize database
    print("\nInitializing HybridMind...")
    db_manager = get_db_manager()
    
    sqlite_store = db_manager.sqlite_store
    vector_index = db_manager.vector_index
    graph_index = db_manager.graph_index
    embedding_engine = db_manager.embedding_engine
    
    # Check existing data
    existing_nodes = sqlite_store.count_nodes()
    if existing_nodes > 0:
        print(f"\nDatabase already contains {existing_nodes} nodes.")
        if not clear_existing:
            response = input("Clear existing data and reload? (y/n): ")
            if response.lower() != 'y':
                print("Keeping existing data.")
                return
        
        print("Clearing existing data...")
        for node in sqlite_store.list_nodes(limit=10000):
            sqlite_store.delete_node(node["id"])
        vector_index.clear()
        graph_index.clear()
    
    # Load papers from Hugging Face
    papers = load_arxiv_from_huggingface(num_papers)
    
    # Generate edges
    edges = generate_citation_edges(papers)
    
    print(f"\nLoading {len(papers)} papers into HybridMind...")
    
    # Load papers
    for i, paper in enumerate(papers, 1):
        if i % 20 == 0 or i == len(papers):
            print(f"  Progress: {i}/{len(papers)} papers...")
        
        # Generate embedding
        embedding = embedding_engine.embed(paper["text"])
        
        # Store in SQLite
        sqlite_store.create_node(
            node_id=paper["id"],
            text=paper["text"],
            metadata=paper["metadata"],
            embedding=embedding
        )
        
        # Add to vector index
        vector_index.add(paper["id"], embedding)
        
        # Add to graph index
        graph_index.add_node(paper["id"])
    
    print(f"\nLoading {len(edges)} relationships...")
    
    # Load edges
    for i, edge in enumerate(edges, 1):
        if i % 50 == 0 or i == len(edges):
            print(f"  Progress: {i}/{len(edges)} edges...")
        
        # Store in SQLite
        sqlite_store.create_edge(
            edge_id=edge["id"],
            source_id=edge["source"],
            target_id=edge["target"],
            edge_type=edge["type"],
            weight=edge["weight"]
        )
        
        # Add to graph index
        graph_index.add_edge(
            source_id=edge["source"],
            target_id=edge["target"],
            edge_type=edge["type"],
            weight=edge["weight"],
            edge_id=edge["id"]
        )
    
    # Save indexes
    print("\nSaving indexes...")
    db_manager.save_indexes()
    
    # Print summary
    stats = db_manager.get_stats()
    print("\n" + "=" * 50)
    print("✅ Data Loaded Successfully!")
    print("=" * 50)
    print(f"  📄 Nodes: {stats['total_nodes']}")
    print(f"  🔗 Edges: {stats['total_edges']}")
    print(f"  📊 Edge Types: {stats['edge_types']}")
    print(f"  🎯 Vector Index: {stats['vector_index_size']} vectors")
    print(f"  🕸️  Graph Nodes: {stats['graph_node_count']}")
    print("\n🔍 Try these searches:")
    print('  - "deep learning neural networks"')
    print('  - "natural language processing transformers"')
    print('  - "computer vision image classification"')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load demo data into HybridMind")
    parser.add_argument("--papers", type=int, default=150, help="Number of papers to load")
    parser.add_argument("--clear", action="store_true", help="Clear existing data without prompt")
    args = parser.parse_args()
    
    load_demo_data(num_papers=args.papers, clear_existing=args.clear)
