"""
Unified Data Loader for HybridMind Comparison Demo
Loads the same dataset into:
1. HybridMind (Hybrid: Vector + Graph)
2. Neo4j (Graph-only)
3. ChromaDB (Vector-only)
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.dependencies import get_db_manager


def load_arxiv_dataset(num_papers: int = 150):
    """Load ArXiv papers from Hugging Face."""
    print("Loading ArXiv dataset from Hugging Face...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        os.system(f"{sys.executable} -m pip install datasets --quiet")
        from datasets import load_dataset
    
    dataset = load_dataset("ccdv/arxiv-classification", split="train")
    
    papers = []
    seen = set()
    
    for item in dataset:
        if len(papers) >= num_papers:
            break
        
        text = item.get("text", "") or ""
        if len(text) < 50:
            continue
        
        title = text[:100].split(".")[0]
        if title in seen:
            continue
        seen.add(title)
        
        label = item.get("label", 0)
        categories = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", 
                      "cs.IR", "cs.RO", "cs.SE", "cs.DS", "cs.CR"]
        category = categories[label % len(categories)]
        
        papers.append({
            "id": f"arxiv-{len(papers):04d}",
            "text": text[:1000],
            "title": title[:200],
            "category": category,
            "tags": [category.split(".")[-1], "research"]
        })
    
    print(f"  Loaded {len(papers)} papers")
    return papers


def generate_edges(papers: list, ratio: float = 2.5):
    """Generate citation relationships."""
    edges = []
    num_edges = int(len(papers) * ratio)
    
    by_category = {}
    for p in papers:
        cat = p["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p["id"])
    
    edge_types = ["cites", "extends", "related_to", "same_topic"]
    
    for _ in range(num_edges):
        if random.random() < 0.7 and by_category:
            cat = random.choice(list(by_category.keys()))
            if len(by_category[cat]) >= 2:
                source, target = random.sample(by_category[cat], 2)
            else:
                source = random.choice([p["id"] for p in papers])
                target = random.choice([p["id"] for p in papers])
        else:
            source = random.choice([p["id"] for p in papers])
            target = random.choice([p["id"] for p in papers])
        
        if source != target:
            edges.append({
                "source": source,
                "target": target,
                "type": random.choice(edge_types),
                "weight": round(random.uniform(0.5, 1.0), 2)
            })
    
    # Deduplicate
    seen = set()
    unique = []
    for e in edges:
        key = (e["source"], e["target"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    
    print(f"  Generated {len(unique)} edges")
    return unique


# ============================================================================
# HYBRIDMIND LOADER
# ============================================================================

def load_hybridmind(papers: list, edges: list):
    """Load data into HybridMind (our hybrid database)."""
    print("\n[1/3] Loading HybridMind (Hybrid: Vector + Graph)...")
    
    db = get_db_manager()
    
    # Clear existing
    existing = db.sqlite_store.count_nodes()
    if existing > 0:
        print(f"  Clearing {existing} existing nodes...")
        for node in db.sqlite_store.list_nodes(limit=10000):
            db.sqlite_store.delete_node(node["id"])
        db.vector_index.clear()
        db.graph_index.clear()
    
    # Load papers
    for i, p in enumerate(papers):
        if (i + 1) % 25 == 0:
            print(f"  Papers: {i + 1}/{len(papers)}")
        
        embedding = db.embedding_engine.embed(p["text"])
        
        db.sqlite_store.create_node(
            node_id=p["id"],
            text=p["text"],
            metadata={"title": p["title"], "category": p["category"], "tags": p["tags"]},
            embedding=embedding
        )
        db.vector_index.add(p["id"], embedding)
        db.graph_index.add_node(p["id"])
    
    # Load edges
    for e in edges:
        edge_id = f"edge-{e['source']}-{e['target']}"
        db.sqlite_store.create_edge(edge_id, e["source"], e["target"], e["type"], e["weight"])
        db.graph_index.add_edge(e["source"], e["target"], e["type"], e["weight"], edge_id)
    
    db.save_indexes()
    print(f"  ✓ HybridMind: {len(papers)} nodes, {len(edges)} edges")


# ============================================================================
# NEO4J LOADER
# ============================================================================

def load_neo4j(papers: list, edges: list, uri: str = "bolt://localhost:7687", 
               user: str = "neo4j", password: str = "password"):
    """Load data into Neo4j (graph-only database)."""
    print("\n[2/3] Loading Neo4j (Graph-only)...")
    
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("  ✗ Neo4j driver not installed")
        return False
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception as e:
        print(f"  ✗ Cannot connect to Neo4j: {e}")
        print(f"    Make sure Neo4j is running at {uri}")
        print(f"    Update credentials if needed (current: {user}/{password})")
        return False
    
    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create nodes
        for i, p in enumerate(papers):
            if (i + 1) % 25 == 0:
                print(f"  Papers: {i + 1}/{len(papers)}")
            
            session.run("""
                CREATE (p:Paper {
                    id: $id,
                    title: $title,
                    text: $text,
                    category: $category
                })
            """, id=p["id"], title=p["title"], text=p["text"], category=p["category"])
        
        # Create edges
        for e in edges:
            session.run("""
                MATCH (a:Paper {id: $source})
                MATCH (b:Paper {id: $target})
                CREATE (a)-[:RELATES {type: $type, weight: $weight}]->(b)
            """, source=e["source"], target=e["target"], type=e["type"], weight=e["weight"])
    
    driver.close()
    print(f"  ✓ Neo4j: {len(papers)} nodes, {len(edges)} edges")
    return True


# ============================================================================
# CHROMADB LOADER
# ============================================================================

def load_chromadb(papers: list):
    """Load data into ChromaDB (vector-only database)."""
    print("\n[3/3] Loading ChromaDB (Vector-only)...")
    
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("  ✗ ChromaDB not installed")
        return False
    
    # Create persistent client
    client = chromadb.PersistentClient(path="data/chromadb")
    
    # Delete existing collection
    try:
        client.delete_collection("arxiv_papers")
    except:
        pass
    
    # Create collection with embedding function
    collection = client.create_collection(
        name="arxiv_papers",
        metadata={"description": "ArXiv papers for comparison"}
    )
    
    # Get embeddings from HybridMind's engine
    db = get_db_manager()
    
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for i, p in enumerate(papers):
        if (i + 1) % 25 == 0:
            print(f"  Papers: {i + 1}/{len(papers)}")
        
        embedding = db.embedding_engine.embed(p["text"])
        
        ids.append(p["id"])
        embeddings.append(embedding.tolist())
        documents.append(p["text"])
        metadatas.append({"title": p["title"], "category": p["category"]})
    
    # Add in batches
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
    
    print(f"  ✓ ChromaDB: {len(papers)} documents")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load data into all three databases")
    parser.add_argument("--papers", type=int, default=150, help="Number of papers")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j loading")
    args = parser.parse_args()
    
    print("=" * 60)
    print("HybridMind Comparison Demo - Data Loader")
    print("=" * 60)
    
    # Load dataset
    papers = load_arxiv_dataset(args.papers)
    edges = generate_edges(papers)
    
    # Load into all databases
    load_hybridmind(papers, edges)
    
    if not args.skip_neo4j:
        load_neo4j(papers, edges, args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    
    load_chromadb(papers)
    
    print("\n" + "=" * 60)
    print("✓ All databases loaded!")
    print("=" * 60)
    print("\nReady for comparison benchmarks.")
    print("Run: streamlit run ui/app.py")


if __name__ == "__main__":
    main()

