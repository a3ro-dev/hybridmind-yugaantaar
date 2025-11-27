"""
Setup script to download data and populate all databases for benchmarking.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("="*60)
    print("HYBRIDMIND BENCHMARK SETUP")
    print("="*60)
    
    # Ensure data directory exists
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    os.chdir(Path(__file__).parent.parent)
    
    # Step 1: Download dataset
    print("\n" + "="*40)
    print("STEP 1: Downloading ArXiv Dataset")
    print("="*40)
    
    from benchmarks.download_arxiv import download_arxiv_dataset, generate_citation_edges, save_dataset
    
    sample_size = 500  # Start with 500 papers for quick demo
    papers = download_arxiv_dataset(sample_size=sample_size)
    edges = generate_citation_edges(papers, edge_ratio=2.5)
    save_dataset(papers, edges)
    
    print(f"\n✅ Dataset ready: {len(papers)} papers, {len(edges)} edges")
    
    # Step 2: Load into HybridMind
    print("\n" + "="*40)
    print("STEP 2: Loading into HybridMind")
    print("="*40)
    
    from benchmarks.load_hybridmind import load_from_files as load_hybridmind
    hybridmind_stats = load_hybridmind()
    
    print(f"\n✅ HybridMind ready")
    
    # Step 3: Load into ChromaDB
    print("\n" + "="*40)
    print("STEP 3: Loading into ChromaDB")
    print("="*40)
    
    try:
        from benchmarks.load_chromadb import load_from_files as load_chromadb
        chromadb_stats = load_chromadb()
        print(f"\n✅ ChromaDB ready")
    except Exception as e:
        print(f"⚠️ ChromaDB setup failed: {e}")
        print("  ChromaDB is optional for benchmarking")
    
    # Step 4: Neo4j (optional - requires running instance)
    print("\n" + "="*40)
    print("STEP 4: Neo4j Setup (Optional)")
    print("="*40)
    print("""
To set up Neo4j for comparison:

1. Start Neo4j with Docker:
   docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \\
       -e NEO4J_AUTH=neo4j/password neo4j:latest

2. Run the Neo4j loader:
   python benchmarks/load_neo4j.py

Neo4j is optional - HybridMind benchmarks work without it.
""")
    
    # Summary
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print(f"""
📊 Data Summary:
   Papers: {len(papers)}
   Edges: {len(edges)}

🚀 Next Steps:

1. Run benchmarks:
   python benchmarks/benchmark_runner.py

2. Start the benchmark UI:
   streamlit run ui/benchmark_app.py

3. Access the API:
   uvicorn hybridmind.main:app --reload --port 8000
   Open: http://localhost:8000/docs

✨ HybridMind is ready to demonstrate hybrid search superiority!
""")


if __name__ == "__main__":
    main()

