"""
Enhanced Streamlit UI for HybridMind with Benchmarking.
Demonstrates superiority over vector-only and graph-only approaches.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybridmind.storage.sqlite_store import SQLiteStore
from hybridmind.storage.vector_index import VectorIndex
from hybridmind.storage.graph_index import GraphIndex
from hybridmind.engine.embedding import EmbeddingEngine
from hybridmind.engine.vector_search import VectorSearchEngine
from hybridmind.engine.graph_search import GraphSearchEngine
from hybridmind.engine.hybrid_ranker import HybridRanker

# Page config
st.set_page_config(
    page_title="HybridMind - Vector + Graph Database",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .result-card {
        border-left: 4px solid #667eea;
        padding: 10px;
        margin: 10px 0;
        background: #f8f9fa;
        border-radius: 0 5px 5px 0;
    }
    .score-badge {
        background: #667eea;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_hybridmind():
    """Initialize HybridMind components."""
    try:
        sqlite_store = SQLiteStore("data/benchmark_hybridmind.db")
        vector_index = VectorIndex(dimension=384, index_path="data/benchmark_vector.index")
        graph_index = GraphIndex(index_path="data/benchmark_graph.pkl")
        
        try:
            vector_index.load()
            graph_index.load()
        except:
            pass
        
        embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        vector_engine = VectorSearchEngine(vector_index, sqlite_store, embedding_engine)
        graph_engine = GraphSearchEngine(graph_index, sqlite_store)
        hybrid_ranker = HybridRanker(vector_engine, graph_engine)
        
        return {
            "sqlite_store": sqlite_store,
            "vector_index": vector_index,
            "graph_index": graph_index,
            "embedding_engine": embedding_engine,
            "vector_engine": vector_engine,
            "graph_engine": graph_engine,
            "hybrid_ranker": hybrid_ranker,
            "available": True
        }
    except Exception as e:
        st.error(f"Error initializing HybridMind: {e}")
        return {"available": False}


@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB for comparison."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="data/chromadb_benchmark")
        collection = client.get_collection("arxiv_papers")
        embedding_engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        return {
            "client": client,
            "collection": collection,
            "embedding_engine": embedding_engine,
            "available": True
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 HybridMind</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Vector + Graph Native Database for AI Retrieval</p>', unsafe_allow_html=True)
    
    # Initialize systems
    hybridmind = init_hybridmind()
    chromadb_sys = init_chromadb()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🔍 Search Demo", "📊 Benchmark Comparison", "📈 Analytics", "ℹ️ About"]
    )
    
    if page == "🔍 Search Demo":
        search_demo_page(hybridmind, chromadb_sys)
    elif page == "📊 Benchmark Comparison":
        benchmark_page(hybridmind, chromadb_sys)
    elif page == "📈 Analytics":
        analytics_page(hybridmind)
    else:
        about_page()


def search_demo_page(hybridmind, chromadb_sys):
    """Interactive search demonstration."""
    st.header("🔍 Search Demonstration")
    
    if not hybridmind.get("available"):
        st.warning("HybridMind not initialized. Please load data first.")
        return
    
    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter your search query",
            value="transformer attention mechanism neural network",
            placeholder="e.g., deep learning optimization techniques"
        )
    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=50, value=10)
    
    # Search type selection
    st.subheader("Compare Search Approaches")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Vector-Only Search")
        st.caption("Pure semantic similarity (like ChromaDB)")
        
        if st.button("Run Vector Search", key="vector_btn"):
            with st.spinner("Searching..."):
                results, time_ms, _ = hybridmind["vector_engine"].search(
                    query_text=query, top_k=top_k
                )
                
                st.metric("Query Time", f"{time_ms:.2f}ms")
                st.metric("Results Found", len(results))
                
                for i, r in enumerate(results[:5]):
                    with st.expander(f"#{i+1}: Score {r['vector_score']:.3f}"):
                        st.write(r.get("text", "")[:500] + "...")
    
    with col2:
        st.markdown("### 🕸️ Graph-Only Search")
        st.caption("Pure relationship traversal (like Neo4j)")
        
        # Get a sample node for graph search
        nodes = hybridmind["sqlite_store"].list_nodes(limit=5)
        if nodes:
            start_node = st.selectbox(
                "Start from node",
                options=[n["id"] for n in nodes],
                format_func=lambda x: x[:20] + "..."
            )
            depth = st.slider("Traversal depth", 1, 3, 2)
            
            if st.button("Run Graph Search", key="graph_btn"):
                with st.spinner("Traversing..."):
                    results, time_ms, _ = hybridmind["graph_engine"].traverse(
                        start_id=start_node, depth=depth
                    )
                    
                    st.metric("Query Time", f"{time_ms:.2f}ms")
                    st.metric("Results Found", len(results))
                    
                    for i, r in enumerate(results[:5]):
                        with st.expander(f"#{i+1}: Depth {r.get('depth', '?')}"):
                            st.write(r.get("text", "")[:500] + "...")
    
    with col3:
        st.markdown("### 🧠 Hybrid Search")
        st.caption("**HybridMind's advantage**")
        
        vector_weight = st.slider("Vector weight (α)", 0.0, 1.0, 0.6)
        graph_weight = 1.0 - vector_weight
        st.caption(f"Graph weight (β): {graph_weight:.2f}")
        
        if st.button("Run Hybrid Search", key="hybrid_btn", type="primary"):
            with st.spinner("Hybrid searching..."):
                results, time_ms, _ = hybridmind["hybrid_ranker"].search(
                    query_text=query,
                    top_k=top_k,
                    vector_weight=vector_weight,
                    graph_weight=graph_weight
                )
                
                st.metric("Query Time", f"{time_ms:.2f}ms")
                st.metric("Results Found", len(results))
                
                for i, r in enumerate(results[:5]):
                    with st.expander(f"#{i+1}: Combined {r['combined_score']:.3f}"):
                        col_a, col_b = st.columns(2)
                        col_a.metric("Vector", f"{r['vector_score']:.3f}")
                        col_b.metric("Graph", f"{r['graph_score']:.3f}")
                        st.write(r.get("text", "")[:500] + "...")
                        if r.get("reasoning"):
                            st.info(r["reasoning"])


def benchmark_page(hybridmind, chromadb_sys):
    """Benchmark comparison page."""
    st.header("📊 Benchmark Comparison")
    
    st.markdown("""
    This page demonstrates **HybridMind's superiority** over traditional databases:
    - **ChromaDB**: Vector-only database
    - **Neo4j**: Graph-only database
    - **HybridMind**: Native hybrid (vector + graph)
    """)
    
    # Test queries
    test_queries = [
        "transformer attention mechanism",
        "deep learning optimization",
        "convolutional neural network",
        "reinforcement learning",
        "generative adversarial network"
    ]
    
    if st.button("🚀 Run Benchmark Suite", type="primary"):
        run_benchmark(hybridmind, chromadb_sys, test_queries)


def run_benchmark(hybridmind, chromadb_sys, queries):
    """Execute benchmark suite."""
    st.subheader("Running Benchmarks...")
    
    results = {
        "hybridmind_vector": [],
        "hybridmind_hybrid": [],
        "chromadb": []
    }
    
    progress = st.progress(0)
    
    for i, query in enumerate(queries):
        progress.progress((i + 1) / len(queries))
        
        # HybridMind Vector Search
        if hybridmind.get("available"):
            _, time_ms, _ = hybridmind["vector_engine"].search(query, top_k=10)
            results["hybridmind_vector"].append(time_ms)
            
            _, time_ms, _ = hybridmind["hybrid_ranker"].search(query, top_k=10)
            results["hybridmind_hybrid"].append(time_ms)
        
        # ChromaDB Search
        if chromadb_sys.get("available"):
            start = time.time()
            emb = chromadb_sys["embedding_engine"].embed(query)
            chromadb_sys["collection"].query(
                query_embeddings=[emb.tolist()],
                n_results=10
            )
            results["chromadb"].append((time.time() - start) * 1000)
    
    # Display results
    st.subheader("📈 Results")
    
    # Create comparison chart
    data = []
    if results["hybridmind_vector"]:
        for i, t in enumerate(results["hybridmind_vector"]):
            data.append({"Query": f"Q{i+1}", "System": "HybridMind (Vector)", "Time (ms)": t})
    if results["hybridmind_hybrid"]:
        for i, t in enumerate(results["hybridmind_hybrid"]):
            data.append({"Query": f"Q{i+1}", "System": "HybridMind (Hybrid)", "Time (ms)": t})
    if results["chromadb"]:
        for i, t in enumerate(results["chromadb"]):
            data.append({"Query": f"Q{i+1}", "System": "ChromaDB (Vector-only)", "Time (ms)": t})
    
    if data:
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df, x="Query", y="Time (ms)", color="System",
            barmode="group",
            title="Query Response Time Comparison",
            color_discrete_map={
                "HybridMind (Vector)": "#667eea",
                "HybridMind (Hybrid)": "#764ba2",
                "ChromaDB (Vector-only)": "#ff6b6b"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if results["hybridmind_vector"]:
                avg = sum(results["hybridmind_vector"]) / len(results["hybridmind_vector"])
                st.metric("HybridMind Vector", f"{avg:.2f}ms avg")
        
        with col2:
            if results["hybridmind_hybrid"]:
                avg = sum(results["hybridmind_hybrid"]) / len(results["hybridmind_hybrid"])
                st.metric("HybridMind Hybrid", f"{avg:.2f}ms avg")
        
        with col3:
            if results["chromadb"]:
                avg = sum(results["chromadb"]) / len(results["chromadb"])
                st.metric("ChromaDB", f"{avg:.2f}ms avg")
    
    # Feature comparison
    st.subheader("🔄 Feature Comparison")
    
    comparison_data = {
        "Feature": [
            "Vector Similarity Search",
            "Graph Traversal",
            "Hybrid Search",
            "Configurable Weights",
            "Score Explanation",
            "Single System"
        ],
        "HybridMind": ["✅", "✅", "✅", "✅", "✅", "✅"],
        "ChromaDB": ["✅", "❌", "❌", "❌", "❌", "✅"],
        "Neo4j": ["❌", "✅", "❌", "❌", "❌", "✅"]
    }
    
    st.table(pd.DataFrame(comparison_data))
    
    # Key advantages
    st.subheader("🏆 HybridMind Advantages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Why Hybrid is Better:**
        1. **Contextual Relevance**: Finds semantically similar content AND related items
        2. **Better Recall**: Discovers connections that pure vector search misses
        3. **Configurable**: Tune vector vs graph weights for your use case
        4. **Explainable**: Understand why each result was returned
        """)
    
    with col2:
        st.markdown("""
        **Real-World Benefits:**
        1. **Research**: Find papers by topic AND citation network
        2. **Enterprise**: Search docs with organizational relationships
        3. **RAG**: Better context for LLM augmentation
        4. **Knowledge Graphs**: Semantic + structural queries
        """)


def analytics_page(hybridmind):
    """Database analytics page."""
    st.header("📈 Database Analytics")
    
    if not hybridmind.get("available"):
        st.warning("HybridMind not initialized")
        return
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    node_count = hybridmind["sqlite_store"].count_nodes()
    edge_count = hybridmind["sqlite_store"].count_edges()
    vector_count = hybridmind["vector_index"].size
    graph_nodes = hybridmind["graph_index"].node_count
    
    col1.metric("📄 Nodes", node_count)
    col2.metric("🔗 Edges", edge_count)
    col3.metric("🎯 Vectors", vector_count)
    col4.metric("🕸️ Graph Nodes", graph_nodes)
    
    # Edge type distribution
    st.subheader("Edge Type Distribution")
    
    edges = hybridmind["sqlite_store"].get_all_edges()
    if edges:
        edge_types = {}
        for e in edges:
            et = e.get("type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1
        
        fig = px.pie(
            values=list(edge_types.values()),
            names=list(edge_types.keys()),
            title="Relationship Types"
        )
        st.plotly_chart(fig, use_container_width=True)


def about_page():
    """About page."""
    st.header("ℹ️ About HybridMind")
    
    st.markdown("""
    ## Vector + Graph Native Database
    
    HybridMind is a novel hybrid database that combines:
    
    ### 🎯 Vector Search
    - Semantic similarity using embeddings
    - Cosine similarity ranking
    - Fast approximate nearest neighbor search
    
    ### 🕸️ Graph Traversal
    - Relationship-based navigation
    - Multi-hop queries
    - Edge-weighted scoring
    
    ### 🧠 Hybrid Retrieval (CRS Algorithm)
    
    ```
    CRS = α × VectorScore + β × GraphScore
    ```
    
    Where:
    - α = Vector weight (default 0.6)
    - β = Graph weight (default 0.4)
    - Configurable per query
    
    ## Architecture
    
    ```
    ┌─────────────────────────────────────────┐
    │            FastAPI Layer                │
    └─────────────────────────────────────────┘
                       │
    ┌─────────────────────────────────────────┐
    │   Vector Engine  │  Graph Engine        │
    │     (FAISS)      │   (NetworkX)         │
    └─────────────────────────────────────────┘
                       │
    ┌─────────────────────────────────────────┐
    │     SQLite Storage + Persistence        │
    └─────────────────────────────────────────┘
    ```
    
    ## Built For
    - DevForge Hackathon
    - Problem Statement 2: Vector + Graph Native Database
    
    ## Team
    - CodeHashira Team
    """)


if __name__ == "__main__":
    main()

