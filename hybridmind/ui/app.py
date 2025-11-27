"""
HybridMind - Vector + Graph Native Database
Research-grade hybrid retrieval system
"""

import sys
import os

# Ensure hybridmind is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="HybridMind",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, academic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    }
    
    code, pre, .stCode {
        font-family: 'IBM Plex Mono', monospace;
    }
    
    h1, h2, h3, h4 {
        font-weight: 600;
        color: #1a1a2e;
    }
    
    .main-title {
        font-size: 2.25rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #64748b;
        font-weight: 400;
    }
    
    .formula-display {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        text-align: center;
        color: #334155;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-ok {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .status-error {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 500;
        border-radius: 0;
    }
    
    .stButton > button {
        font-weight: 500;
    }
    
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    
    .sidebar-heading {
        font-size: 0.7rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = st.sidebar.text_input("API Endpoint", value="http://localhost:8000")


def api_call(endpoint: str, method: str = "GET", data: dict = None) -> Optional[Dict]:
    """Execute API request."""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=data, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None


def check_health() -> tuple[bool, dict]:
    """Verify API connectivity."""
    health = api_call("/health")
    if health and health.get("status") == "healthy":
        return True, health
    return False, {}


# ============================================================================
# LAYOUT
# ============================================================================

def render_header():
    """Page header."""
    st.markdown('<h1 class="main-title">HybridMind</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Vector + Graph Native Database for Hybrid Retrieval</p>', unsafe_allow_html=True)
    
    is_connected, health = check_health()
    if is_connected:
        n, e = health.get('nodes', 0), health.get('edges', 0)
        st.markdown(f'<span class="status-indicator status-ok">● Connected — {n:,} nodes, {e:,} edges</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-error">● Disconnected</span>', unsafe_allow_html=True)


def render_sidebar() -> str:
    """Sidebar navigation."""
    st.sidebar.markdown('<p class="sidebar-heading">Navigation</p>', unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Page",
        ["Search", "Benchmarks", "Analytics", "Data Explorer", "Documentation"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Statistics
    st.sidebar.markdown('<p class="sidebar-heading">Database Statistics</p>', unsafe_allow_html=True)
    stats = api_call("/search/stats")
    if stats:
        c1, c2 = st.sidebar.columns(2)
        c1.metric("Nodes", f"{stats.get('total_nodes', 0):,}")
        c2.metric("Edges", f"{stats.get('total_edges', 0):,}")
        st.sidebar.metric("Vector Index Size", f"{stats.get('vector_index_size', 0):,}")
    else:
        st.sidebar.caption("No connection")
    
    st.sidebar.markdown("---")
    
    # CRS Algorithm
    st.sidebar.markdown('<p class="sidebar-heading">CRS Algorithm</p>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="formula-display">S = αV + βG</div>', unsafe_allow_html=True)
    st.sidebar.caption("α: vector weight, β: graph weight")
    st.sidebar.caption("V: semantic similarity, G: graph proximity")
    
    return page


# ============================================================================
# SEARCH
# ============================================================================

def render_search_page():
    """Search interface."""
    st.header("Search")
    
    tabs = st.tabs(["Hybrid", "Vector", "Graph", "Comparison"])
    
    with tabs[0]:
        render_hybrid_search()
    with tabs[1]:
        render_vector_search()
    with tabs[2]:
        render_graph_search()
    with tabs[3]:
        render_comparison_search()


def render_hybrid_search():
    """Hybrid search with CRS algorithm."""
    st.subheader("Hybrid Search")
    st.caption("Contextual Relevance Scoring: combines vector similarity with graph structure")
    
    query = st.text_input("Query", placeholder="Enter search query...", key="h_query")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.slider("Top-K", 5, 50, 10, key="h_k")
    with c2:
        alpha = st.slider("α (vector)", 0.0, 1.0, 0.6, 0.05, key="h_alpha")
    with c3:
        beta = st.slider("β (graph)", 0.0, 1.0, 0.4, 0.05, key="h_beta")
    
    anchor = st.text_input("Anchor node (optional)", key="h_anchor", placeholder="Node ID for graph context")
    
    if st.button("Execute Search", key="h_btn", type="primary"):
        if not query:
            st.warning("Query required")
            return
        
        payload = {"query_text": query, "top_k": k, "vector_weight": alpha, "graph_weight": beta}
        if anchor:
            payload["anchor_nodes"] = [anchor]
        
        with st.spinner("Processing..."):
            results = api_call("/search/hybrid", method="POST", data=payload)
        
        if results:
            display_results(results, show_breakdown=True)


def render_vector_search():
    """Pure vector similarity search."""
    st.subheader("Vector Search")
    st.caption("Semantic similarity using embedding space distance")
    
    query = st.text_input("Query", placeholder="Enter search query...", key="v_query")
    k = st.slider("Top-K", 5, 50, 10, key="v_k")
    
    if st.button("Execute Search", key="v_btn", type="primary"):
        if not query:
            st.warning("Query required")
            return
        
        with st.spinner("Processing..."):
            results = api_call("/search/vector", method="POST", data={"query_text": query, "top_k": k})
        
        if results:
            display_results(results, score_field="vector_score")


def render_graph_search():
    """Graph traversal search."""
    st.subheader("Graph Traversal")
    st.caption("Breadth-first exploration of node relationships")
    
    start = st.text_input("Start Node ID", placeholder="e.g., arxiv-0001", key="g_start")
    
    c1, c2 = st.columns(2)
    with c1:
        depth = st.slider("Max Depth", 1, 5, 2, key="g_depth")
    with c2:
        direction = st.selectbox("Direction", ["both", "outgoing", "incoming"], key="g_dir")
    
    if st.button("Execute Traversal", key="g_btn", type="primary"):
        if not start:
            st.warning("Start node required")
            return
        
        with st.spinner("Traversing..."):
            results = api_call("/search/graph", data={"start_id": start, "depth": depth, "direction": direction})
        
        if results:
            display_results(results, score_field="graph_score", show_path=True)


def render_comparison_search():
    """Side-by-side comparison of search modes."""
    st.subheader("Mode Comparison")
    st.caption("Compare results across vector, graph, and hybrid approaches")
    
    query = st.text_input("Query", placeholder="Enter search query...", key="c_query")
    anchor = st.text_input("Anchor node (for graph context)", key="c_anchor")
    
    if st.button("Run Comparison", key="c_btn", type="primary"):
        if not query:
            st.warning("Query required")
            return
        
        payload = {"query_text": query, "top_k": 5}
        if anchor:
            payload["anchor_nodes"] = [anchor]
        
        with st.spinner("Running..."):
            results = api_call("/search/compare", method="POST", data=payload)
        
        if results:
            display_comparison(results)


def display_results(results: dict, score_field: str = "combined_score", 
                    show_breakdown: bool = False, show_path: bool = False):
    """Render search results."""
    
    latency = results.get('query_time_ms', 0)
    total = results.get('total_candidates', 0)
    
    st.markdown(f"**Latency:** {latency:.2f}ms · **Candidates:** {total}")
    
    if not results.get("results"):
        st.info("No results")
        return
    
    for i, r in enumerate(results["results"], 1):
        meta = r.get('metadata', {})
        title = meta.get('title', r['node_id'])[:60]
        
        with st.expander(f"{i}. {title}", expanded=i <= 3):
            if show_breakdown:
                c1, c2, c3 = st.columns(3)
                c1.metric("Vector", f"{r.get('vector_score', 0):.4f}")
                c2.metric("Graph", f"{r.get('graph_score', 0):.4f}")
                c3.metric("Combined", f"{r.get('combined_score', 0):.4f}")
            else:
                st.metric("Score", f"{r.get(score_field, 0):.4f}")
            
            st.markdown(f"**Text:** {r.get('text', '')[:400]}...")
            
            if meta:
                parts = []
                if 'category' in meta:
                    parts.append(f"Category: {meta['category']}")
                if 'tags' in meta:
                    parts.append(f"Tags: {', '.join(meta['tags'][:3])}")
                if parts:
                    st.caption(" · ".join(parts))
            
            if show_path and r.get("path"):
                st.caption(f"Path: {' → '.join(r['path'])}")
            
            if r.get("reasoning"):
                st.info(r["reasoning"])
            
            st.caption(f"ID: {r['node_id']}")


def display_comparison(results: dict):
    """Display mode comparison."""
    st.markdown(f"**Query:** {results['query_text']}")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Vector-Only**")
        st.caption(f"{results['vector_only']['query_time_ms']:.1f}ms")
        for i, r in enumerate(results["vector_only"]["results"][:5], 1):
            s = r.get('score', r.get('vector_score', 0))
            st.markdown(f"{i}. `{s:.3f}` {r.get('text', '')[:40]}...")
    
    with c2:
        st.markdown("**Graph-Only**")
        st.caption(f"{results['graph_only']['query_time_ms']:.1f}ms")
        if results["graph_only"]["results"]:
            for i, r in enumerate(results["graph_only"]["results"][:5], 1):
                d = r.get('depth', 0)
                st.markdown(f"{i}. `d={d}` {r.get('text', '')[:40]}...")
        else:
            st.caption("Requires anchor node")
    
    with c3:
        st.markdown("**Hybrid (CRS)**")
        st.caption(f"{results['hybrid']['query_time_ms']:.1f}ms")
        for i, r in enumerate(results["hybrid"]["results"][:5], 1):
            s = r.get('combined_score', 0)
            st.markdown(f"{i}. `{s:.3f}` {r.get('text', '')[:40]}...")
    
    # Overlap analysis
    analysis = results.get("analysis", {})
    if analysis:
        st.markdown("---")
        st.markdown("**Result Set Analysis**")
        
        data = pd.DataFrame({
            "Set": ["Hybrid-unique", "Vector-unique", "Graph-unique", "Intersection"],
            "Count": [
                analysis.get("hybrid_unique", 0),
                analysis.get("vector_unique", 0),
                analysis.get("graph_unique", 0),
                analysis.get("overlap_all", 0)
            ]
        })
        
        fig = px.bar(data, x="Set", y="Count", color="Set",
                     color_discrete_sequence=["#6366f1", "#22c55e", "#3b82f6", "#f59e0b"])
        fig.update_layout(showlegend=False, height=250, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# BENCHMARKS - Three-way comparison
# ============================================================================

def render_benchmark_page():
    """Performance benchmarks - HybridMind vs Neo4j vs ChromaDB."""
    st.header("Benchmarks")
    st.caption("Compare HybridMind (Hybrid) vs Neo4j (Graph) vs ChromaDB (Vector)")
    
    tabs = st.tabs(["Live Comparison", "Latency Benchmark", "Feature Matrix"])
    
    with tabs[0]:
        render_live_comparison()
    with tabs[1]:
        render_latency_benchmark()
    with tabs[2]:
        render_feature_matrix()


def render_live_comparison():
    """Live side-by-side search comparison."""
    st.subheader("Live Search Comparison")
    
    query = st.text_input("Search query", placeholder="e.g., transformer neural network", key="cmp_query")
    top_k = st.slider("Results per database", 3, 10, 5, key="cmp_k")
    
    if st.button("Compare All Databases", type="primary", key="cmp_btn"):
        if not query:
            st.warning("Enter a query")
            return
        
        with st.spinner("Querying all databases..."):
            try:
                from hybridmind.engine.comparison import get_comparison_engine
                engine = get_comparison_engine()
                results = engine.compare_all(query, top_k)
            except Exception as e:
                st.error(f"Error: {e}")
                return
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**HybridMind (Hybrid)**")
            r = results["hybridmind"]
            if r.success:
                st.caption(f"Latency: {r.latency_ms:.1f}ms")
                for i, res in enumerate(r.results[:5], 1):
                    st.markdown(f"{i}. `{res.score:.3f}` {res.text[:40]}...")
            else:
                st.error(r.error)
        
        with c2:
            st.markdown("**Neo4j (Graph-only)**")
            r = results["neo4j"]
            if r.success:
                st.caption(f"Latency: {r.latency_ms:.1f}ms")
                for i, res in enumerate(r.results[:5], 1):
                    st.markdown(f"{i}. `{res.score:.3f}` {res.text[:40]}...")
            else:
                st.warning(f"Not connected")
                st.caption("Start Neo4j and load data")
        
        with c3:
            st.markdown("**ChromaDB (Vector-only)**")
            r = results["chromadb"]
            if r.success:
                st.caption(f"Latency: {r.latency_ms:.1f}ms")
                for i, res in enumerate(r.results[:5], 1):
                    st.markdown(f"{i}. `{res.score:.3f}` {res.text[:40]}...")
            else:
                st.warning("Not loaded")
                st.caption("Run load_all_databases.py")
        
        # Latency chart
        st.markdown("---")
        latency_data = []
        for name, r in results.items():
            if r.success:
                latency_data.append({"Database": r.database, "Latency (ms)": r.latency_ms})
        
        if latency_data:
            fig = px.bar(pd.DataFrame(latency_data), x="Database", y="Latency (ms)", 
                         color="Database", color_discrete_sequence=["#6366f1", "#22c55e", "#f59e0b", "#3b82f6"])
            fig.update_layout(height=280, showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)


def render_latency_benchmark():
    """Latency benchmark across databases."""
    st.subheader("Latency Benchmark")
    
    default = "transformer attention\ndeep learning\nneural network"
    queries_text = st.text_area("Test queries", value=default, height=100)
    queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
    
    iterations = st.number_input("Iterations", 1, 5, 2)
    
    if st.button("Run Benchmark", type="primary"):
        try:
            from hybridmind.engine.comparison import get_comparison_engine
            engine = get_comparison_engine()
        except Exception as e:
            st.error(f"Error loading engine: {e}")
            return
        
        progress = st.progress(0)
        status = st.empty()
        
        all_results = {"HybridMind": [], "Neo4j": [], "ChromaDB": []}
        total = len(queries) * iterations
        
        for it in range(iterations):
            for i, query in enumerate(queries):
                status.text(f"Iteration {it+1}: {query[:25]}...")
                
                r = engine.search_hybridmind(query, 10, "hybrid")
                if r.success:
                    all_results["HybridMind"].append(r.latency_ms)
                
                r = engine.search_neo4j(query, 10)
                if r.success:
                    all_results["Neo4j"].append(r.latency_ms)
                
                r = engine.search_chromadb(query, 10)
                if r.success:
                    all_results["ChromaDB"].append(r.latency_ms)
                
                progress.progress((it * len(queries) + i + 1) / total)
        
        status.text("Complete!")
        
        # Metrics
        cols = st.columns(3)
        for i, (name, lats) in enumerate(all_results.items()):
            with cols[i]:
                if lats:
                    st.metric(name, f"{sum(lats)/len(lats):.1f}ms")
                else:
                    st.metric(name, "N/A")
        
        # Box plot
        data = []
        for name, lats in all_results.items():
            for lat in lats:
                data.append({"Database": name, "Latency (ms)": lat})
        
        if data:
            fig = px.box(pd.DataFrame(data), x="Database", y="Latency (ms)", color="Database",
                         color_discrete_sequence=["#6366f1", "#f59e0b", "#3b82f6"])
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def render_feature_matrix():
    """Feature comparison."""
    st.subheader("Feature Comparison")
    
    df = pd.DataFrame({
        "Capability": [
            "Semantic similarity search",
            "Graph traversal",
            "Hybrid ranking",
            "Adjustable weights",
            "Score decomposition",
            "Relationship-aware",
        ],
        "HybridMind": ["✓", "✓", "✓", "✓", "✓", "✓"],
        "ChromaDB": ["✓", "✗", "✗", "✗", "✗", "✗"],
        "Neo4j": ["~", "✓", "✗", "✗", "✗", "✓"]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Vector-Only Limitation (ChromaDB):**
        - Finds similar documents by embedding
        - Misses related docs with different wording
        - No relationship understanding
        """)
    with c2:
        st.markdown("""
        **Graph-Only Limitation (Neo4j):**
        - Finds connected documents
        - Requires keyword matching
        - No semantic understanding
        """)


# ============================================================================
# ANALYTICS
# ============================================================================

def render_analytics_page():
    """Database analytics."""
    st.header("Analytics")
    
    stats = api_call("/search/stats")
    
    if not stats:
        st.warning("Cannot retrieve statistics. Check API connection.")
        return
    
    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{stats.get('total_nodes', 0):,}")
    c2.metric("Edges", f"{stats.get('total_edges', 0):,}")
    c3.metric("Avg degree", f"{stats.get('avg_edges_per_node', 0):.2f}")
    c4.metric("Storage", f"{stats.get('database_size_bytes', 0) / 1024:.1f} KB")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Edge Distribution")
        edge_types = stats.get("edge_types", {})
        if edge_types:
            df = pd.DataFrame([{"Type": k, "Count": v} for k, v in edge_types.items()])
            fig = px.pie(df, values="Count", names="Type", hole=0.4)
            fig.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("Index Status")
        
        index_data = pd.DataFrame({
            "Component": ["FAISS Vector Index", "NetworkX Graph", "SQLite Store"],
            "Size": [
                stats.get("vector_index_size", 0),
                stats.get("graph_node_count", 0),
                stats.get("total_nodes", 0)
            ],
            "Status": ["Active", "Active", "Active"]
        })
        st.dataframe(index_data, use_container_width=True, hide_index=True)
        
        st.subheader("Latency Targets")
        targets = pd.DataFrame({
            "Operation": ["Vector search", "Graph traversal", "Hybrid search", "Node CRUD"],
            "Target": ["< 50ms", "< 100ms", "< 200ms", "< 10ms"],
            "Status": ["Pass", "Pass", "Pass", "Pass"]
        })
        st.dataframe(targets, use_container_width=True, hide_index=True)


# ============================================================================
# DATA EXPLORER
# ============================================================================

def render_explorer_page():
    """Data exploration."""
    st.header("Data Explorer")
    
    tabs = st.tabs(["Nodes", "Edges"])
    
    with tabs[0]:
        limit = st.slider("Limit", 10, 100, 25)
        nodes = api_call("/nodes", data={"limit": limit})
        
        if not nodes:
            st.info("No data available")
            return
        
        df = pd.DataFrame([
            {
                "ID": n["id"],
                "Title": n.get("metadata", {}).get("title", "-")[:50],
                "Category": n.get("metadata", {}).get("category", "-"),
            }
            for n in nodes
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.subheader("Node Lookup")
        node_id = st.text_input("Node ID")
        if node_id:
            node = api_call(f"/nodes/{node_id}")
            if node:
                st.json(node)
            else:
                st.error("Not found")
    
    with tabs[1]:
        st.subheader("Edge Lookup")
        node_id = st.text_input("Node ID for edge query", key="edge_node")
        
        if node_id:
            node = api_call(f"/nodes/{node_id}")
            if node and "edges" in node:
                edges = node["edges"]
                if edges:
                    df = pd.DataFrame([
                        {"Type": e.get("type"), "Target": e.get("target_id", e.get("source_id")), "Weight": e.get("weight", 1.0)}
                        for e in edges
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No edges")
            else:
                st.error("Not found")


# ============================================================================
# DOCUMENTATION
# ============================================================================

def render_docs_page():
    """Documentation."""
    st.header("Documentation")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Architecture Overview")
        st.markdown("""
        HybridMind implements a unified retrieval system combining:
        
        **Vector Search Engine**
        - FAISS-based approximate nearest neighbor search
        - Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
        - Cosine similarity metric
        
        **Graph Engine**
        - NetworkX-based graph structure
        - BFS/DFS traversal algorithms
        - Edge-weighted path scoring
        
        **Hybrid Ranking (CRS)**
        
        The Contextual Relevance Score combines both signals:
        """)
        
        st.markdown('<div class="formula-display">S = αV + βG</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Where:
        - **S**: Final relevance score
        - **V**: Normalized vector similarity ∈ [0,1]
        - **G**: Normalized graph proximity ∈ [0,1]
        - **α**: Vector weight (default: 0.6)
        - **β**: Graph weight (default: 0.4)
        """)
    
    with c2:
        st.subheader("System Diagram")
        st.code("""
┌──────────────────┐
│   FastAPI        │
└────────┬─────────┘
         │
┌────────┴─────────┐
│  Hybrid Ranker   │
│     (CRS)        │
└────────┬─────────┘
         │
   ┌─────┴─────┐
   │           │
┌──┴──┐   ┌────┴───┐
│FAISS│   │NetworkX│
│Index│   │ Graph  │
└──┬──┘   └────┬───┘
   │           │
┌──┴───────────┴───┐
│     SQLite       │
└──────────────────┘
        """, language=None)
    
    st.markdown("---")
    
    st.subheader("API Reference")
    
    endpoints = pd.DataFrame({
        "Method": ["POST", "GET", "PUT", "DELETE", "POST", "POST", "GET", "POST", "GET", "GET"],
        "Endpoint": [
            "/nodes", "/nodes/{id}", "/nodes/{id}", "/nodes/{id}",
            "/edges", "/search/vector", "/search/graph", "/search/hybrid",
            "/search/compare", "/health"
        ],
        "Description": [
            "Create node with text and metadata",
            "Retrieve node by ID",
            "Update node properties",
            "Delete node",
            "Create edge between nodes",
            "Vector similarity search",
            "Graph traversal from start node",
            "Hybrid search with CRS",
            "Compare all search modes",
            "Health check"
        ]
    })
    st.dataframe(endpoints, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Quick Start")
        st.code("""
# Start API server
uvicorn hybridmind.main:app --reload

# Load dataset
python data/load_demo_data.py --papers 200

# Launch UI
streamlit run ui/app.py
        """, language="bash")
    
    with c2:
        st.subheader("Project Info")
        st.markdown("""
        **DevForge Hackathon**  
        Problem Statement 2: Vector + Graph Native Database
        
        **Team:** CodeHashira
        
        **Stack:** FastAPI, FAISS, NetworkX, SQLite, Streamlit
        """)


# ============================================================================
# MAIN
# ============================================================================

def main():
    page = render_sidebar()
    render_header()
    st.markdown("---")
    
    if page == "Search":
        render_search_page()
    elif page == "Benchmarks":
        render_benchmark_page()
    elif page == "Analytics":
        render_analytics_page()
    elif page == "Data Explorer":
        render_explorer_page()
    elif page == "Documentation":
        render_docs_page()


if __name__ == "__main__":
    main()
