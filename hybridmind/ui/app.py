"""
HybridMind Streamlit UI - Interactive demo interface.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="HybridMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
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
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .vector-score { background: #4CAF50; color: white; }
    .graph-score { background: #2196F3; color: white; }
    .hybrid-score { background: #9C27B0; color: white; }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")


def api_call(endpoint: str, method: str = "GET", data: dict = None):
    """Make API call to HybridMind backend."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=data, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the server is running.")
        return None
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        return None


def render_header():
    """Render page header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<p class="main-header">🧠 HybridMind</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Vector + Graph Native Database for AI Retrieval</p>', unsafe_allow_html=True)
    with col2:
        # Health check
        health = api_call("/health")
        if health and health.get("status") == "healthy":
            st.success(f"✅ Connected | {health.get('nodes', 0)} nodes")
        else:
            st.error("❌ Disconnected")


def render_stats():
    """Render database statistics."""
    stats = api_call("/search/stats")
    if not stats:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Nodes", stats["total_nodes"])
    with col2:
        st.metric("Total Edges", stats["total_edges"])
    with col3:
        st.metric("Avg Edges/Node", f"{stats['avg_edges_per_node']:.2f}")
    with col4:
        st.metric("DB Size", f"{stats['database_size_bytes'] / 1024:.1f} KB")
    
    if stats.get("edge_types"):
        st.subheader("Edge Type Distribution")
        df = pd.DataFrame([
            {"Type": k, "Count": v}
            for k, v in stats["edge_types"].items()
        ])
        fig = px.pie(df, values="Count", names="Type", hole=0.4)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def render_search():
    """Render search interface."""
    st.subheader("🔍 Search")
    
    # Search mode tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Hybrid Search", "Vector Search", "Graph Search", "Compare Modes"])
    
    with tab1:
        render_hybrid_search()
    
    with tab2:
        render_vector_search()
    
    with tab3:
        render_graph_search()
    
    with tab4:
        render_compare_search()


def render_hybrid_search():
    """Render hybrid search interface."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Search Query", placeholder="e.g., transformer attention mechanisms", key="hybrid_query")
    
    with col2:
        top_k = st.slider("Results", 5, 50, 10, key="hybrid_k")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        vector_weight = st.slider("Vector Weight (α)", 0.0, 1.0, 0.6, 0.1)
    
    with col4:
        graph_weight = st.slider("Graph Weight (β)", 0.0, 1.0, 0.4, 0.1)
    
    with col5:
        anchor = st.text_input("Anchor Node ID (optional)", key="hybrid_anchor")
    
    if st.button("🚀 Hybrid Search", type="primary", key="hybrid_btn"):
        if not query:
            st.warning("Please enter a search query")
            return
        
        payload = {
            "query_text": query,
            "top_k": top_k,
            "vector_weight": vector_weight,
            "graph_weight": graph_weight
        }
        if anchor:
            payload["anchor_nodes"] = [anchor]
        
        with st.spinner("Searching..."):
            results = api_call("/search/hybrid", method="POST", data=payload)
        
        if results:
            render_search_results(results, show_all_scores=True)


def render_vector_search():
    """Render vector search interface."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Search Query", placeholder="e.g., deep learning optimization", key="vector_query")
    
    with col2:
        top_k = st.slider("Results", 5, 50, 10, key="vector_k")
    
    if st.button("🔍 Vector Search", type="primary", key="vector_btn"):
        if not query:
            st.warning("Please enter a search query")
            return
        
        with st.spinner("Searching..."):
            results = api_call("/search/vector", method="POST", data={
                "query_text": query,
                "top_k": top_k
            })
        
        if results:
            render_search_results(results, score_key="vector_score")


def render_graph_search():
    """Render graph search interface."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        start_id = st.text_input("Start Node ID", placeholder="e.g., paper-transformer", key="graph_start")
    
    with col2:
        depth = st.slider("Depth", 1, 5, 2, key="graph_depth")
    
    with col3:
        direction = st.selectbox("Direction", ["both", "outgoing", "incoming"], key="graph_dir")
    
    if st.button("🌐 Graph Traverse", type="primary", key="graph_btn"):
        if not start_id:
            st.warning("Please enter a start node ID")
            return
        
        with st.spinner("Traversing graph..."):
            results = api_call("/search/graph", data={
                "start_id": start_id,
                "depth": depth,
                "direction": direction
            })
        
        if results:
            render_search_results(results, score_key="graph_score", show_path=True)


def render_compare_search():
    """Render search comparison interface."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Search Query", placeholder="e.g., language model pre-training", key="compare_query")
    
    with col2:
        anchor = st.text_input("Anchor Node (optional)", key="compare_anchor")
    
    if st.button("📊 Compare All Modes", type="primary", key="compare_btn"):
        if not query:
            st.warning("Please enter a search query")
            return
        
        payload = {"query_text": query, "top_k": 5}
        if anchor:
            payload["anchor_nodes"] = [anchor]
        
        with st.spinner("Running comparisons..."):
            results = api_call("/search/compare", method="POST", data=payload)
        
        if results:
            render_comparison_results(results)


def render_search_results(results: dict, score_key: str = "combined_score", show_all_scores: bool = False, show_path: bool = False):
    """Render search results."""
    st.markdown(f"**Query Time:** {results['query_time_ms']:.2f}ms | **Candidates:** {results['total_candidates']}")
    
    if not results["results"]:
        st.info("No results found")
        return
    
    for i, result in enumerate(results["results"], 1):
        with st.expander(f"#{i} | {result.get('metadata', {}).get('title', result['node_id'][:20])}...", expanded=i <= 3):
            # Scores
            score_cols = st.columns(4)
            
            if show_all_scores:
                with score_cols[0]:
                    vs = result.get("vector_score", 0)
                    st.markdown(f'<span class="score-badge vector-score">Vector: {vs:.3f}</span>', unsafe_allow_html=True)
                with score_cols[1]:
                    gs = result.get("graph_score", 0)
                    st.markdown(f'<span class="score-badge graph-score">Graph: {gs:.3f}</span>', unsafe_allow_html=True)
                with score_cols[2]:
                    cs = result.get("combined_score", 0)
                    st.markdown(f'<span class="score-badge hybrid-score">Combined: {cs:.3f}</span>', unsafe_allow_html=True)
            else:
                score = result.get(score_key, 0)
                with score_cols[0]:
                    st.metric("Score", f"{score:.3f}")
            
            # Content
            st.markdown(f"**Text:** {result['text'][:300]}...")
            
            # Metadata
            if result.get("metadata"):
                meta = result["metadata"]
                cols = st.columns(3)
                if "year" in meta:
                    cols[0].write(f"📅 {meta['year']}")
                if "tags" in meta:
                    cols[1].write(f"🏷️ {', '.join(meta['tags'][:3])}")
                if "venue" in meta:
                    cols[2].write(f"📍 {meta['venue']}")
            
            # Path for graph search
            if show_path and result.get("path"):
                st.write(f"**Path:** {' → '.join(result['path'])}")
            
            # Reasoning
            if result.get("reasoning"):
                st.caption(f"💡 {result['reasoning']}")
            
            st.caption(f"ID: `{result['node_id']}`")


def render_comparison_results(results: dict):
    """Render search mode comparison."""
    st.subheader(f"Query: \"{results['query_text']}\"")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔵 Vector-Only")
        st.caption(f"Time: {results['vector_only']['query_time_ms']:.2f}ms")
        for i, r in enumerate(results["vector_only"]["results"][:5], 1):
            st.markdown(f"**{i}.** [{r['score']:.3f}] {r['text'][:60]}...")
    
    with col2:
        st.markdown("### 🟢 Graph-Only")
        st.caption(f"Time: {results['graph_only']['query_time_ms']:.2f}ms")
        if results["graph_only"]["results"]:
            for i, r in enumerate(results["graph_only"]["results"][:5], 1):
                st.markdown(f"**{i}.** [d={r.get('depth', 0)}] {r['text'][:60]}...")
        else:
            st.info("No anchor provided")
    
    with col3:
        st.markdown("### 🟣 Hybrid")
        st.caption(f"Time: {results['hybrid']['query_time_ms']:.2f}ms")
        for i, r in enumerate(results["hybrid"]["results"][:5], 1):
            st.markdown(f"**{i}.** [{r['combined_score']:.3f}] {r['text'][:60]}...")
    
    # Analysis
    st.subheader("📊 Analysis")
    analysis = results["analysis"]
    
    data = {
        "Metric": ["Hybrid Unique", "Vector Unique", "Graph Unique", "Overlap All"],
        "Count": [
            analysis["hybrid_unique"],
            analysis["vector_unique"],
            analysis["graph_unique"],
            analysis["overlap_all"]
        ]
    }
    
    fig = px.bar(pd.DataFrame(data), x="Metric", y="Count", color="Metric")
    st.plotly_chart(fig, use_container_width=True)


def render_node_explorer():
    """Render node explorer interface."""
    st.subheader("📚 Node Explorer")
    
    # List nodes
    nodes = api_call("/nodes", data={"limit": 20})
    
    if not nodes:
        st.info("No nodes found. Load the demo dataset first.")
        return
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            "ID": n["id"][:12] + "...",
            "Title": n.get("metadata", {}).get("title", "Untitled")[:40],
            "Year": n.get("metadata", {}).get("year", "N/A"),
            "Edges": len(n.get("edges", []))
        }
        for n in nodes
    ])
    
    st.dataframe(df, use_container_width=True)
    
    # Node details
    st.subheader("Node Details")
    node_id = st.text_input("Enter Node ID to view details")
    
    if node_id:
        node = api_call(f"/nodes/{node_id}")
        if node:
            st.json(node)


def render_sidebar():
    """Render sidebar."""
    st.sidebar.title("🧠 HybridMind")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🔍 Search", "📊 Statistics", "📚 Explorer"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    HybridMind combines:
    - **Vector Search**: Semantic similarity
    - **Graph Search**: Relationship traversal
    - **Hybrid Search**: CRS algorithm
    
    CRS = α × V + β × G
    """)
    
    return page


def main():
    """Main application entry point."""
    page = render_sidebar()
    render_header()
    
    st.markdown("---")
    
    if "Search" in page:
        render_search()
    elif "Statistics" in page:
        render_stats()
    elif "Explorer" in page:
        render_node_explorer()


if __name__ == "__main__":
    main()

