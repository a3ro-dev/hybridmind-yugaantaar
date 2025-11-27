"""
Comparison API endpoints for HybridMind vs Neo4j vs ChromaDB.
Enables side-by-side comparison and benchmarking across database systems.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Query

from models.comparison import (
    ComparisonSearchRequest,
    ComparisonResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    SystemStatusResponse
)
from engine.comparison import get_comparison_engine
from config import settings

router = APIRouter(prefix="/comparison", tags=["Comparison"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Get availability status of all database systems.
    
    Returns whether each system (HybridMind, Neo4j, ChromaDB) is:
    - Available and connected
    - Data loaded (node/document counts)
    - Connection details
    
    Use this to verify all systems are ready before running comparisons.
    """
    engine = get_comparison_engine(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        chromadb_path=settings.chromadb_path
    )
    
    return engine.get_system_status()


@router.post("/search", response_model=ComparisonResponse)
async def compare_search(request: ComparisonSearchRequest):
    """
    Run the same search query across all three database systems.
    
    **Systems Compared:**
    - **HybridMind**: Hybrid vector + graph search using CRS algorithm
    - **Neo4j**: Pure graph-based search (full-text matching + traversal)
    - **ChromaDB**: Pure vector similarity search
    
    **Returns:**
    - Results from each system
    - Latency comparison
    - Overlap analysis (which results appear in multiple systems)
    - Unique results per system
    
    **Example Use Cases:**
    - Compare retrieval quality across different approaches
    - Identify which system finds unique relevant results
    - Benchmark latency for your queries
    """
    engine = get_comparison_engine(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        chromadb_path=settings.chromadb_path
    )
    
    result = engine.compare_all(
        query_text=request.query_text,
        top_k=request.top_k,
        vector_weight=request.vector_weight,
        graph_weight=request.graph_weight
    )
    
    return result


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest):
    """
    Run a comprehensive performance benchmark across all systems.
    
    **What it measures:**
    - Average, P50, P95, P99 latencies
    - Throughput (queries per second)
    - Result count consistency
    
    **How it works:**
    1. Warm-up run to initialize caches
    2. Run each query multiple times (configurable iterations)
    3. Calculate statistics across all runs
    4. Determine winner by latency and throughput
    
    **Best Practices:**
    - Use representative queries from your use case
    - Run with at least 3 iterations for stable results
    - Include both simple and complex queries
    
    **Note:** This endpoint may take a while for large query sets.
    """
    if len(request.queries) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 queries allowed per benchmark run"
        )
    
    engine = get_comparison_engine(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        chromadb_path=settings.chromadb_path
    )
    
    result = engine.run_benchmark(
        queries=request.queries,
        top_k=request.top_k,
        iterations=request.iterations
    )
    
    return result


@router.get("/quick")
async def quick_compare(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20, description="Results per system")
):
    """
    Quick comparison with minimal parameters.
    
    Simplified endpoint for fast comparisons without full configuration.
    Uses default weights (α=0.6, β=0.4) for HybridMind.
    
    Great for:
    - Quick demos
    - Interactive exploration
    - Testing system availability
    """
    engine = get_comparison_engine(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        chromadb_path=settings.chromadb_path
    )
    
    result = engine.compare_all(
        query_text=query,
        top_k=top_k,
        vector_weight=0.6,
        graph_weight=0.4
    )
    
    # Simplified response for quick view
    return {
        "query": query,
        "hybridmind": {
            "count": result["results"]["hybridmind"]["count"],
            "latency_ms": round(result["results"]["hybridmind"]["latency_ms"], 1),
            "top_result": result["results"]["hybridmind"]["items"][0] if result["results"]["hybridmind"]["items"] else None,
            "available": result["system_status"]["hybridmind"]
        },
        "neo4j": {
            "count": result["results"]["neo4j"]["count"],
            "latency_ms": round(result["results"]["neo4j"]["latency_ms"], 1),
            "top_result": result["results"]["neo4j"]["items"][0] if result["results"]["neo4j"]["items"] else None,
            "available": result["system_status"]["neo4j"]
        },
        "chromadb": {
            "count": result["results"]["chromadb"]["count"],
            "latency_ms": round(result["results"]["chromadb"]["latency_ms"], 1),
            "top_result": result["results"]["chromadb"]["items"][0] if result["results"]["chromadb"]["items"] else None,
            "available": result["system_status"]["chromadb"]
        },
        "analysis": {
            "common_to_all": result["analysis"]["common_to_all"],
            "fastest": result["analysis"]["latency_comparison"]["fastest"]
        }
    }


@router.get("/sample-queries")
async def get_sample_queries():
    """
    Get sample queries for testing comparisons.
    
    Returns a curated list of queries that work well for demonstrating
    differences between vector, graph, and hybrid search.
    """
    return {
        "semantic_queries": [
            "deep learning neural networks",
            "natural language processing transformers",
            "computer vision image classification",
            "reinforcement learning reward optimization",
            "generative adversarial networks"
        ],
        "conceptual_queries": [
            "how machines learn from data",
            "understanding human language with AI",
            "teaching robots to see",
            "algorithms that improve themselves",
            "artificial creativity and generation"
        ],
        "specific_queries": [
            "attention mechanism in transformers",
            "convolutional neural network architecture",
            "backpropagation gradient descent",
            "recurrent neural network sequence modeling",
            "transfer learning pretrained models"
        ],
        "benchmark_suite": [
            "machine learning",
            "deep learning neural networks",
            "natural language processing",
            "computer vision",
            "reinforcement learning",
            "transformer attention mechanism",
            "convolutional neural network",
            "knowledge graph embedding",
            "semantic similarity search",
            "graph neural network"
        ]
    }


@router.post("/effectiveness")
async def evaluate_effectiveness(request: ComparisonSearchRequest):
    """
    Evaluate search effectiveness with quantitative metrics.
    
    **Metrics Computed:**
    - **Precision@K**: Fraction of retrieved documents that are relevant
    - **Recall@K**: Fraction of relevant documents retrieved
    - **MRR**: Mean Reciprocal Rank (1/rank of first relevant result)
    - **NDCG**: Normalized Discounted Cumulative Gain
    - **Coverage**: Percentage of relevant set found
    
    **Comparison:**
    Shows percentage improvement of hybrid search over vector-only and graph-only.
    
    **Relevance Determination:**
    Uses pooled relevance strategy - results appearing in multiple systems
    or with high scores are considered "relevant".
    
    This endpoint provides the quantitative proof that hybrid search
    improves retrieval quality over single-mode approaches.
    """
    engine = get_comparison_engine(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        chromadb_path=settings.chromadb_path
    )
    
    result = engine.compare_all(
        query_text=request.query_text,
        top_k=request.top_k,
        vector_weight=request.vector_weight,
        graph_weight=request.graph_weight,
        include_effectiveness=True
    )
    
    return {
        "query": request.query_text,
        "top_k": request.top_k,
        "weights": {
            "vector_weight": request.vector_weight,
            "graph_weight": request.graph_weight
        },
        "effectiveness_metrics": result.get("effectiveness_metrics", {}),
        "interpretation": _interpret_effectiveness(result.get("effectiveness_metrics", {}))
    }


def _interpret_effectiveness(metrics: dict) -> dict:
    """Generate human-readable interpretation of effectiveness metrics."""
    if not metrics:
        return {"error": "No effectiveness metrics available"}
    
    hybrid = metrics.get("hybridmind", {})
    vector = metrics.get("vector_only", {})
    improvements = metrics.get("improvements", {})
    
    interpretation = {
        "headline": "",
        "details": [],
        "recommendation": ""
    }
    
    # Headline based on winner
    winner = metrics.get("winner", "")
    if winner == "hybrid":
        interpretation["headline"] = "✓ Hybrid search demonstrates superior retrieval quality"
    elif winner == "vector":
        interpretation["headline"] = "Vector search leads, but hybrid provides unique value"
    else:
        interpretation["headline"] = "Graph search excels for relationship-focused queries"
    
    # Detailed improvements
    prec_imp = improvements.get("precision_vs_vector_pct", 0)
    ndcg_imp = improvements.get("ndcg_vs_vector_pct", 0)
    unique = improvements.get("unique_relevant_by_hybrid", 0)
    
    if prec_imp > 0:
        interpretation["details"].append(
            f"Precision improved by {prec_imp:+.1f}% over vector-only search"
        )
    if ndcg_imp > 0:
        interpretation["details"].append(
            f"NDCG (ranking quality) improved by {ndcg_imp:+.1f}%"
        )
    if unique > 0:
        interpretation["details"].append(
            f"Hybrid found {unique} relevant results that vector-only missed"
        )
    
    # Recommendation
    if hybrid.get("ndcg", 0) > 0.5:
        interpretation["recommendation"] = "Hybrid search is effective for this query type"
    else:
        interpretation["recommendation"] = "Consider tuning weights or using anchor nodes for better results"
    
    return interpretation


@router.post("/ablation")
async def run_ablation_study(
    query: str = Query(..., min_length=1, description="Test query"),
    top_k: int = Query(default=10, ge=1, le=50, description="Results per search")
):
    """
    Run ablation study to justify default weights (α=0.6, β=0.4).
    
    Tests multiple weight combinations and evaluates effectiveness metrics
    for each, showing which weights perform best for the given query.
    
    **Weight Combinations Tested:**
    - α=0.1, β=0.9 (Graph-heavy)
    - α=0.3, β=0.7
    - α=0.5, β=0.5 (Balanced)
    - α=0.6, β=0.4 (Default)
    - α=0.7, β=0.3
    - α=0.9, β=0.1 (Vector-heavy)
    - α=1.0, β=0.0 (Pure vector)
    
    **Returns:**
    - NDCG, Precision, and MRR for each weight combination
    - Best weights by NDCG
    - Whether default weights are optimal or near-optimal
    - Recommendation for this query type
    
    Use this to provide data-driven justification for the CRS weights.
    """
    try:
        from api.dependencies import get_db_manager
        from engine.effectiveness import get_effectiveness_calculator
        
        db = get_db_manager()
        hybrid_ranker = db.hybrid_ranker
        
        calc = get_effectiveness_calculator()
        result = calc.run_ablation_study(
            query_text=query,
            hybrid_ranker=hybrid_ranker,
            top_k=top_k
        )
        
        return {
            "status": "success",
            **result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ablation study failed: {str(e)}"
        )


@router.get("/effectiveness/summary")
async def get_effectiveness_summary():
    """
    Get summary of effectiveness across multiple benchmark queries.
    
    Runs effectiveness evaluation on a suite of test queries and
    aggregates the results to show overall hybrid search performance.
    
    This provides the definitive proof that hybrid search improves
    retrieval quality across diverse query types.
    """
    benchmark_queries = [
        "deep learning neural networks",
        "natural language processing",
        "attention mechanism transformers",
        "reinforcement learning",
        "knowledge graph embedding"
    ]
    
    engine = get_comparison_engine(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        chromadb_path=settings.chromadb_path
    )
    
    all_results = []
    hybrid_wins = 0
    total_precision_improvement = 0
    total_ndcg_improvement = 0
    
    for query in benchmark_queries:
        result = engine.compare_all(
            query_text=query,
            top_k=10,
            include_effectiveness=True
        )
        
        effectiveness = result.get("effectiveness_metrics", {})
        if effectiveness:
            all_results.append({
                "query": query,
                "hybrid_ndcg": effectiveness.get("hybridmind", {}).get("ndcg", 0),
                "vector_ndcg": effectiveness.get("vector_only", {}).get("ndcg", 0),
                "winner": effectiveness.get("winner", ""),
                "improvement_pct": effectiveness.get("improvements", {}).get("ndcg_vs_vector_pct", 0)
            })
            
            if effectiveness.get("winner") == "hybrid":
                hybrid_wins += 1
            
            total_precision_improvement += effectiveness.get("improvements", {}).get("precision_vs_vector_pct", 0)
            total_ndcg_improvement += effectiveness.get("improvements", {}).get("ndcg_vs_vector_pct", 0)
    
    n = len(all_results)
    
    return {
        "summary": {
            "queries_evaluated": n,
            "hybrid_wins": hybrid_wins,
            "win_rate": f"{hybrid_wins/n*100:.1f}%" if n > 0 else "0%",
            "avg_precision_improvement": f"{total_precision_improvement/n:+.1f}%" if n > 0 else "0%",
            "avg_ndcg_improvement": f"{total_ndcg_improvement/n:+.1f}%" if n > 0 else "0%"
        },
        "per_query_results": all_results,
        "conclusion": (
            f"Hybrid search outperforms vector-only in {hybrid_wins}/{n} queries "
            f"with average NDCG improvement of {total_ndcg_improvement/n:+.1f}%."
            if n > 0 else "No results available"
        )
    }

