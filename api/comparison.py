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

