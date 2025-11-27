"""
Search request and response models for HybridMind.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class VectorSearchRequest(BaseModel):
    """Request model for vector similarity search."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_text": "transformer attention mechanisms in NLP",
                "top_k": 10,
                "min_score": 0.5
            }
        }
    )
    
    query_text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Search query text"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filter results by metadata fields"
    )


class GraphSearchRequest(BaseModel):
    """Request model for graph traversal search."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_id": "550e8400-e29b-41d4-a716-446655440000",
                "depth": 2,
                "edge_types": ["cites", "related_to"],
                "direction": "both"
            }
        }
    )
    
    start_id: str = Field(
        ...,
        description="Starting node ID for traversal"
    )
    depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum traversal depth"
    )
    edge_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by specific edge types"
    )
    direction: str = Field(
        default="both",
        description="Traversal direction: 'outgoing', 'incoming', or 'both'"
    )
    
    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v: str) -> str:
        valid = ['outgoing', 'incoming', 'both']
        if v not in valid:
            raise ValueError(f"direction must be one of {valid}")
        return v


class HybridSearchRequest(BaseModel):
    """Request model for hybrid vector + graph search."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_text": "deep learning optimization techniques",
                "top_k": 10,
                "vector_weight": 0.6,
                "graph_weight": 0.4,
                "anchor_nodes": ["550e8400-e29b-41d4-a716-446655440000"],
                "max_depth": 2
            }
        }
    )
    
    query_text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Search query text"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity score (α)"
    )
    graph_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for graph proximity score (β)"
    )
    anchor_nodes: Optional[List[str]] = Field(
        default=None,
        description="Node IDs to anchor graph search (optional)"
    )
    max_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum graph traversal depth"
    )
    edge_type_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for different edge types (optional)"
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum combined score threshold"
    )
    
    @field_validator('graph_weight')
    @classmethod
    def validate_weights(cls, v: float, info) -> float:
        # Note: We don't enforce sum=1 to allow flexibility
        return v


class SearchResult(BaseModel):
    """Individual search result."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "node_id": "550e8400-e29b-41d4-a716-446655440001",
                "text": "BERT is a transformer-based model...",
                "metadata": {"title": "BERT Paper"},
                "vector_score": 0.85,
                "graph_score": 0.72,
                "combined_score": 0.80,
                "reasoning": "High semantic similarity (0.85) + 2-hop citation connection"
            }
        }
    )
    
    node_id: str = Field(description="Node ID")
    text: str = Field(description="Node text content")
    metadata: Dict[str, Any] = Field(description="Node metadata")
    vector_score: Optional[float] = Field(
        default=None,
        description="Vector similarity score (cosine)"
    )
    graph_score: Optional[float] = Field(
        default=None,
        description="Graph proximity score"
    )
    combined_score: Optional[float] = Field(
        default=None,
        description="Hybrid combined score (CRS)"
    )
    depth: Optional[int] = Field(
        default=None,
        description="Distance from start node (graph search)"
    )
    path: Optional[List[str]] = Field(
        default=None,
        description="Path from anchor node (graph search)"
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of why this result was returned"
    )


class SearchResponse(BaseModel):
    """Response model for search operations."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "node_id": "node-1",
                        "text": "Attention mechanisms...",
                        "metadata": {},
                        "vector_score": 0.85,
                        "graph_score": 0.72,
                        "combined_score": 0.80,
                        "reasoning": "High semantic similarity"
                    }
                ],
                "query_time_ms": 45.2,
                "total_candidates": 150,
                "search_type": "hybrid"
            }
        }
    )
    
    results: List[SearchResult] = Field(description="Search results")
    query_time_ms: float = Field(description="Query execution time in milliseconds")
    total_candidates: int = Field(
        default=0,
        description="Total number of candidates evaluated"
    )
    search_type: str = Field(
        default="hybrid",
        description="Type of search performed: 'vector', 'graph', or 'hybrid'"
    )


class StatsResponse(BaseModel):
    """Response model for database statistics."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_nodes": 500,
                "total_edges": 1200,
                "edge_types": {"cites": 800, "authored_by": 400},
                "avg_edges_per_node": 2.4,
                "vector_index_size": 500,
                "database_size_bytes": 5242880
            }
        }
    )
    
    total_nodes: int
    total_edges: int
    edge_types: Dict[str, int]
    avg_edges_per_node: float
    vector_index_size: int
    database_size_bytes: int
