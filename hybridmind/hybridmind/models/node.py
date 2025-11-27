"""
Node data models for HybridMind.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class NodeCreate(BaseModel):
    """Request model for creating a new node."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Attention mechanisms allow neural networks to focus on relevant parts of the input.",
                "metadata": {
                    "title": "Attention Is All You Need",
                    "tags": ["transformer", "attention", "NLP"],
                    "source": "arxiv",
                    "year": 2017
                }
            }
        }
    )
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Text content of the node"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata (title, tags, source, etc.)"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Pre-computed embedding vector (optional, will be generated if not provided)"
    )


class NodeUpdate(BaseModel):
    """Request model for updating a node."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Updated text content about transformers",
                "metadata": {"tags": ["transformer", "updated"]},
                "regenerate_embedding": True
            }
        }
    )
    
    text: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=50000,
        description="Updated text content"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated metadata"
    )
    regenerate_embedding: bool = Field(
        default=True,
        description="Whether to regenerate the embedding after text update"
    )


class EdgeSummary(BaseModel):
    """Summary of an edge connected to a node."""
    
    edge_id: str
    target_id: str
    type: str
    weight: float
    direction: str = Field(description="'outgoing' or 'incoming'")


class NodeResponse(BaseModel):
    """Response model for node operations."""
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "text": "Attention mechanisms allow neural networks...",
                "metadata": {"title": "Attention Is All You Need"},
                "created_at": "2025-11-27T10:00:00Z",
                "updated_at": "2025-11-27T10:00:00Z",
                "edges": [
                    {
                        "edge_id": "edge-1",
                        "target_id": "node-2",
                        "type": "cites",
                        "weight": 1.0,
                        "direction": "outgoing"
                    }
                ]
            }
        }
    )
    
    id: str = Field(description="Unique node identifier (UUID)")
    text: str = Field(description="Text content of the node")
    metadata: Dict[str, Any] = Field(description="Node metadata")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    edges: List[EdgeSummary] = Field(
        default_factory=list,
        description="Connected edges"
    )


class NodeWithEmbedding(NodeResponse):
    """Node response including the embedding vector."""
    
    embedding: List[float] = Field(description="Vector embedding of the node")


class NodeDeleteResponse(BaseModel):
    """Response model for node deletion."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "deleted": True,
                "node_id": "550e8400-e29b-41d4-a716-446655440000",
                "edges_removed": 5
            }
        }
    )
    
    deleted: bool = Field(description="Whether deletion was successful")
    node_id: str = Field(description="ID of the deleted node")
    edges_removed: int = Field(description="Number of edges removed")
