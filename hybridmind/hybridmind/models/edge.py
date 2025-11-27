"""
Edge data models for HybridMind.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EdgeCreate(BaseModel):
    """Request model for creating a new edge."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_id": "550e8400-e29b-41d4-a716-446655440000",
                "target_id": "550e8400-e29b-41d4-a716-446655440001",
                "type": "cites",
                "weight": 0.9,
                "metadata": {"context": "methodology section"}
            }
        }
    )
    
    source_id: str = Field(
        ...,
        description="Source node ID (UUID)"
    )
    target_id: str = Field(
        ...,
        description="Target node ID (UUID)"
    )
    type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Relationship type (e.g., 'cites', 'authored_by', 'related_to')"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Edge weight/strength (0.0 to 1.0)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional edge metadata"
    )
    
    @field_validator('source_id', 'target_id')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ID cannot be empty")
        return v


class EdgeUpdate(BaseModel):
    """Request model for updating an edge."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "weight": 0.95,
                "metadata": {"context": "updated context"}
            }
        }
    )
    
    type: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Updated relationship type"
    )
    weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Updated edge weight"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated metadata"
    )


class EdgeResponse(BaseModel):
    """Response model for edge operations."""
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "edge-550e8400-e29b-41d4-a716-446655440000",
                "source_id": "550e8400-e29b-41d4-a716-446655440000",
                "target_id": "550e8400-e29b-41d4-a716-446655440001",
                "type": "cites",
                "weight": 0.9,
                "metadata": {"context": "methodology section"},
                "created_at": "2025-11-27T10:00:00Z"
            }
        }
    )
    
    id: str = Field(description="Unique edge identifier (UUID)")
    source_id: str = Field(description="Source node ID")
    target_id: str = Field(description="Target node ID")
    type: str = Field(description="Relationship type")
    weight: float = Field(description="Edge weight")
    metadata: Dict[str, Any] = Field(description="Edge metadata")
    created_at: datetime = Field(description="Creation timestamp")


class EdgeDeleteResponse(BaseModel):
    """Response model for edge deletion."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "deleted": True,
                "edge_id": "edge-550e8400-e29b-41d4-a716-446655440000"
            }
        }
    )
    
    deleted: bool = Field(description="Whether deletion was successful")
    edge_id: str = Field(description="ID of the deleted edge")
