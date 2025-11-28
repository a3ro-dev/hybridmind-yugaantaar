"""Edge-related Pydantic models."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class EdgeCreate(BaseModel):
    source_id: str
    target_id: str
    type: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class EdgeUpdate(BaseModel):
    type: Optional[str] = None
    weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class EdgeResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    type: str
    weight: float
    metadata: Dict[str, Any]
    created_at: datetime


class EdgeDeleteResponse(BaseModel):
    deleted: bool
    edge_id: str
