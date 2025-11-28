"""Comparison-related Pydantic models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ComparisonSearchRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    graph_weight: float = Field(default=0.4, ge=0.0, le=1.0)


class SystemResult(BaseModel):
    items: List[Dict[str, Any]]
    count: int
    latency_ms: float
    error: Optional[str] = None


class ComparisonResponse(BaseModel):
    query: str
    system_status: Dict[str, bool]
    results: Dict[str, SystemResult]
    analysis: Dict[str, Any]


class BenchmarkRequest(BaseModel):
    queries: List[str]
    top_k: int = Field(default=10, ge=1, le=50)
    iterations: int = Field(default=3, ge=1, le=10)


class SystemBenchmark(BaseModel):
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    total_queries: int
    errors: int


class BenchmarkResponse(BaseModel):
    config: Dict[str, Any]
    results: Dict[str, SystemBenchmark]
    winner: Dict[str, str]
    summary: str


class SystemStatusResponse(BaseModel):
    hybridmind: Dict[str, Any]
    neo4j: Dict[str, Any]
    chromadb: Dict[str, Any]
    all_available: bool
