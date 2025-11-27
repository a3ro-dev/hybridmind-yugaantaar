"""
Bulk operations API endpoints for HybridMind.
Fast batch import for nodes and edges.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import (
    get_sqlite_store,
    get_vector_index,
    get_graph_index,
    get_embedding_engine,
    get_db_manager,
)
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.graph_index import GraphIndex
from engine.embedding import EmbeddingEngine
from engine.cache import invalidate_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bulk", tags=["Bulk Operations"])


# ==================== Request/Response Models ====================

class BulkNodeCreate(BaseModel):
    """Single node in bulk create request."""
    id: Optional[str] = Field(default=None, description="Optional custom ID")
    text: str = Field(..., min_length=1, max_length=50000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulkNodesRequest(BaseModel):
    """Bulk node creation request."""
    nodes: List[BulkNodeCreate] = Field(..., min_length=1, max_length=1000)
    generate_embeddings: bool = Field(
        default=True,
        description="Generate embeddings for nodes"
    )


class BulkEdgeCreate(BaseModel):
    """Single edge in bulk create request."""
    id: Optional[str] = Field(default=None, description="Optional custom ID")
    source_id: str
    target_id: str
    type: str = Field(..., min_length=1)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BulkEdgesRequest(BaseModel):
    """Bulk edge creation request."""
    edges: List[BulkEdgeCreate] = Field(..., min_length=1, max_length=5000)
    skip_validation: bool = Field(
        default=False,
        description="Skip node existence validation for faster import"
    )


class BulkResult(BaseModel):
    """Result of bulk operation."""
    success: bool
    created: int
    failed: int
    errors: List[str]
    elapsed_ms: float


class BulkImportRequest(BaseModel):
    """Combined bulk import of nodes and edges."""
    nodes: List[BulkNodeCreate] = Field(default_factory=list)
    edges: List[BulkEdgeCreate] = Field(default_factory=list)
    generate_embeddings: bool = Field(default=True)


class BulkImportResult(BaseModel):
    """Result of combined bulk import."""
    nodes: BulkResult
    edges: BulkResult
    total_elapsed_ms: float


# ==================== Endpoints ====================

@router.post("/nodes", response_model=BulkResult)
async def bulk_create_nodes(
    request: BulkNodesRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
):
    """
    Bulk create nodes with optional embedding generation.
    
    Features:
    - Batch embedding generation for efficiency
    - Batch vector index insertion
    - Automatic ID generation if not provided
    
    Maximum 1000 nodes per request.
    """
    start_time = time.perf_counter()
    created = 0
    failed = 0
    errors = []
    
    nodes_to_create = []
    
    # Prepare nodes
    for node in request.nodes:
        node_id = node.id or f"node_{uuid.uuid4().hex[:12]}"
        
        # Check if ID already exists
        if sqlite_store.get_node(node_id):
            errors.append(f"Node {node_id} already exists")
            failed += 1
            continue
        
        nodes_to_create.append({
            "id": node_id,
            "text": node.text,
            "metadata": node.metadata
        })
    
    if not nodes_to_create:
        return BulkResult(
            success=False,
            created=0,
            failed=failed,
            errors=errors,
            elapsed_ms=round((time.perf_counter() - start_time) * 1000, 2)
        )
    
    # Generate embeddings in batch
    embeddings = None
    if request.generate_embeddings:
        texts = [n["text"] for n in nodes_to_create]
        try:
            embeddings = embedding_engine.embed_batch(texts, show_progress=False)
        except Exception as e:
            errors.append(f"Embedding generation failed: {str(e)}")
            embeddings = None
    
    # Create nodes in database
    vector_batch = []
    
    for i, node_data in enumerate(nodes_to_create):
        try:
            embedding = embeddings[i] if embeddings is not None else None
            
            # Create in SQLite
            sqlite_store.create_node(
                node_id=node_data["id"],
                text=node_data["text"],
                metadata=node_data["metadata"],
                embedding=embedding
            )
            
            # Add to graph index
            graph_index.add_node(node_data["id"])
            
            # Prepare vector batch
            if embedding is not None:
                vector_batch.append((node_data["id"], embedding))
            
            created += 1
            
        except Exception as e:
            errors.append(f"Failed to create node {node_data['id']}: {str(e)}")
            failed += 1
    
    # Batch add to vector index
    if vector_batch:
        try:
            vector_index.add_batch(vector_batch)
        except Exception as e:
            errors.append(f"Vector index batch add failed: {str(e)}")
    
    # Invalidate cache
    invalidate_cache()
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Bulk created {created} nodes in {elapsed:.0f}ms")
    
    return BulkResult(
        success=failed == 0,
        created=created,
        failed=failed,
        errors=errors[:10],  # Limit error messages
        elapsed_ms=round(elapsed, 2)
    )


@router.post("/edges", response_model=BulkResult)
async def bulk_create_edges(
    request: BulkEdgesRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    graph_index: GraphIndex = Depends(get_graph_index),
):
    """
    Bulk create edges between existing nodes.
    
    Features:
    - Optional node existence validation (disable for faster import)
    - Automatic ID generation if not provided
    
    Maximum 5000 edges per request.
    """
    start_time = time.perf_counter()
    created = 0
    failed = 0
    errors = []
    
    # Validate nodes exist (unless skip_validation)
    if not request.skip_validation:
        node_ids = set()
        for edge in request.edges:
            node_ids.add(edge.source_id)
            node_ids.add(edge.target_id)
        
        # Check which nodes exist
        missing = []
        for node_id in node_ids:
            if not sqlite_store.get_node(node_id):
                missing.append(node_id)
        
        if missing:
            return BulkResult(
                success=False,
                created=0,
                failed=len(request.edges),
                errors=[f"Missing nodes: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"],
                elapsed_ms=round((time.perf_counter() - start_time) * 1000, 2)
            )
    
    # Create edges
    for edge in request.edges:
        edge_id = edge.id or f"edge_{uuid.uuid4().hex[:12]}"
        
        try:
            # Create in SQLite
            sqlite_store.create_edge(
                edge_id=edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.type,
                weight=edge.weight,
                metadata=edge.metadata
            )
            
            # Add to graph index
            graph_index.add_edge(
                edge_id=edge_id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                edge_type=edge.type,
                weight=edge.weight
            )
            
            created += 1
            
        except Exception as e:
            errors.append(f"Failed to create edge {edge_id}: {str(e)}")
            failed += 1
    
    # Invalidate cache
    invalidate_cache()
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Bulk created {created} edges in {elapsed:.0f}ms")
    
    return BulkResult(
        success=failed == 0,
        created=created,
        failed=failed,
        errors=errors[:10],
        elapsed_ms=round(elapsed, 2)
    )


@router.post("/import", response_model=BulkImportResult)
async def bulk_import(
    request: BulkImportRequest,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine),
):
    """
    Combined bulk import of nodes and edges.
    
    Imports nodes first, then edges. Useful for loading
    complete knowledge graphs in a single request.
    """
    total_start = time.perf_counter()
    
    # Import nodes
    node_result = await bulk_create_nodes(
        BulkNodesRequest(
            nodes=request.nodes,
            generate_embeddings=request.generate_embeddings
        ),
        sqlite_store=sqlite_store,
        vector_index=vector_index,
        graph_index=graph_index,
        embedding_engine=embedding_engine,
    ) if request.nodes else BulkResult(
        success=True, created=0, failed=0, errors=[], elapsed_ms=0
    )
    
    # Import edges (skip validation since we just created the nodes)
    edge_result = await bulk_create_edges(
        BulkEdgesRequest(
            edges=request.edges,
            skip_validation=False  # Still validate in case of partial node failures
        ),
        sqlite_store=sqlite_store,
        graph_index=graph_index,
    ) if request.edges else BulkResult(
        success=True, created=0, failed=0, errors=[], elapsed_ms=0
    )
    
    total_elapsed = (time.perf_counter() - total_start) * 1000
    
    logger.info(
        f"Bulk import complete: {node_result.created} nodes, "
        f"{edge_result.created} edges in {total_elapsed:.0f}ms"
    )
    
    return BulkImportResult(
        nodes=node_result,
        edges=edge_result,
        total_elapsed_ms=round(total_elapsed, 2)
    )


@router.delete("/clear", response_model=dict)
async def clear_all_data():
    """
    Clear all data from the database.
    
    **WARNING**: This permanently deletes all nodes, edges, and indexes.
    Use with caution.
    """
    start_time = time.perf_counter()
    
    try:
        db_manager = get_db_manager()
        
        # Get counts before clearing
        stats = db_manager.get_stats()
        nodes_count = stats["total_nodes"]
        edges_count = stats["total_edges"]
        
        # Clear vector index
        db_manager.vector_index.clear()
        
        # Clear graph index
        db_manager.graph_index.clear()
        
        # Clear SQLite (need to delete all records)
        with db_manager.sqlite_store._cursor() as cursor:
            cursor.execute("DELETE FROM edges")
            cursor.execute("DELETE FROM nodes")
        
        # Invalidate cache
        invalidate_cache()
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        return {
            "success": True,
            "deleted_nodes": nodes_count,
            "deleted_edges": edges_count,
            "elapsed_ms": round(elapsed, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")

