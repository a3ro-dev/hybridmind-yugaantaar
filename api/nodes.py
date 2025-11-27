"""
Node CRUD API endpoints for HybridMind.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from models.node import (
    NodeCreate,
    NodeUpdate,
    NodeResponse,
    NodeDeleteResponse,
    EdgeSummary
)
from api.dependencies import (
    get_sqlite_store,
    get_vector_index,
    get_graph_index,
    get_embedding_engine
)
from storage.sqlite_store import SQLiteStore
from storage.vector_index import VectorIndex
from storage.graph_index import GraphIndex
from engine.embedding import EmbeddingEngine
from engine.cache import invalidate_cache

router = APIRouter(prefix="/nodes", tags=["Nodes"])


@router.post("", response_model=NodeResponse, status_code=201)
async def create_node(
    node: NodeCreate,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine)
) -> NodeResponse:
    """
    Create a new node with text and optional embedding.
    
    If no embedding is provided, one will be generated automatically
    using the configured embedding model (all-MiniLM-L6-v2 by default).
    """
    # Generate node ID
    node_id = str(uuid.uuid4())
    
    # Generate or use provided embedding
    if node.embedding:
        import numpy as np
        embedding = np.array(node.embedding, dtype=np.float32)
    else:
        embedding = embedding_engine.embed(node.text)
    
    # Store in SQLite
    result = sqlite_store.create_node(
        node_id=node_id,
        text=node.text,
        metadata=node.metadata or {},
        embedding=embedding
    )
    
    # Add to vector index
    vector_index.add(node_id, embedding)
    
    # Add to graph index (as isolated node initially)
    graph_index.add_node(node_id)
    
    # Invalidate search cache
    invalidate_cache()
    
    return NodeResponse(
        id=result["id"],
        text=result["text"],
        metadata=result["metadata"],
        created_at=result["created_at"],
        updated_at=result["updated_at"],
        edges=[]
    )


@router.get("/{node_id}", response_model=NodeResponse)
async def get_node(
    node_id: str,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> NodeResponse:
    """
    Retrieve a node by ID with its relationships.
    """
    node = sqlite_store.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Get connected edges
    edges_data = sqlite_store.get_node_edges(node_id)
    edges = []
    for edge in edges_data:
        # Determine target node
        if edge["source_id"] == node_id:
            target_id = edge["target_id"]
            direction = "outgoing"
        else:
            target_id = edge["source_id"]
            direction = "incoming"
        
        edges.append(EdgeSummary(
            edge_id=edge["id"],
            target_id=target_id,
            type=edge["type"],
            weight=edge["weight"],
            direction=direction
        ))
    
    return NodeResponse(
        id=node["id"],
        text=node["text"],
        metadata=node["metadata"],
        created_at=node["created_at"],
        updated_at=node["updated_at"],
        edges=edges
    )


@router.put("/{node_id}", response_model=NodeResponse)
async def update_node(
    node_id: str,
    update: NodeUpdate,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    embedding_engine: EmbeddingEngine = Depends(get_embedding_engine)
) -> NodeResponse:
    """
    Update node content and optionally regenerate embedding.
    """
    # Check if node exists
    existing = sqlite_store.get_node(node_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Prepare update values
    new_text = update.text if update.text is not None else existing["text"]
    new_metadata = update.metadata if update.metadata is not None else existing["metadata"]
    
    # Regenerate embedding if requested and text changed
    new_embedding = existing["embedding"]
    if update.regenerate_embedding and update.text is not None:
        new_embedding = embedding_engine.embed(new_text)
    
    # Update in SQLite
    result = sqlite_store.update_node(
        node_id=node_id,
        text=new_text,
        metadata=new_metadata,
        embedding=new_embedding
    )
    
    # Update vector index if embedding changed
    if new_embedding is not None:
        vector_index.add(node_id, new_embedding)  # add() handles replacement
    
    # Invalidate search cache
    invalidate_cache()
    
    # Get edges for response
    edges_data = sqlite_store.get_node_edges(node_id)
    edges = []
    for edge in edges_data:
        if edge["source_id"] == node_id:
            target_id = edge["target_id"]
            direction = "outgoing"
        else:
            target_id = edge["source_id"]
            direction = "incoming"
        
        edges.append(EdgeSummary(
            edge_id=edge["id"],
            target_id=target_id,
            type=edge["type"],
            weight=edge["weight"],
            direction=direction
        ))
    
    return NodeResponse(
        id=result["id"],
        text=result["text"],
        metadata=result["metadata"],
        created_at=result["created_at"],
        updated_at=result["updated_at"],
        edges=edges
    )


@router.delete("/{node_id}", response_model=NodeDeleteResponse)
async def delete_node(
    node_id: str,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    vector_index: VectorIndex = Depends(get_vector_index),
    graph_index: GraphIndex = Depends(get_graph_index)
) -> NodeDeleteResponse:
    """
    Delete a node and all its associated edges.
    """
    # Check if node exists
    existing = sqlite_store.get_node(node_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    # Delete from SQLite (cascades to edges)
    deleted, edges_removed = sqlite_store.delete_node(node_id)
    
    # Remove from vector index
    vector_index.remove(node_id)
    
    # Remove from graph index
    graph_index.remove_node(node_id)
    
    # Invalidate search cache
    invalidate_cache()
    
    return NodeDeleteResponse(
        deleted=deleted,
        node_id=node_id,
        edges_removed=edges_removed
    )


@router.get("", response_model=List[NodeResponse])
async def list_nodes(
    skip: int = Query(default=0, ge=0, description="Number of nodes to skip"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum nodes to return"),
    sqlite_store: SQLiteStore = Depends(get_sqlite_store)
) -> List[NodeResponse]:
    """
    List all nodes with pagination.
    """
    nodes = sqlite_store.list_nodes(skip=skip, limit=limit)
    
    results = []
    for node in nodes:
        # Get edges for each node
        edges_data = sqlite_store.get_node_edges(node["id"])
        edges = []
        for edge in edges_data:
            if edge["source_id"] == node["id"]:
                target_id = edge["target_id"]
                direction = "outgoing"
            else:
                target_id = edge["source_id"]
                direction = "incoming"
            
            edges.append(EdgeSummary(
                edge_id=edge["id"],
                target_id=target_id,
                type=edge["type"],
                weight=edge["weight"],
                direction=direction
            ))
        
        results.append(NodeResponse(
            id=node["id"],
            text=node["text"],
            metadata=node["metadata"],
            created_at=node["created_at"],
            updated_at=node["updated_at"],
            edges=edges
        ))
    
    return results

