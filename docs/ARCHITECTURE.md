# HybridMind Architecture

## Overview

HybridMind is a hybrid database combining vector embeddings with graph relationships for superior AI retrieval. This document describes the system architecture, component design, and key algorithms.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Layer                          │
│              (REST endpoints + Auto-generated docs)            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Query Engine                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │Vector Search│  │Graph Search │  │  Hybrid Ranker (CRS)    │ │
│  │  (FAISS)    │  │ (NetworkX)  │  │  - Score fusion         │ │
│  │             │  │             │  │  - Re-ranking           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Unified Storage Layer                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Node Store (SQLite)                    │  │
│  │  - id, text, metadata (JSON), embedding (BLOB)           │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Edge Store (SQLite)                    │  │
│  │  - id, source_id, target_id, type, weight, metadata      │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Vector Index (FAISS - FlatIP)              │  │
│  │  - In-memory for speed, persisted to disk                │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Graph Index (NetworkX DiGraph)             │  │
│  │  - In-memory, serialized via pickle                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Embedding Pipeline                           │
│         sentence-transformers (all-MiniLM-L6-v2)               │
│                   384 dimensions, fast inference                │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Storage Layer

#### SQLite Store (`storage/sqlite_store.py`)
- **Purpose**: ACID-compliant persistent storage for nodes and edges
- **Tables**:
  - `nodes`: id, text, metadata (JSON), embedding (BLOB), timestamps
  - `edges`: id, source_id, target_id, type, weight, metadata (JSON)
- **Optimizations**:
  - WAL journal mode for concurrent reads
  - Indexed columns for fast lookups
  - Foreign key constraints with cascade delete

#### Vector Index (`storage/vector_index.py`)
- **Purpose**: Fast similarity search using FAISS
- **Index Type**: `IndexFlatIP` (Inner Product for cosine similarity)
- **Features**:
  - Normalized vectors for cosine similarity
  - In-memory operation with disk persistence
  - ID mapping (FAISS index → node UUID)
  - Fallback to NumPy if FAISS unavailable

#### Graph Index (`storage/graph_index.py`)
- **Purpose**: Relationship storage and traversal
- **Implementation**: NetworkX DiGraph
- **Features**:
  - Directed edges with type and weight
  - BFS/DFS traversal
  - Shortest path computation
  - Proximity scoring

### 2. Engine Layer

#### Embedding Engine (`engine/embedding.py`)
- **Model**: sentence-transformers `all-MiniLM-L6-v2`
- **Dimension**: 384
- **Features**:
  - Lazy model loading
  - Batch processing
  - Mock embeddings for testing
  - Normalized output for cosine similarity

#### Vector Search Engine (`engine/vector_search.py`)
- **Algorithm**: k-NN search with cosine similarity
- **Features**:
  - Query embedding generation
  - Metadata filtering (comparison operators)
  - Min score threshold
  - Result enrichment from SQLite

#### Graph Search Engine (`engine/graph_search.py`)
- **Algorithm**: BFS traversal with depth limit
- **Features**:
  - Directional traversal (outgoing/incoming/both)
  - Edge type filtering
  - Proximity score computation
  - Path finding

#### Hybrid Ranker (`engine/hybrid_ranker.py`)
- **Algorithm**: Contextual Relevance Score (CRS)
- **Features**:
  - Configurable weights (α, β)
  - Anchor node support
  - Edge type bonuses
  - Score explanation generation

### 3. API Layer

#### FastAPI Application (`main.py`)
- Auto-generated OpenAPI documentation
- CORS middleware
- Request timing headers
- Health checks
- Lifespan management

#### Endpoint Routers
- `api/nodes.py`: Node CRUD operations
- `api/edges.py`: Edge CRUD operations
- `api/search.py`: Search operations

## CRS Algorithm

### Formula

```
CRS = α × V + β × G + γ × R
```

Where:
- **V** = Vector similarity score (cosine similarity, 0-1)
- **G** = Graph proximity score (inverse shortest path, 0-1)
- **R** = Relationship bonus (optional, based on edge types)
- **α** = Vector weight (default: 0.6)
- **β** = Graph weight (default: 0.4)
- **γ** = Relationship bonus coefficient

### Vector Score (V)

```python
V = cosine_similarity(query_embedding, node_embedding)
  = dot(q, n) / (||q|| × ||n||)
```

For normalized vectors: `V = dot(q, n)`

### Graph Score (G)

```python
G = 1 / (1 + min_distance)
```

Where `min_distance` is the shortest path length from any reference node.

- Direct connection (distance=1): G = 0.5
- 2-hop connection (distance=2): G = 0.33
- No connection: G = 0

### Hybrid Search Algorithm

```
FUNCTION hybrid_search(query_text, top_k, α, β, anchor_nodes):
    
    # Step 1: Vector Search
    query_embedding = embed(query_text)
    vector_candidates = vector_index.search(query_embedding, top_k * 3)
    
    # Step 2: Determine Reference Nodes
    IF anchor_nodes IS NOT EMPTY:
        reference_nodes = anchor_nodes
    ELSE:
        reference_nodes = top 3 nodes from vector_candidates
    
    # Step 3: Compute Graph Scores
    FOR each candidate IN vector_candidates:
        graph_score = compute_graph_proximity(candidate, reference_nodes)
        candidate.graph_score = graph_score
    
    # Step 4: Compute Hybrid Scores
    FOR each candidate IN vector_candidates:
        candidate.hybrid_score = α * candidate.vector_score + β * candidate.graph_score
    
    # Step 5: Re-rank and Return
    sorted_results = SORT(vector_candidates, BY hybrid_score, DESC)
    RETURN sorted_results[:top_k]
```

## Data Flow

### Node Creation

```
1. API receives POST /nodes
2. Generate embedding via EmbeddingEngine
3. Store in SQLite (text, metadata, embedding)
4. Add to Vector Index (embedding → node_id mapping)
5. Add to Graph Index (isolated node)
6. Return response
```

### Hybrid Search

```
1. API receives POST /search/hybrid
2. Generate query embedding
3. Vector search → get candidates with vector scores
4. Determine reference nodes (anchors or top vector results)
5. Compute graph proximity scores for candidates
6. Apply CRS formula to compute combined scores
7. Sort by combined score, take top-k
8. Generate reasoning explanations
9. Return response
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Vector Search | O(n × d) | n=nodes, d=dimension; can be O(log n) with IVF |
| Graph Traversal | O(V + E) | V=vertices, E=edges in subgraph |
| Hybrid Search | O(k × V) | k=candidates, V=reference nodes |
| Node Create | O(d) | d=embedding dimension |
| Node Delete | O(degree) | Cascade edge removal |

### Space Complexity

| Component | Memory | Notes |
|-----------|--------|-------|
| Vector Index | O(n × d × 4) bytes | n nodes × d floats × 4 bytes |
| Graph Index | O(V + E) | NetworkX overhead |
| SQLite | Disk-based | With WAL mode |

### Latency Targets

| Operation | Target | Typical |
|-----------|--------|---------|
| Node Create | < 50ms | ~30ms (including embedding) |
| Node Read | < 5ms | ~2ms |
| Vector Search | < 50ms | ~20ms (1K nodes) |
| Graph Traversal | < 30ms | ~15ms (depth=2) |
| Hybrid Search | < 100ms | ~60ms |

## Scalability Considerations

### Current Limits (Demo Scale)

- Nodes: 10,000+
- Edges: 50,000+
- Query latency: < 100ms

### Scaling Strategies

1. **Vector Index**: Switch from FlatIP to IVF or HNSW
2. **Graph Index**: Partition large graphs or use Neo4j
3. **SQLite**: Migrate to PostgreSQL for concurrency
4. **Embedding**: Use GPU or batching for large imports

## Fault Tolerance

### Data Persistence

- SQLite with WAL mode (crash recovery)
- Periodic index snapshots
- Manual snapshot via `/snapshot` endpoint

### Index Rebuild

On startup:
1. Load SQLite data
2. Rebuild vector index from stored embeddings
3. Rebuild graph index from stored edges

### Graceful Shutdown

1. Save indexes to disk
2. Close SQLite connections
3. Log shutdown complete

## Security Considerations

### Current (Local Demo)

- No authentication
- CORS allows all origins
- Local file access only

### Production Recommendations

- Add API key authentication
- Restrict CORS origins
- Use HTTPS
- Rate limiting
- Input validation (already implemented)

