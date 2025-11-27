# HybridMind API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required for local deployment.

---

## Node Operations

### Create Node

**POST** `/nodes`

Create a new node with text content and optional embedding.

**Request Body:**
```json
{
  "text": "string (required)",
  "metadata": {
    "title": "string",
    "tags": ["string"],
    "source": "string"
  },
  "embedding": [0.1, 0.2, ...]  // optional, 384 dimensions
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "text": "string",
  "metadata": {},
  "created_at": "timestamp",
  "updated_at": "timestamp",
  "edges": []
}
```

### Get Node

**GET** `/nodes/{node_id}`

Retrieve a node by ID with its relationships.

**Response:** `200 OK`
```json
{
  "id": "uuid",
  "text": "string",
  "metadata": {},
  "created_at": "timestamp",
  "updated_at": "timestamp",
  "edges": [
    {
      "edge_id": "uuid",
      "target_id": "uuid",
      "type": "string",
      "weight": 0.9,
      "direction": "outgoing"
    }
  ]
}
```

### Update Node

**PUT** `/nodes/{node_id}`

Update node text, metadata, or regenerate embedding.

**Request Body:**
```json
{
  "text": "string",
  "metadata": {},
  "regenerate_embedding": true
}
```

### Delete Node

**DELETE** `/nodes/{node_id}`

Delete a node and all its edges.

**Response:** `200 OK`
```json
{
  "deleted": true,
  "node_id": "uuid",
  "edges_removed": 5
}
```

### List Nodes

**GET** `/nodes?skip=0&limit=100`

List all nodes with pagination.

---

## Edge Operations

### Create Edge

**POST** `/edges`

Create a relationship between two nodes.

**Request Body:**
```json
{
  "source_id": "uuid",
  "target_id": "uuid",
  "type": "cites",
  "weight": 0.9,
  "metadata": {}
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid",
  "source_id": "uuid",
  "target_id": "uuid",
  "type": "cites",
  "weight": 0.9,
  "metadata": {},
  "created_at": "timestamp"
}
```

### Get Edge

**GET** `/edges/{edge_id}`

Retrieve an edge by ID.

### Delete Edge

**DELETE** `/edges/{edge_id}`

Delete an edge.

### Get Node Edges

**GET** `/edges/node/{node_id}?direction=both`

Get all edges connected to a node.

**Query Parameters:**
- `direction`: `outgoing`, `incoming`, or `both`

---

## Search Operations

### Vector Search

**POST** `/search/vector`

Pure semantic similarity search using cosine distance.

**Request Body:**
```json
{
  "query_text": "string",
  "top_k": 10,
  "min_score": 0.0,
  "filter_metadata": {
    "year": {"$gte": 2020}
  }
}
```

**Response:** `200 OK`
```json
{
  "results": [
    {
      "node_id": "uuid",
      "text": "string",
      "metadata": {},
      "vector_score": 0.92,
      "reasoning": "Semantic similarity: 92%"
    }
  ],
  "query_time_ms": 45.2,
  "total_candidates": 100,
  "search_type": "vector"
}
```

### Graph Search

**GET** `/search/graph?start_id={id}&depth=2&direction=both`

Graph traversal from a starting node.

**Query Parameters:**
- `start_id`: Starting node UUID (required)
- `depth`: Maximum traversal depth (1-5, default: 2)
- `edge_types`: Filter by edge types (optional)
- `direction`: `outgoing`, `incoming`, or `both`

**Response:** `200 OK`
```json
{
  "results": [
    {
      "node_id": "uuid",
      "text": "string",
      "metadata": {},
      "graph_score": 0.5,
      "depth": 2,
      "path": ["uuid1", "uuid2", "uuid3"],
      "reasoning": "Reachable in 2 hop(s)"
    }
  ],
  "query_time_ms": 30.1,
  "total_candidates": 25,
  "search_type": "graph"
}
```

### Hybrid Search

**POST** `/search/hybrid`

Combined vector + graph search using CRS algorithm.

**Request Body:**
```json
{
  "query_text": "string",
  "top_k": 10,
  "vector_weight": 0.6,
  "graph_weight": 0.4,
  "anchor_nodes": ["uuid1", "uuid2"],
  "max_depth": 2,
  "edge_type_weights": {
    "cites": 1.0,
    "related_to": 0.5
  },
  "min_score": 0.0
}
```

**Response:** `200 OK`
```json
{
  "results": [
    {
      "node_id": "uuid",
      "text": "string",
      "metadata": {},
      "vector_score": 0.85,
      "graph_score": 0.72,
      "combined_score": 0.80,
      "reasoning": "High semantic similarity (85%), strongly connected"
    }
  ],
  "query_time_ms": 95.5,
  "total_candidates": 150,
  "search_type": "hybrid"
}
```

### Compare Search Modes

**POST** `/search/compare`

Compare results across all three search modes.

**Request Body:** Same as hybrid search

**Response:** `200 OK`
```json
{
  "query_text": "string",
  "vector_only": {
    "results": [...],
    "query_time_ms": 40.2
  },
  "graph_only": {
    "results": [...],
    "query_time_ms": 25.1
  },
  "hybrid": {
    "results": [...],
    "query_time_ms": 90.5
  },
  "analysis": {
    "vector_unique": 3,
    "graph_unique": 2,
    "hybrid_unique": 1,
    "overlap_all": 4
  }
}
```

### Find Path

**GET** `/search/path/{source_id}/{target_id}`

Find shortest path between two nodes.

**Response:** `200 OK`
```json
{
  "source_id": "uuid",
  "target_id": "uuid",
  "path_exists": true,
  "path": ["uuid1", "uuid2", "uuid3"],
  "length": 2,
  "total_weight": 1.8,
  "nodes": [...],
  "edges": [...]
}
```

---

## Utility Operations

### Get Statistics

**GET** `/search/stats`

Get database statistics.

**Response:** `200 OK`
```json
{
  "total_nodes": 500,
  "total_edges": 1200,
  "edge_types": {"cites": 800, "related_to": 400},
  "avg_edges_per_node": 2.4,
  "vector_index_size": 500,
  "database_size_bytes": 5242880
}
```

### Health Check

**GET** `/health`

Check API health status.

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "database": "connected",
  "nodes": 500,
  "edges": 1200,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

### Create Snapshot

**POST** `/snapshot`

Persist indexes to disk.

**Response:** `200 OK`
```json
{
  "status": "success",
  "message": "Indexes saved to disk"
}
```

---

## Error Responses

### 404 Not Found
```json
{
  "detail": "Node {id} not found"
}
```

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error",
  "error": "Error message"
}
```

---

## Metadata Filter Operators

Available operators for `filter_metadata`:

| Operator | Description | Example |
|----------|-------------|---------|
| `$gt` | Greater than | `{"year": {"$gt": 2019}}` |
| `$gte` | Greater than or equal | `{"year": {"$gte": 2020}}` |
| `$lt` | Less than | `{"score": {"$lt": 0.5}}` |
| `$lte` | Less than or equal | `{"score": {"$lte": 0.5}}` |
| `$ne` | Not equal | `{"status": {"$ne": "draft"}}` |
| `$in` | In list | `{"type": {"$in": ["a", "b"]}}` |
| `$nin` | Not in list | `{"type": {"$nin": ["x"]}}` |

---

## Rate Limits

No rate limits for local deployment. For production, consider implementing rate limiting via middleware.

