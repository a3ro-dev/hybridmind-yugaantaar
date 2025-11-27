# HybridMind

**Vector + Graph Native Database** - Hybrid retrieval combining semantic search with graph relationships.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start API
uvicorn main:app --reload --port 8000

# 3. Load demo data
python data/load_demo_data.py --papers 200

# 4. Launch UI
streamlit run ui/app.py
```

**Access:**
- API: http://localhost:8000/docs
- UI: http://localhost:8501

## Project Structure

```
yugaantar/
├── main.py              # FastAPI entry point
├── config.py            # Settings
├── api/                 # REST endpoints
│   ├── nodes.py         # Node CRUD
│   ├── edges.py         # Edge CRUD
│   ├── search.py        # Search endpoints
│   └── bulk.py          # Bulk operations
├── engine/              # Core algorithms
│   ├── embedding.py     # Text embeddings
│   ├── vector_search.py # FAISS vector search
│   ├── graph_search.py  # NetworkX graph traversal
│   ├── hybrid_ranker.py # CRS algorithm
│   └── cache.py         # Query caching
├── storage/             # Data layer
│   ├── sqlite_store.py  # Persistent storage
│   ├── vector_index.py  # FAISS index
│   └── graph_index.py   # NetworkX graph
├── middleware/          # Rate limiting
├── models/              # Pydantic schemas
├── ui/app.py            # Streamlit dashboard
├── data/                # Database files
│   ├── hybridmind.db    # SQLite
│   ├── vector.index     # FAISS
│   └── graph.pkl        # NetworkX
└── tests/               # Test suite
```

## Docker

```bash
docker-compose up --build
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/nodes` | POST | Create node |
| `/nodes/{id}` | GET | Get node |
| `/edges` | POST | Create edge |
| `/search/vector` | POST | Vector search |
| `/search/graph` | GET | Graph traversal |
| `/search/hybrid` | POST | **Hybrid CRS search** |
| `/bulk/nodes` | POST | Bulk import nodes |
| `/health` | GET | Health check |

## CRS Algorithm

```
CRS(q) = α·V(q) + β·G(q)

V = cosine similarity (FAISS)
G = graph proximity (NetworkX)
α = 0.6 (default)
β = 0.4 (default)
```

## Tests

```bash
pytest tests/ -v
```

---
**DevForge Hackathon** | Team CodeHashira
