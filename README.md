# HybridMind

**Vector + Graph Native Database** - Hybrid retrieval combining semantic search with graph relationships.

> 🧠 Uses the **`.mind`** file format — a self-contained database bundling vectors, graphs, and metadata.

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

## The `.mind` File Format

HybridMind uses **`.mind`** as its native database extension — a directory-based format that bundles everything:

```
hybridmind.mind/
├── manifest.json      # Version, stats, metadata
├── store.db           # SQLite database (nodes, edges)
├── vectors.faiss      # FAISS vector index
├── vectors.map        # ID mappings
└── graph.nx           # NetworkX graph (pickle)
```

### Why `.mind`?

| Feature | Benefit |
|---------|---------|
| **Self-contained** | One "file" contains everything |
| **Portable** | Export as `.mind.zip`, share anywhere |
| **Versioned** | Manifest tracks format version |
| **Inspectable** | `manifest.json` shows stats |

### CLI Commands

```bash
# Show database info
python -m cli.mind info data/hybridmind.mind

# Create new database
python -m cli.mind create knowledge.mind

# Export for sharing
python -m cli.mind export data/hybridmind.mind backup.mind.zip

# List all .mind files
python -m cli.mind list data/
```

## Project Structure

```
yugaantar/
├── main.py              # FastAPI entry point
├── config.py            # Settings
├── api/                 # REST endpoints
│   ├── nodes.py         # Node CRUD
│   ├── edges.py         # Edge CRUD
│   ├── search.py        # Search endpoints
│   ├── comparison.py    # DB comparison endpoints
│   └── bulk.py          # Bulk operations
├── engine/              # Core algorithms
│   ├── embedding.py     # Text embeddings
│   ├── vector_search.py # FAISS vector search
│   ├── graph_search.py  # NetworkX graph traversal
│   ├── hybrid_ranker.py # CRS algorithm
│   ├── comparison.py    # Neo4j/ChromaDB comparison
│   └── cache.py         # Query caching
├── storage/             # Data layer
│   ├── sqlite_store.py  # Persistent storage
│   ├── vector_index.py  # FAISS index
│   ├── graph_index.py   # NetworkX graph
│   └── mindfile.py      # .mind format handler
├── cli/                 # Command-line tools
│   ├── main.py          # Main CLI
│   └── mind.py          # .mind file manager
├── middleware/          # Rate limiting
├── models/              # Pydantic schemas
├── ui/app.py            # Streamlit dashboard
├── data/                # Database files
│   └── hybridmind.mind/ # .mind database
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
