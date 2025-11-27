# HybridMind

**Vector + Graph Native Database** - A hybrid retrieval system combining semantic vector search with graph-based relationship traversal.

## Quick Start

```bash
# 1. Start the API
cd hybridmind
uvicorn hybridmind.main:app --reload

# 2. Load demo data (200 papers)
python data/load_demo_data.py --papers 200

# 3. Launch UI
streamlit run ui/app.py
```

## Project Structure

```
hybridmind/
├── hybridmind/          # Core library
│   ├── api/             # FastAPI endpoints
│   ├── engine/          # Search engines, embedding, caching
│   ├── middleware/      # Rate limiting
│   ├── models/          # Pydantic schemas
│   ├── storage/         # SQLite, FAISS, NetworkX
│   └── main.py          # FastAPI app entry point
│
├── ui/                  # Streamlit dashboard
│   └── app.py
│
├── data/                # Database files & loaders
│   ├── hybridmind.db    # SQLite database
│   ├── load_demo_data.py        # Load into HybridMind only
│   └── load_all_databases.py    # Load into all 3 DBs (for benchmarks)
│
├── tests/               # Pytest test suite
├── benchmarks/          # Comparison with Neo4j & ChromaDB
├── monitoring/          # Prometheus & Grafana configs
│
├── Dockerfile           # Development Docker image
├── Dockerfile.prod      # Production Docker image (multi-stage)
├── docker-compose.yml   # Dev: API + UI
└── docker-compose.prod.yml  # Prod: API + UI + Prometheus + Grafana
```

## Usage Options

### Option 1: Local Development (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn hybridmind.main:app --reload --port 8000

# In another terminal, start UI
streamlit run ui/app.py
```

### Option 2: Docker (Development)

```bash
docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

### Option 3: Docker (Production)

```bash
docker-compose -f docker-compose.prod.yml up --build -d
# API: http://localhost:8000
# UI:  http://localhost:8501
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/hybridmind)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with component status |
| GET | `/ready` | Kubernetes readiness probe |
| GET | `/live` | Kubernetes liveness probe |
| POST | `/nodes` | Create node |
| GET | `/nodes/{id}` | Get node by ID |
| POST | `/edges` | Create edge |
| POST | `/search/vector` | Vector-only search |
| GET | `/search/graph` | Graph traversal |
| POST | `/search/hybrid` | **Hybrid CRS search** |
| GET | `/cache/stats` | Cache statistics |
| POST | `/bulk/nodes` | Bulk node import |
| POST | `/bulk/edges` | Bulk edge import |

## CRS Algorithm

The **Contextual Relevance Score** combines vector similarity with graph proximity:

```
CRS(q) = α·V(q) + β·G(q)

Where:
  V(q) = cosine similarity from FAISS
  G(q) = graph proximity score
  α = vector weight (default: 0.6)
  β = graph weight (default: 0.4)
```

## Data Loading

**For demo/testing:**
```bash
python data/load_demo_data.py --papers 200
```

**For benchmarks (loads Neo4j & ChromaDB too):**
```bash
python data/load_all_databases.py --papers 200
```

## Running Tests

```bash
pytest tests/ -v
```

## File Explanations

| File | Purpose |
|------|---------|
| `Dockerfile` | Development image (uvicorn) |
| `Dockerfile.prod` | Production image (gunicorn, multi-stage) |
| `docker-compose.yml` | Dev stack: API + UI |
| `docker-compose.prod.yml` | Prod stack: API + UI + monitoring |
| `test_data/` | Auto-created during tests, can be deleted |
| `data/chromadb/` | ChromaDB storage (for benchmarks only) |

---

**DevForge Hackathon** | Team CodeHashira
