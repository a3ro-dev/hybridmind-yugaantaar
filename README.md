# HybridMind

**Vector + Graph Native Database** for AI Retrieval — Devfolio Problem Statement 1

> 🧠 Hybrid retrieval combining semantic vector search with graph relationships using the **CRS Algorithm**.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start API
uvicorn main:app --reload --port 8000

# 3. Load demo data (ArXiv papers with semantic edges)
python data/load_demo_data.py --papers 150 --clear

# 4. Run full endpoint test
python data/test_endpoints.py

# 5. Launch UI (optional)
streamlit run ui/app.py
```

**Access:**
- 📖 API Docs: http://localhost:8000/docs
- 🖥️ UI Dashboard: http://localhost:8501

---

## 📋 Problem Statement Compliance

| Requirement | Implementation |
|-------------|----------------|
| **Vector storage with cosine similarity** | FAISS IndexFlatIP with sentence-transformers |
| **Graph storage with nodes/edges/metadata** | NetworkX + SQLite persistence |
| **Hybrid retrieval** | CRS Algorithm: `α·Vector + β·Graph` |
| **API endpoints (CRUD + Search)** | FastAPI with full REST endpoints |
| **Scoring/ranking mechanism** | Contextual Relevance Score (CRS) |
| **Embeddings pipeline** | all-MiniLM-L6-v2 (384-dim) |
| **Local persistence** | `.mind` file format (SQLite + FAISS + NetworkX) |
| **Real use-case dataset** | ArXiv ML papers with semantic edges |

---

## 🔌 API Endpoints

### Node CRUD

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/nodes` | Create node with text, metadata, optional embedding |
| `GET` | `/nodes/{id}` | Get node with properties and relationships |
| `PUT` | `/nodes/{id}` | Update node, optionally regenerate embedding |
| `DELETE` | `/nodes/{id}` | Delete node and all associated edges |
| `GET` | `/nodes` | List nodes with pagination |

### Edge CRUD

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/edges` | Create relationship: `{source, target, type, weight}` |
| `GET` | `/edges/{id}` | Get edge details |
| `PUT` | `/edges/{id}` | Update edge type, weight, metadata |
| `DELETE` | `/edges/{id}` | Delete edge |
| `GET` | `/edges` | List edges with filtering |

### Vector Search

```bash
POST /search/vector
Body: {"query_text": "...", "top_k": 10}
```
Returns ranked matches by **cosine similarity**.

### Graph Traversal

```bash
GET /search/graph?start_id=...&depth=2&direction=both
```
Returns reachable nodes up to specified depth.

### Hybrid Search (CRS Algorithm)

```bash
POST /search/hybrid
Body: {
  "query_text": "neural network optimization",
  "top_k": 10,
  "vector_weight": 0.6,  # α
  "graph_weight": 0.4,   # β
  "anchor_nodes": ["optional-node-id"]
}
```
Returns **merged scores + ranked output** using:
```
CRS = α × vector_score + β × graph_score
```

### Additional Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/search/compare` | Compare vector-only vs graph-only vs hybrid |
| `GET` | `/search/path/{source}/{target}` | Find shortest path (multi-hop) |
| `GET` | `/search/stats` | Database statistics |
| `GET` | `/health` | Health check with uptime |
| `POST` | `/comparison/effectiveness` | Quantitative metrics (Precision, NDCG, MRR) |
| `POST` | `/comparison/ablation` | Weight optimization study |

---

## 🧮 CRS Algorithm

The **Contextual Relevance Score** combines semantic similarity with graph relationships:

```
CRS(query, node) = α × V(query, node) + β × G(anchor, node)

Where:
  V = cosine_similarity(query_embedding, node_embedding)
  G = graph_proximity(anchor_nodes, node)
  α = 0.6 (semantic weight)
  β = 0.4 (graph weight)
```

### Why α=0.6, β=0.4?

1. **Semantic primacy**: Vector similarity captures core query intent
2. **Graph as enhancer**: Relationships provide contextual re-ranking
3. **Empirically validated**: Run `/comparison/ablation` to verify

---

## 📊 Proving Hybrid Effectiveness

### Quantitative Metrics

```bash
# Single query effectiveness
curl -X POST http://localhost:8000/comparison/effectiveness \
  -H "Content-Type: application/json" \
  -d '{"query_text": "neural network optimization", "top_k": 10}'

# Ablation study (tests different weights)
curl -X POST "http://localhost:8000/comparison/ablation?query=deep+learning&top_k=10"

# Multi-query summary
curl http://localhost:8000/comparison/effectiveness/summary
```

### Metrics Computed
- **Precision@K**: Fraction of retrieved results that are relevant
- **Recall@K**: Fraction of relevant results retrieved
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain
- **Coverage**: Percentage of relevant set found

---

## 📁 The `.mind` File Format

HybridMind uses **`.mind`** as its native database format:

```
hybridmind.mind/
├── manifest.json      # Version, stats, metadata
├── store.db           # SQLite (nodes, edges, embeddings)
├── vectors.faiss      # FAISS vector index
└── graph.nx           # NetworkX graph (pickle)
```

### CLI Commands

```bash
python -m cli.mind info data/hybridmind.mind    # Show database info
python -m cli.mind create knowledge.mind         # Create new database
python -m cli.mind export data/hybridmind.mind backup.mind.zip
```

---

## 🏗️ Project Structure

```
yugaantar/
├── main.py              # FastAPI application
├── config.py            # Configuration settings
├── api/                 # REST API endpoints
│   ├── nodes.py         # Node CRUD
│   ├── edges.py         # Edge CRUD
│   ├── search.py        # Vector, Graph, Hybrid search
│   ├── comparison.py    # Effectiveness metrics
│   └── bulk.py          # Bulk operations
├── engine/              # Core algorithms
│   ├── embedding.py     # Sentence-transformers
│   ├── vector_search.py # FAISS-based search
│   ├── graph_search.py  # NetworkX traversal
│   ├── hybrid_ranker.py # CRS algorithm
│   ├── effectiveness.py # Metrics calculation
│   └── cache.py         # Query caching
├── storage/             # Persistence layer
│   ├── sqlite_store.py  # SQLite backend
│   ├── vector_index.py  # FAISS index
│   ├── graph_index.py   # NetworkX graph
│   └── mindfile.py      # .mind format handler
├── ui/app.py            # Streamlit dashboard
├── data/                # Database & demo data
└── tests/               # Test suite
```

---

## 🐳 Docker

```bash
docker-compose up --build
```

---

## ✅ Testing

```bash
# Run all tests
pytest tests/ -v

# Run endpoint verification
python data/test_endpoints.py
```

---

## 📈 Evaluation Criteria Mapping

### Round 1: Technical Qualifier (50 pts)
- ✅ **Core functionality (20 pts)**: Full CRUD, vector search, graph traversal
- ✅ **Hybrid retrieval logic (10 pts)**: CRS algorithm with configurable weights
- ✅ **API quality (10 pts)**: FastAPI with OpenAPI docs, clear structure
- ✅ **Performance (10 pts)**: Query caching, GPU support, <100ms latency

### Round 2: Final Demo (100 pts)
- ✅ **Real-world demo (30 pts)**: ArXiv paper search with semantic edges
- ✅ **Hybrid effectiveness (25 pts)**: Quantitative proof via `/comparison/effectiveness`
- ✅ **System design (20 pts)**: CRS algorithm, `.mind` format, ablation study
- ✅ **Code quality (15 pts)**: Modular architecture, typed, tested
- ✅ **Presentation (10 pts)**: Interactive UI, comprehensive docs

---

**DevForge Hackathon** | Team CodeHashira
