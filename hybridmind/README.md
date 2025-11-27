# 🧠 HybridMind

**Vector + Graph Native Database for AI Retrieval**

HybridMind is a high-performance hybrid database that combines vector embeddings with graph-based relationships to deliver superior AI retrieval. Unlike pure vector databases that miss relational context, or graph databases that lack semantic understanding, HybridMind merges both paradigms into a unified query engine.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Key Features

- **Vector Search**: Semantic similarity using FAISS with cosine distance
- **Graph Search**: Relationship traversal using NetworkX with BFS/DFS
- **Hybrid Search**: Novel **Contextual Relevance Score (CRS)** algorithm combining both approaches
- **Auto-generated Embeddings**: Uses sentence-transformers (all-MiniLM-L6-v2)
- **SQLite Persistence**: ACID-compliant storage with automatic persistence
- **RESTful API**: Complete CRUD operations with OpenAPI documentation
- **Sub-100ms Queries**: Optimized for real-time AI applications

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/hybridmind.git
cd hybridmind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start the Server

```bash
# Start FastAPI server
uvicorn hybridmind.main:app --reload --port 8000

# Or use the CLI
python -m cli.main serve
```

### Load Demo Data

```bash
python data/load_demo_data.py
```

### Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000/

### Start the UI (Optional)

```bash
streamlit run ui/app.py
```

## 📖 API Overview

### Node Operations

```bash
# Create a node
curl -X POST http://localhost:8000/nodes \
  -H "Content-Type: application/json" \
  -d '{"text": "Transformers use attention mechanisms", "metadata": {"title": "ML Paper"}}'

# Get a node
curl http://localhost:8000/nodes/{node_id}

# Delete a node
curl -X DELETE http://localhost:8000/nodes/{node_id}
```

### Edge Operations

```bash
# Create an edge
curl -X POST http://localhost:8000/edges \
  -H "Content-Type: application/json" \
  -d '{"source_id": "node-1", "target_id": "node-2", "type": "cites", "weight": 0.9}'
```

### Search Operations

```bash
# Vector search (semantic similarity)
curl -X POST http://localhost:8000/search/vector \
  -H "Content-Type: application/json" \
  -d '{"query_text": "neural networks", "top_k": 10}'

# Graph search (traversal)
curl "http://localhost:8000/search/graph?start_id=node-1&depth=2"

# Hybrid search (CRS algorithm)
curl -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "deep learning optimization",
    "vector_weight": 0.6,
    "graph_weight": 0.4,
    "top_k": 10
  }'
```

## 🧮 CRS Algorithm

The **Contextual Relevance Score (CRS)** is our key differentiator:

```
CRS = α × vector_score + β × graph_score + γ × relationship_bonus
```

Where:
- `α` (vector_weight): Weight for semantic similarity (default: 0.6)
- `β` (graph_weight): Weight for graph proximity (default: 0.4)
- `vector_score`: Cosine similarity (0-1)
- `graph_score`: Inverse shortest path distance (0-1)
- `γ`: Optional bonus for specific edge types

### Why Hybrid Works Better

| Approach | Strength | Weakness |
|----------|----------|----------|
| Vector-Only | Semantic similarity | No relationship reasoning |
| Graph-Only | Deep traversals | No semantic search |
| **Hybrid (CRS)** | **Both capabilities** | **Principled ranking** |

## 📁 Project Structure

```
hybridmind/
├── hybridmind/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration
│   ├── models/              # Pydantic models
│   ├── storage/             # SQLite, FAISS, NetworkX
│   ├── engine/              # Search & ranking logic
│   ├── api/                 # API endpoints
│   └── utils/               # Utilities
├── cli/                     # Command line interface
├── ui/                      # Streamlit UI
├── data/                    # Demo datasets
├── tests/                   # Test suite
└── docs/                    # Documentation
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hybridmind --cov-report=html
```

## 📊 Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Node Create | < 50ms | Including embedding generation |
| Node Read | < 5ms | From SQLite |
| Vector Search | < 50ms | FAISS is fast |
| Graph Traversal | < 30ms | NetworkX BFS |
| Hybrid Search | < 100ms | Combined pipeline |

## 🎮 Demo Scenarios

### Scenario 1: Vector-Only vs Hybrid

**Query**: "transformer attention mechanisms"

- Vector-only finds papers by semantic similarity
- Hybrid also boosts papers connected through citations
- **Result**: Discovers related work that vector-only missed

### Scenario 2: Multi-Hop Reasoning

**Query**: "Find papers cited by papers about BERT"

- Start from BERT paper
- Traverse citations (depth=2)
- Combine with semantic relevance
- **Result**: Surfaces foundational work and related research

### Scenario 3: Anchor-Based Search

**Query**: "optimization techniques" with anchor on Adam paper

- Vector search finds optimization papers
- Graph scores boost papers connected to Adam
- **Result**: Prioritizes papers from the same research lineage

## 🛠️ Technology Stack

- **Backend**: Python 3.9+, FastAPI
- **Vector Search**: FAISS (CPU)
- **Graph**: NetworkX
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Storage**: SQLite
- **CLI**: Typer + Rich
- **UI**: Streamlit

## 📚 Use Cases

1. **Research Paper Discovery**: Find papers by topic AND citation relationships
2. **Knowledge Graphs**: Query entities with semantic + relational context
3. **RAG Pipelines**: Better retrieval for LLM applications
4. **Enterprise Search**: Documents connected by organizational relationships

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search
- [NetworkX](https://networkx.org/) - Graph algorithms
- [sentence-transformers](https://www.sbert.net/) - Embedding models
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

---

**Built with ❤️ for DevForge Hackathon**

*HybridMind - Where semantics meet structure.*

