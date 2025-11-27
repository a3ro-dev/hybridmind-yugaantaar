# Database Comparison: HybridMind vs Neo4j vs ChromaDB

This document explains the comparison feature that benchmarks HybridMind against pure graph (Neo4j) and pure vector (ChromaDB) databases.

## Overview

| System | Type | Strengths | Weaknesses |
|--------|------|-----------|------------|
| **HybridMind** | Hybrid (Vector + Graph) | Best of both worlds, context-aware | Higher complexity |
| **Neo4j** | Graph-only | Excellent for traversals, relationships | No semantic similarity |
| **ChromaDB** | Vector-only | Fast semantic search | No relationship awareness |

## Quick Start

### 1. Load Data into All Systems

```bash
# Load the same ArXiv dataset into all three databases
python data/load_all_databases.py --papers 150

# Options:
#   --papers N        Number of papers to load (default: 150)
#   --skip-neo4j      Skip Neo4j if not available
#   --neo4j-uri       Neo4j connection URI (default: bolt://localhost:7687)
#   --neo4j-user      Neo4j username (default: neo4j)
#   --neo4j-password  Neo4j password (default: password)
```

### 2. Start the API

```bash
uvicorn main:app --reload --port 8000
```

### 3. Run Comparisons

**Via API:**
```bash
# Quick comparison
curl "http://localhost:8000/comparison/quick?query=machine%20learning&top_k=5"

# Full comparison with analysis
curl -X POST http://localhost:8000/comparison/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "deep learning neural networks", "top_k": 10}'

# Run benchmark
curl -X POST http://localhost:8000/comparison/benchmark \
  -H "Content-Type: application/json" \
  -d '{"queries": ["machine learning", "neural networks", "NLP"], "iterations": 3}'
```

**Via UI:**
```bash
streamlit run ui/app.py
# Navigate to "Comparison" tab
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/comparison/status` | GET | Check availability of all systems |
| `/comparison/search` | POST | Run same query on all 3 systems |
| `/comparison/benchmark` | POST | Run performance benchmark |
| `/comparison/quick` | GET | Quick comparison (simplified) |
| `/comparison/sample-queries` | GET | Get test queries |

## CRS Algorithm

HybridMind uses the **Contextual Relevance Score (CRS)** algorithm:

```
CRS = α × V(q) + β × G(q)

Where:
  α = vector_weight (default: 0.6)
  β = graph_weight (default: 0.4)
  V(q) = cosine similarity from vector search
  G(q) = graph proximity score (inverse distance)
```

## Example Comparison Results

### Query: "artificial intelligence machine learning"

#### Vector-Only (ChromaDB)
| # | Score | Text |
|---|-------|------|
| 1 | 0.490 | Machine Learning Application in the Life Time... |
| 2 | 0.401 | Human-in-the-loop Artificial Intelligence... |
| 3 | 0.354 | Efficient PAC Learning from the Crowd... |

#### Graph-Only (Neo4j)
| # | Depth | Score | Text |
|---|-------|-------|------|
| 1 | 1 | 0.500 | Sparse-to-Dense: Depth Prediction... |
| 2 | 1 | 0.500 | On-demand Relational Concept Analysis... |
| 3 | 2 | 0.333 | Dynamic Integration of Background Knowledge... |

#### Hybrid (HybridMind)
| # | Vector | Graph | CRS | Text |
|---|--------|-------|-----|------|
| 1 | 0.490 | 1.000 | **0.694** | Machine Learning Application... |
| 2 | 0.401 | 1.000 | **0.641** | Human-in-the-loop AI... |
| 3 | 0.354 | 1.000 | **0.612** | Efficient PAC Learning... |

### Key Observations

1. **HybridMind combines strengths**: Results that are both semantically similar AND well-connected rank highest
2. **ChromaDB finds semantic matches**: Good at finding conceptually similar content
3. **Neo4j finds related content**: Discovers connected papers through citations
4. **Unique discoveries**: Each system finds results the others miss

## Benchmark Metrics

The benchmark measures:

| Metric | Description |
|--------|-------------|
| **Avg Latency** | Average query time in milliseconds |
| **P50/P95/P99** | Latency percentiles |
| **Throughput** | Queries per second (QPS) |
| **Result Count** | Consistency of result retrieval |

### Typical Results

| System | Avg Latency | P95 | Throughput |
|--------|-------------|-----|------------|
| HybridMind | 30-50ms | 80ms | 20-33 QPS |
| Neo4j | 10-20ms | 40ms | 50-100 QPS |
| ChromaDB | 5-15ms | 30ms | 66-200 QPS |

**Note:** HybridMind is slower because it:
1. Generates embeddings
2. Runs vector search
3. Runs graph search
4. Combines results with CRS

The trade-off is **better retrieval quality** through contextual ranking.

## Setup External Databases

### Neo4j

```bash
# Using Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Or install locally: https://neo4j.com/download/
```

### ChromaDB

ChromaDB is embedded and created automatically when you run `load_all_databases.py`.

```bash
# Install if needed
pip install chromadb
```

## Configuration

Environment variables or `.env` file:

```env
# Neo4j
HYBRIDMIND_NEO4J_URI=bolt://localhost:7687
HYBRIDMIND_NEO4J_USER=neo4j
HYBRIDMIND_NEO4J_PASSWORD=password

# ChromaDB
HYBRIDMIND_CHROMADB_PATH=data/chromadb
HYBRIDMIND_CHROMADB_COLLECTION=arxiv_papers
```

## When to Use Each

| Use Case | Recommended System |
|----------|-------------------|
| Semantic search only | ChromaDB |
| Relationship traversal only | Neo4j |
| Context-aware retrieval | **HybridMind** |
| RAG with citations | **HybridMind** |
| Knowledge graph queries | Neo4j |
| Similar document finding | ChromaDB |
| Research paper discovery | **HybridMind** |

---

**DevForge Hackathon** | Team CodeHashira
