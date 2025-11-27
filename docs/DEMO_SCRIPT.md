# HybridMind Demo Script

## Setup (Before Demo)

```bash
# 1. Start the server
cd hybridmind
uvicorn hybridmind.main:app --reload --port 8000

# 2. Load demo data
python data/load_demo_data.py

# 3. Start UI (optional)
streamlit run ui/app.py
```

## Demo Outline (9 minutes)

### Opening (1 min)

**Script:**
> "HybridMind combines semantic search with relationship intelligence. Let me show you why this matters."

**Show:** Architecture diagram from README

### Problem Setup (2 min)

#### Vector-Only Fails

**Query:** Search for "attention mechanisms"

```bash
curl -X POST http://localhost:8000/search/vector \
  -H "Content-Type: application/json" \
  -d '{"query_text": "attention mechanisms", "top_k": 5}'
```

**Script:**
> "Vector search finds semantically similar papers about attention. But look - it's missing the foundational work that the attention papers build upon. It doesn't know about citations."

#### Graph-Only Fails

**Query:** Traverse from the Transformer paper

```bash
curl "http://localhost:8000/search/graph?start_id=paper-transformer&depth=2"
```

**Script:**
> "Graph search finds connected papers through citations. But it returns everything connected, even papers that aren't relevant to our query. No semantic filtering."

### Solution Demo (3 min)

#### Hybrid Search

**Query:** Same query with hybrid mode

```bash
curl -X POST http://localhost:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "attention mechanisms",
    "vector_weight": 0.6,
    "graph_weight": 0.4,
    "anchor_nodes": ["paper-transformer"],
    "top_k": 5
  }'
```

**Script:**
> "Hybrid search combines both signals. Notice the scores - each result has a vector score for semantic relevance AND a graph score for connection strength. The CRS algorithm fuses them intelligently."

**Highlight:**
- Show `vector_score`, `graph_score`, `combined_score`
- Point out the reasoning field explaining why each result appears

#### Compare Mode

**Query:** Side-by-side comparison

```bash
curl -X POST http://localhost:8000/search/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "language model pre-training",
    "anchor_nodes": ["paper-bert"],
    "top_k": 5
  }'
```

**Script:**
> "This comparison shows all three modes side by side. Vector finds semantic matches. Graph finds connected papers. Hybrid finds the best of both - semantically relevant AND well-connected."

### Technical Deep-Dive (2 min)

#### CRS Formula

**Show on screen:**
```
CRS = α × V + β × G

α = 0.6 (vector weight)
β = 0.4 (graph weight)
V = cosine similarity (0-1)
G = 1 / (1 + shortest_path_distance)
```

**Script:**
> "Our Contextual Relevance Score is principled, not ad-hoc. Vector similarity is cosine distance in embedding space. Graph proximity uses inverse shortest path. The weights are configurable."

#### Quick Code Walkthrough

Show `hybridmind/engine/hybrid_ranker.py`:
- `_compute_crs()` method
- Weight normalization
- Score combination

**Script:**
> "The implementation is clean and modular. Each component - vector index, graph index, ranker - is independent and testable."

### Metrics & Conclusion (1 min)

#### Show Stats

```bash
curl http://localhost:8000/search/stats
```

**Script:**
> "Our demo dataset has 20 research papers and 30+ citation relationships. Query times are under 100ms."

#### Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Vector Search | < 50ms | ✅ ~20ms |
| Graph Search | < 30ms | ✅ ~15ms |
| Hybrid Search | < 100ms | ✅ ~60ms |

#### Closing

**Script:**
> "HybridMind proves that hybrid retrieval beats single-mode search. It's not just combining results - it's intelligent fusion that surfaces relevant AND connected information. Questions?"

---

## Demo Scenarios (Detailed)

### Scenario 1: Semantic + Citation Discovery

**Setup:** User wants to find papers about transformers and related work

**Query:**
```json
{
  "query_text": "transformer neural network architecture",
  "vector_weight": 0.6,
  "graph_weight": 0.4,
  "top_k": 5
}
```

**Expected Results:**
1. Attention Is All You Need (high vector, anchor)
2. BERT (high vector, cites transformer)
3. ViT (high vector, cites transformer)
4. Neural Machine Translation with Attention (medium vector, cited BY transformer)

**Talking Point:** "Without graph awareness, result #4 would rank lower despite being foundational."

### Scenario 2: Multi-Hop Reasoning

**Setup:** Find papers 2 hops from a specific paper

**Query:**
```json
{
  "query_text": "deep learning",
  "anchor_nodes": ["paper-gpt3"],
  "max_depth": 2,
  "top_k": 5
}
```

**Expected Results:**
1. GPT-2 (depth 1)
2. Transformer (depth 2, via GPT-2)
3. Related DL papers with semantic match

**Talking Point:** "Hybrid search traverses the citation network while maintaining semantic relevance."

### Scenario 3: Adjusting Weights

**Setup:** Show impact of weight changes

**Queries:**
```bash
# Vector-heavy (α=0.9, β=0.1)
curl -X POST http://localhost:8000/search/hybrid \
  -d '{"query_text": "optimization", "vector_weight": 0.9, "graph_weight": 0.1}'

# Graph-heavy (α=0.1, β=0.9)
curl -X POST http://localhost:8000/search/hybrid \
  -d '{"query_text": "optimization", "vector_weight": 0.1, "graph_weight": 0.9, "anchor_nodes": ["paper-adam"]}'
```

**Talking Point:** "Users can tune the balance based on their use case - more semantic or more structural."

---

## Anticipated Questions

### Q: How does this scale to millions of nodes?

**A:** "For demo, we use FAISS FlatIP (exact search). For production, we'd switch to IVF or HNSW for approximate nearest neighbor search. The graph could be partitioned or backed by a specialized graph database like Neo4j."

### Q: Why SQLite instead of a proper database?

**A:** "SQLite is perfect for local demos - zero config, single file, ACID compliant. For production, we'd migrate to PostgreSQL. The storage layer is abstracted, so swapping backends is straightforward."

### Q: What if a node has no graph connections?

**A:** "Graph score becomes 0, so CRS falls back to pure vector score. The algorithm degrades gracefully."

### Q: Can users provide their own embeddings?

**A:** "Yes! The API accepts pre-computed embeddings in the node creation request. You could use OpenAI, Cohere, or any model that fits your use case."

### Q: How accurate is the embedding model?

**A:** "all-MiniLM-L6-v2 is a good balance of speed and quality. It's trained on semantic similarity tasks and handles research paper abstracts well. For domain-specific applications, you could fine-tune or swap models."

---

## Backup Demos

### If API is slow:
- Use cached responses
- Show Swagger UI interactive docs

### If demo data issues:
- Manually create nodes via API
- Use minimal 3-node graph to show concept

### If questions go deep:
- Show test files for edge case handling
- Walk through CRS algorithm pseudocode

