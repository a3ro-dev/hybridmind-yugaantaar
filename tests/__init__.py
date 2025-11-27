"""
HybridMind Test Suite

Based on Devfolio Hackathon Test Case Markers for:
"Vector + Graph Native Database for Efficient AI Retrieval"

Test Categories:
---------------
1. API & CRUD (TC-API-01 to TC-API-05)
   - test_api_crud.py

2. Vector Search (TC-VEC-01 to TC-VEC-03)
   - test_vector_search.py

3. Graph Traversal (TC-GRAPH-01 to TC-GRAPH-03)
   - test_graph_traversal.py

4. Hybrid Search (TC-HYB-01 to TC-HYB-03)
   - test_hybrid_search.py

5. Canonical Dataset (Section 10)
   - test_canonical_dataset.py

Priority Markers:
-----------------
- P0: Blocking (must pass)
- P1: Important
- P2: Nice-to-have

Scope Markers:
--------------
- integration: Integration tests
- system: System/end-to-end tests
- edge_case: Edge case tests
- performance: Performance tests

Usage:
------
# Run all tests
pytest tests/ -v

# Run by priority
pytest tests/ -v -m priority_p0

# Run by scope
pytest tests/ -v -m integration

# Run specific test file
pytest tests/test_api_crud.py -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html
"""
