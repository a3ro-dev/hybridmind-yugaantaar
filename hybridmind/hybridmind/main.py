"""
HybridMind FastAPI Application Entry Point.

Vector + Graph Native Database for AI Retrieval.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from hybridmind.config import settings
from hybridmind.api.nodes import router as nodes_router
from hybridmind.api.edges import router as edges_router
from hybridmind.api.search import router as search_router
from hybridmind.api.dependencies import get_db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting HybridMind...")
    db_manager = get_db_manager()
    stats = db_manager.get_stats()
    logger.info(f"Database loaded: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HybridMind...")
    db_manager.save_indexes()
    db_manager.close()
    logger.info("HybridMind shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="HybridMind",
    description="""
## Vector + Graph Native Database for AI Retrieval

HybridMind is a high-performance hybrid database that combines vector embeddings 
with graph-based relationships for superior AI retrieval.

### Key Features

- **Vector Search**: Semantic similarity using cosine distance with FAISS
- **Graph Search**: Relationship traversal using NetworkX
- **Hybrid Search**: Contextual Relevance Score (CRS) combining both approaches

### Hybrid Scoring Algorithm (CRS)

```
CRS = α × vector_score + β × graph_score
```

Where:
- `α` = vector_weight (default 0.6)
- `β` = graph_weight (default 0.4)
- `vector_score` = cosine similarity (0-1)
- `graph_score` = inverse shortest path distance (0-1)

### Use Case

Research paper knowledge graph with semantic search and citation relationships.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(nodes_router)
app.include_router(edges_router)
app.include_router(search_router)


# Root endpoint
@app.get("/", tags=["Utility"])
async def root():
    """Welcome endpoint with API overview."""
    return {
        "name": "HybridMind",
        "version": "1.0.0",
        "description": "Vector + Graph Native Database for AI Retrieval",
        "docs": "/docs",
        "endpoints": {
            "nodes": "/nodes",
            "edges": "/edges",
            "search": {
                "vector": "/search/vector",
                "graph": "/search/graph",
                "hybrid": "/search/hybrid",
                "compare": "/search/compare"
            },
            "stats": "/search/stats"
        }
    }


# Health check endpoint
@app.get("/health", tags=["Utility"])
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        db_manager = get_db_manager()
        stats = db_manager.get_stats()
        return {
            "status": "healthy",
            "database": "connected",
            "nodes": stats["total_nodes"],
            "edges": stats["total_edges"],
            "embedding_model": stats["embedding_model"]
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Snapshot endpoint
@app.post("/snapshot", tags=["Utility"])
async def create_snapshot():
    """Create a persistence snapshot of indexes."""
    try:
        db_manager = get_db_manager()
        db_manager.save_indexes()
        return {
            "status": "success",
            "message": "Indexes saved to disk"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hybridmind.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )

