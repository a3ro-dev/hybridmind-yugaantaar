"""
Configuration management for HybridMind.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "HybridMind"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_path: str = "data/hybridmind.db"
    
    # Vector Index
    vector_index_path: str = "data/vector.index"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Graph Index
    graph_index_path: str = "data/graph.pkl"
    
    # Search Defaults
    default_top_k: int = 10
    default_vector_weight: float = 0.6
    default_graph_weight: float = 0.4
    max_traversal_depth: int = 5
    
    # Performance
    batch_size: int = 32
    cache_size: int = 1000
    
    # API
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_prefix = "HYBRIDMIND_"
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_data_dir(self) -> Path:
        """Get the data directory, creating it if necessary."""
        data_dir = Path(self.database_path).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def get_database_url(self) -> str:
        """Get the SQLite database URL."""
        return f"sqlite+aiosqlite:///{self.database_path}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()

