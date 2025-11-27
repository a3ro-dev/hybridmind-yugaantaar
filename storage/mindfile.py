"""
HybridMind Custom Database Format (.mind)

A .mind file is a directory-based database format that bundles:
- SQLite database (nodes, edges, metadata)
- FAISS vector index
- NetworkX graph index
- Manifest with metadata

Structure:
    database.mind/
    ├── manifest.json      # Version, stats, metadata
    ├── store.db           # SQLite database
    ├── vectors.faiss      # FAISS index
    ├── vectors.map        # ID mappings for FAISS
    └── graph.nx           # NetworkX graph (pickle)

This creates a portable, self-contained knowledge base.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# File extension
MIND_EXTENSION = ".mind"
MANIFEST_FILE = "manifest.json"
SQLITE_FILE = "store.db"
VECTOR_INDEX_FILE = "vectors.faiss"
VECTOR_MAP_FILE = "vectors.map"
GRAPH_FILE = "graph.nx"


class MindFile:
    """
    HybridMind database file format (.mind).
    
    A .mind file is a directory containing all database components:
    - SQLite for persistent storage
    - FAISS for vector search
    - NetworkX for graph operations
    
    Usage:
        # Create new database
        db = MindFile("knowledge.mind")
        db.initialize()
        
        # Open existing
        db = MindFile("knowledge.mind")
        paths = db.get_paths()
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, path: str):
        """
        Initialize MindFile handler.
        
        Args:
            path: Path to .mind file (directory)
        """
        # Ensure .mind extension
        if not path.endswith(MIND_EXTENSION):
            path = path + MIND_EXTENSION
        
        self.path = Path(path)
        self.name = self.path.stem
        
    @property
    def exists(self) -> bool:
        """Check if the .mind file exists."""
        return self.path.exists() and self.path.is_dir()
    
    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_FILE
    
    @property
    def sqlite_path(self) -> Path:
        return self.path / SQLITE_FILE
    
    @property
    def vector_index_path(self) -> Path:
        return self.path / VECTOR_INDEX_FILE
    
    @property
    def vector_map_path(self) -> Path:
        return self.path / VECTOR_MAP_FILE
    
    @property
    def graph_path(self) -> Path:
        return self.path / GRAPH_FILE
    
    def get_paths(self) -> Dict[str, str]:
        """Get all component file paths."""
        return {
            "root": str(self.path),
            "manifest": str(self.manifest_path),
            "sqlite": str(self.sqlite_path),
            "vector_index": str(self.vector_index_path),
            "vector_map": str(self.vector_map_path),
            "graph": str(self.graph_path)
        }
    
    def initialize(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize a new .mind database.
        
        Creates the directory structure and manifest file.
        
        Args:
            metadata: Optional metadata to include in manifest
            
        Returns:
            True if created successfully
        """
        if self.exists:
            logger.warning(f"MindFile already exists: {self.path}")
            return False
        
        try:
            # Create directory
            self.path.mkdir(parents=True, exist_ok=True)
            
            # Create manifest
            manifest = {
                "format": "HybridMind",
                "version": self.VERSION,
                "name": self.name,
                "created": datetime.now(timezone.utc).isoformat(),
                "modified": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "sqlite": SQLITE_FILE,
                    "vector_index": VECTOR_INDEX_FILE,
                    "vector_map": VECTOR_MAP_FILE,
                    "graph": GRAPH_FILE
                },
                "stats": {
                    "nodes": 0,
                    "edges": 0,
                    "vectors": 0
                },
                "metadata": metadata or {}
            }
            
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Created MindFile: {self.path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create MindFile: {e}")
            return False
    
    def read_manifest(self) -> Optional[Dict[str, Any]]:
        """Read the manifest file."""
        if not self.manifest_path.exists():
            return None
        
        try:
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read manifest: {e}")
            return None
    
    def update_manifest(self, updates: Dict[str, Any]) -> bool:
        """Update manifest with new values."""
        manifest = self.read_manifest()
        if manifest is None:
            return False
        
        try:
            # Deep merge updates
            for key, value in updates.items():
                if isinstance(value, dict) and key in manifest:
                    manifest[key].update(value)
                else:
                    manifest[key] = value
            
            manifest["modified"] = datetime.now(timezone.utc).isoformat()
            
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update manifest: {e}")
            return False
    
    def update_stats(self, nodes: int = None, edges: int = None, vectors: int = None) -> bool:
        """Update database statistics in manifest."""
        stats = {}
        if nodes is not None:
            stats["nodes"] = nodes
        if edges is not None:
            stats["edges"] = edges
        if vectors is not None:
            stats["vectors"] = vectors
        
        if stats:
            return self.update_manifest({"stats": stats})
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get database info including size and stats."""
        manifest = self.read_manifest() or {}
        
        # Calculate size
        total_size = 0
        component_sizes = {}
        
        for name, path in [
            ("sqlite", self.sqlite_path),
            ("vector_index", self.vector_index_path),
            ("graph", self.graph_path)
        ]:
            if path.exists():
                size = path.stat().st_size
                component_sizes[name] = size
                total_size += size
        
        return {
            "path": str(self.path),
            "name": self.name,
            "exists": self.exists,
            "version": manifest.get("version", "unknown"),
            "created": manifest.get("created"),
            "modified": manifest.get("modified"),
            "stats": manifest.get("stats", {}),
            "size_bytes": total_size,
            "size_human": format_size(total_size),
            "component_sizes": component_sizes,
            "metadata": manifest.get("metadata", {})
        }
    
    def export(self, output_path: str, compress: bool = True) -> Optional[str]:
        """
        Export the .mind database to a portable archive.
        
        Args:
            output_path: Path for the exported file
            compress: Whether to compress (creates .mind.zip)
            
        Returns:
            Path to exported file, or None if failed
        """
        if not self.exists:
            logger.error("Cannot export: MindFile does not exist")
            return None
        
        try:
            if compress:
                # Create zip archive
                if not output_path.endswith('.zip'):
                    output_path = output_path + '.zip'
                
                shutil.make_archive(
                    output_path.replace('.zip', ''),
                    'zip',
                    self.path.parent,
                    self.path.name
                )
                logger.info(f"Exported to: {output_path}")
                return output_path
            else:
                # Copy directory
                shutil.copytree(self.path, output_path)
                logger.info(f"Exported to: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    @classmethod
    def import_from(cls, archive_path: str, target_path: str) -> Optional['MindFile']:
        """
        Import a .mind database from an archive.
        
        Args:
            archive_path: Path to .mind.zip or .mind directory
            target_path: Where to extract/copy
            
        Returns:
            MindFile instance, or None if failed
        """
        try:
            if archive_path.endswith('.zip'):
                # Extract zip
                shutil.unpack_archive(archive_path, target_path)
                # Find the .mind directory
                for item in Path(target_path).iterdir():
                    if item.suffix == MIND_EXTENSION:
                        return cls(str(item))
            else:
                # Copy directory
                if not target_path.endswith(MIND_EXTENSION):
                    target_path = target_path + MIND_EXTENSION
                shutil.copytree(archive_path, target_path)
                return cls(target_path)
                
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return None
    
    def delete(self) -> bool:
        """Delete the .mind database."""
        if not self.exists:
            return True
        
        try:
            shutil.rmtree(self.path)
            logger.info(f"Deleted MindFile: {self.path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            return False


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def list_mind_files(directory: str = ".") -> list:
    """List all .mind files in a directory."""
    mind_files = []
    for item in Path(directory).iterdir():
        if item.is_dir() and item.suffix == MIND_EXTENSION:
            mf = MindFile(str(item))
            mind_files.append(mf.get_info())
    return mind_files


# Convenience function for creating default database
def create_default_mind(name: str = "hybridmind", data_dir: str = "data") -> MindFile:
    """Create the default HybridMind database."""
    path = os.path.join(data_dir, name)
    mind = MindFile(path)
    
    if not mind.exists:
        mind.initialize(metadata={
            "description": "HybridMind Vector + Graph Database",
            "author": "CodeHashira"
        })
    
    return mind

