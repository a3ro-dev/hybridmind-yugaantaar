"""
Persistence utilities for HybridMind.
Handles snapshots and data export/import.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class SnapshotManager:
    """
    Manages database snapshots for backup and restore.
    """
    
    def __init__(self, data_dir: str = "data", snapshot_dir: str = "snapshots"):
        """
        Initialize snapshot manager.
        
        Args:
            data_dir: Directory containing database files
            snapshot_dir: Directory for storing snapshots
        """
        self.data_dir = Path(data_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    def create_snapshot(self, name: Optional[str] = None) -> str:
        """
        Create a snapshot of the current database state.
        
        Args:
            name: Optional snapshot name (defaults to timestamp)
            
        Returns:
            Snapshot directory path
        """
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        snapshot_path = self.snapshot_dir / name
        snapshot_path.mkdir(exist_ok=True)
        
        # Copy database files
        files_to_backup = [
            "hybridmind.db",
            "vector.index",
            "vector.faiss",
            "graph.pkl"
        ]
        
        for filename in files_to_backup:
            src = self.data_dir / filename
            if src.exists():
                shutil.copy2(src, snapshot_path / filename)
        
        # Create manifest
        manifest = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "files": [f for f in files_to_backup if (self.data_dir / f).exists()]
        }
        
        with open(snapshot_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        return str(snapshot_path)
    
    def restore_snapshot(self, name: str) -> bool:
        """
        Restore database from a snapshot.
        
        Args:
            name: Snapshot name to restore
            
        Returns:
            True if successful
        """
        snapshot_path = self.snapshot_dir / name
        
        if not snapshot_path.exists():
            raise ValueError(f"Snapshot {name} not found")
        
        # Read manifest
        manifest_path = snapshot_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            files = manifest.get("files", [])
        else:
            files = list(snapshot_path.glob("*"))
        
        # Restore files
        for filename in files:
            src = snapshot_path / filename
            if src.exists() and src.is_file():
                shutil.copy2(src, self.data_dir / filename)
        
        return True
    
    def list_snapshots(self) -> list:
        """List all available snapshots."""
        snapshots = []
        
        for path in self.snapshot_dir.iterdir():
            if path.is_dir():
                manifest_path = path / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    snapshots.append(manifest)
                else:
                    snapshots.append({
                        "name": path.name,
                        "created_at": datetime.fromtimestamp(
                            path.stat().st_mtime
                        ).isoformat()
                    })
        
        return sorted(snapshots, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def delete_snapshot(self, name: str) -> bool:
        """Delete a snapshot."""
        snapshot_path = self.snapshot_dir / name
        
        if not snapshot_path.exists():
            return False
        
        shutil.rmtree(snapshot_path)
        return True


def export_to_json(sqlite_store, output_path: str):
    """
    Export database to JSON format.
    
    Args:
        sqlite_store: SQLite store instance
        output_path: Output file path
    """
    nodes = sqlite_store.list_nodes(limit=100000)
    edges = sqlite_store.get_all_edges()
    
    # Convert datetime objects to strings
    for node in nodes:
        node["created_at"] = node["created_at"].isoformat() if node.get("created_at") else None
        node["updated_at"] = node["updated_at"].isoformat() if node.get("updated_at") else None
        # Remove embedding from export (too large)
        node.pop("embedding", None)
    
    for edge in edges:
        edge["created_at"] = edge["created_at"].isoformat() if edge.get("created_at") else None
    
    data = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "nodes": nodes,
        "edges": edges
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def import_from_json(sqlite_store, embedding_engine, input_path: str):
    """
    Import database from JSON format.
    
    Args:
        sqlite_store: SQLite store instance
        embedding_engine: Embedding engine for generating embeddings
        input_path: Input file path
    """
    with open(input_path) as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    # Import nodes
    for node in nodes:
        embedding = embedding_engine.embed(node["text"])
        sqlite_store.create_node(
            node_id=node["id"],
            text=node["text"],
            metadata=node.get("metadata", {}),
            embedding=embedding
        )
    
    # Import edges
    for edge in edges:
        sqlite_store.create_edge(
            edge_id=edge["id"],
            source_id=edge["source_id"],
            target_id=edge["target_id"],
            edge_type=edge["type"],
            weight=edge.get("weight", 1.0),
            metadata=edge.get("metadata", {})
        )

