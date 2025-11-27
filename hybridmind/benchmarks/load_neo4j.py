"""
Load ArXiv dataset into Neo4j graph database for comparison.
Requires Neo4j running locally (docker recommended).

To start Neo4j:
    docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/password neo4j:latest
"""

import json
import time
from typing import List, Dict, Tuple, Optional

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Neo4j driver not installed. Run: pip install neo4j")


class Neo4jLoader:
    """Load data into Neo4j graph database."""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "password"):
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._clear_database()
    
    def _clear_database(self):
        """Clear existing data."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def close(self):
        """Close the driver connection."""
        self.driver.close()
    
    def load_papers(self, papers: List[Dict], batch_size: int = 100) -> Dict:
        """Load papers as nodes into Neo4j."""
        stats = {
            "total_papers": len(papers),
            "loaded": 0,
            "time": 0
        }
        
        start = time.time()
        
        with self.driver.session() as session:
            for i in range(0, len(papers), batch_size):
                batch = papers[i:i+batch_size]
                
                # Create nodes in batch
                for paper in batch:
                    session.run(
                        """
                        CREATE (p:Paper {
                            id: $id,
                            title: $title,
                            text: $text
                        })
                        """,
                        id=paper['id'],
                        title=paper.get('title', ''),
                        text=paper['text'][:1000]  # Limit text size
                    )
                    stats["loaded"] += 1
                
                if (i + batch_size) % 200 == 0:
                    print(f"  Loaded {stats['loaded']}/{len(papers)} papers...")
            
            # Create index for faster lookups
            session.run("CREATE INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)")
        
        stats["time"] = time.time() - start
        return stats
    
    def load_edges(self, edges: List[Tuple[str, str, str]]) -> Dict:
        """Load edges as relationships into Neo4j."""
        stats = {
            "total_edges": len(edges),
            "loaded": 0,
            "time": 0
        }
        
        start = time.time()
        
        with self.driver.session() as session:
            for source, target, edge_type in edges:
                try:
                    result = session.run(
                        """
                        MATCH (a:Paper {id: $source})
                        MATCH (b:Paper {id: $target})
                        CREATE (a)-[r:RELATES {type: $type}]->(b)
                        RETURN r
                        """,
                        source=source,
                        target=target,
                        type=edge_type
                    )
                    if result.single():
                        stats["loaded"] += 1
                except Exception as e:
                    pass  # Skip if nodes don't exist
        
        stats["time"] = time.time() - start
        return stats
    
    def search_graph(self, start_id: str, depth: int = 2) -> Tuple[List[Dict], float]:
        """Graph traversal search - what Neo4j does best."""
        start = time.time()
        
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start:Paper {id: $start_id})-[*1..""" + str(depth) + """]->(related:Paper)
                RETURN DISTINCT related.id as id, related.title as title, 
                       length(path) as depth
                ORDER BY depth
                LIMIT 20
                """,
                start_id=start_id
            )
            
            results = [dict(record) for record in result]
        
        query_time = (time.time() - start) * 1000  # ms
        return results, query_time
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        
        return {"nodes": nodes, "edges": edges}


def load_from_files(papers_file: str = "data/arxiv_papers.json",
                    edges_file: str = "data/arxiv_edges.json",
                    neo4j_uri: str = "bolt://localhost:7687",
                    neo4j_user: str = "neo4j",
                    neo4j_password: str = "password") -> Optional[Dict]:
    """Load dataset from JSON files into Neo4j."""
    
    if not NEO4J_AVAILABLE:
        print("Neo4j driver not installed")
        return None
    
    print("Loading ArXiv dataset into Neo4j...")
    
    try:
        # Load data
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        with open(edges_file, 'r', encoding='utf-8') as f:
            edges_data = json.load(f)
        edges = [(e['source'], e['target'], e['type']) for e in edges_data]
        
        # Initialize loader
        loader = Neo4jLoader(neo4j_uri, neo4j_user, neo4j_password)
        
        # Load papers
        print(f"\nLoading {len(papers)} papers...")
        paper_stats = loader.load_papers(papers)
        print(f"  Papers loaded in {paper_stats['time']:.2f}s")
        
        # Load edges
        print(f"\nLoading {len(edges)} edges...")
        edge_stats = loader.load_edges(edges)
        print(f"  Edges loaded in {edge_stats['time']:.2f}s")
        
        # Get stats
        stats = loader.get_stats()
        print(f"\nNeo4j database ready:")
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Edges: {stats['edges']}")
        
        loader.close()
        return stats
        
    except Exception as e:
        print(f"Neo4j error: {e}")
        print("Make sure Neo4j is running on localhost:7687")
        return None


if __name__ == "__main__":
    load_from_files()

