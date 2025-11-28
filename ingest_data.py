"""
Script to ingest data.txt into HybridMind database via POST /nodes
"""

import requests
import re

API_URL = "http://localhost:8000"

def load_and_chunk_data(filepath: str):
    """Load data.txt and split into chunks by paragraphs."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\n+', content)
    
    # Filter out short/empty paragraphs and navigation noise
    chunks = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        # Skip very short paragraphs or navigation elements
        if len(para) > 50 and not para.startswith(('Contents', 'Search', 'Donate', 'Tools', 'Appearance')):
            chunks.append({
                "text": para[:2000],  # Limit chunk size
                "metadata": {
                    "source": "data.txt",
                    "chunk_index": i,
                    "type": "wikipedia_article",
                    "subject": "Franz Kafka"
                }
            })
    
    return chunks

def ingest_to_database(chunks: list):
    """POST each chunk to /nodes endpoint."""
    print(f"Ingesting {len(chunks)} chunks into database...")
    
    success = 0
    failed = 0
    
    for i, chunk in enumerate(chunks):
        try:
            response = requests.post(
                f"{API_URL}/nodes",
                json=chunk,
                timeout=30
            )
            if response.status_code in (200, 201):
                success += 1
                print(f"  [{i+1}/{len(chunks)}] ✓ Ingested chunk {i}")
            else:
                failed += 1
                print(f"  [{i+1}/{len(chunks)}] ✗ Failed: {response.status_code}")
        except Exception as e:
            failed += 1
            print(f"  [{i+1}/{len(chunks)}] ✗ Error: {e}")
    
    print(f"\nDone! Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    chunks = load_and_chunk_data("data.txt")
    print(f"Found {len(chunks)} meaningful chunks")
    ingest_to_database(chunks)
