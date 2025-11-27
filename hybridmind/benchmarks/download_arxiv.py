"""
Download and prepare the ML-ArXiv-Papers dataset from HuggingFace.
https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import random

def download_arxiv_dataset(sample_size: int = 1000) -> List[Dict]:
    """
    Download ML-ArXiv-Papers dataset from HuggingFace.
    
    Args:
        sample_size: Number of papers to sample (full dataset has 118k papers)
    
    Returns:
        List of paper dictionaries with title and abstract
    """
    print(f"Downloading ML-ArXiv-Papers dataset (sampling {sample_size} papers)...")
    
    try:
        from datasets import load_dataset
        
        # Load the dataset
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
        
        # Sample if needed (full dataset has ~118k papers)
        total_size = len(dataset)
        print(f"Dataset loaded: {total_size} total papers")
        
        if sample_size < total_size:
            indices = random.sample(range(total_size), sample_size)
            papers = [dataset[i] for i in indices]
        else:
            papers = [dataset[i] for i in range(total_size)]
        
        # Clean and format papers
        cleaned_papers = []
        for i, paper in enumerate(papers):
            title = paper.get('title', '').strip()
            abstract = paper.get('abstract', '').strip()
            
            if title and abstract and len(abstract) > 50:
                cleaned_papers.append({
                    'id': f"arxiv-{i}",
                    'title': title,
                    'abstract': abstract,
                    'text': f"{title}\n\n{abstract}"
                })
        
        print(f"Cleaned dataset: {len(cleaned_papers)} papers")
        return cleaned_papers
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to synthetic data...")
        return generate_synthetic_papers(sample_size)


def generate_synthetic_papers(count: int = 100) -> List[Dict]:
    """Generate synthetic ML papers for testing."""
    topics = [
        ("transformer", "attention", "BERT", "GPT", "language model"),
        ("convolutional", "CNN", "image", "vision", "ResNet"),
        ("reinforcement learning", "policy", "reward", "agent", "Q-learning"),
        ("GAN", "generative", "discriminator", "adversarial", "synthesis"),
        ("optimization", "gradient", "Adam", "SGD", "learning rate"),
        ("neural network", "deep learning", "backpropagation", "activation"),
        ("embedding", "vector", "representation", "semantic", "similarity"),
        ("graph neural network", "GNN", "node", "edge", "message passing"),
        ("recurrent", "LSTM", "sequence", "time series", "RNN"),
        ("autoencoder", "VAE", "latent", "reconstruction", "encoding"),
    ]
    
    papers = []
    for i in range(count):
        topic_idx = i % len(topics)
        topic_words = topics[topic_idx]
        
        title = f"Advances in {topic_words[0].title()} Networks using {topic_words[1].title()} Mechanisms"
        abstract = f"""This paper presents novel approaches to {topic_words[0]} architectures 
        with improved {topic_words[1]} capabilities. We demonstrate significant improvements 
        in {topic_words[2]} performance through our proposed {topic_words[3]} method. 
        Experiments on benchmark datasets show that our approach achieves state-of-the-art 
        results in {topic_words[4]} tasks, outperforming existing methods by a significant margin.
        Our contributions include: (1) a new {topic_words[0]} architecture, (2) efficient 
        {topic_words[1]} computation, and (3) comprehensive evaluation on multiple benchmarks."""
        
        papers.append({
            'id': f"synthetic-{i}",
            'title': title,
            'abstract': abstract,
            'text': f"{title}\n\n{abstract}",
            'topic': topic_words[0]
        })
    
    return papers


def generate_citation_edges(papers: List[Dict], edge_ratio: float = 2.0) -> List[Tuple[str, str, str]]:
    """
    Generate realistic citation relationships between papers.
    Papers on similar topics are more likely to cite each other.
    
    Args:
        papers: List of paper dictionaries
        edge_ratio: Average number of edges per node
    
    Returns:
        List of (source_id, target_id, edge_type) tuples
    """
    edges = []
    num_edges = int(len(papers) * edge_ratio)
    
    # Group papers by rough topic (based on keywords in title)
    topic_groups = {}
    keywords = ['transformer', 'cnn', 'gan', 'lstm', 'graph', 'embedding', 
                'optimization', 'reinforcement', 'autoencoder', 'attention']
    
    for paper in papers:
        text_lower = paper['text'].lower()
        for kw in keywords:
            if kw in text_lower:
                if kw not in topic_groups:
                    topic_groups[kw] = []
                topic_groups[kw].append(paper['id'])
                break
        else:
            if 'other' not in topic_groups:
                topic_groups['other'] = []
            topic_groups['other'].append(paper['id'])
    
    paper_ids = [p['id'] for p in papers]
    
    for _ in range(num_edges):
        # 70% chance of same-topic citation, 30% cross-topic
        if random.random() < 0.7 and topic_groups:
            topic = random.choice(list(topic_groups.keys()))
            group = topic_groups[topic]
            if len(group) >= 2:
                source, target = random.sample(group, 2)
            else:
                source, target = random.sample(paper_ids, 2)
        else:
            source, target = random.sample(paper_ids, 2)
        
        if source != target:
            edge_type = random.choice(['cites', 'related_to', 'extends', 'compares_with'])
            edges.append((source, target, edge_type))
    
    # Remove duplicates
    edges = list(set(edges))
    print(f"Generated {len(edges)} citation edges")
    return edges


def save_dataset(papers: List[Dict], edges: List[Tuple], output_dir: str = "data"):
    """Save dataset to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save papers
    papers_file = output_path / "arxiv_papers.json"
    with open(papers_file, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(papers)} papers to {papers_file}")
    
    # Save edges
    edges_file = output_path / "arxiv_edges.json"
    edges_data = [{"source": s, "target": t, "type": typ} for s, t, typ in edges]
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump(edges_data, f, indent=2)
    print(f"Saved {len(edges)} edges to {edges_file}")
    
    return papers_file, edges_file


if __name__ == "__main__":
    # Download and prepare dataset
    papers = download_arxiv_dataset(sample_size=500)  # Start with 500 for testing
    edges = generate_citation_edges(papers, edge_ratio=2.5)
    save_dataset(papers, edges)
    
    print("\nDataset ready for benchmarking!")
    print(f"  Papers: {len(papers)}")
    print(f"  Edges: {len(edges)}")

