"""
Demo data loader for HybridMind.
Creates a research paper knowledge graph with semantic connections.
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybridmind.api.dependencies import get_db_manager

# Sample research papers dataset (AI/ML focused)
DEMO_PAPERS = [
    {
        "id": "paper-transformer",
        "text": "Attention Is All You Need. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality.",
        "metadata": {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani", "Shazeer", "Parmar", "Uszkoreit", "Jones", "Gomez", "Kaiser", "Polosukhin"],
            "year": 2017,
            "tags": ["transformer", "attention", "NLP", "machine translation"],
            "venue": "NeurIPS"
        }
    },
    {
        "id": "paper-bert",
        "text": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
        "metadata": {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Devlin", "Chang", "Lee", "Toutanova"],
            "year": 2018,
            "tags": ["BERT", "transformer", "NLP", "pre-training", "language model"],
            "venue": "NAACL"
        }
    },
    {
        "id": "paper-gpt2",
        "text": "Language Models are Unsupervised Multitask Learners. We demonstrate that language models begin to learn NLP tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText. GPT-2 is a large transformer-based language model with 1.5 billion parameters.",
        "metadata": {
            "title": "Language Models are Unsupervised Multitask Learners",
            "authors": ["Radford", "Wu", "Child", "Luan", "Amodei", "Sutskever"],
            "year": 2019,
            "tags": ["GPT", "language model", "unsupervised learning", "transformer"],
            "venue": "OpenAI"
        }
    },
    {
        "id": "paper-gpt3",
        "text": "Language Models are Few-Shot Learners. We demonstrate that scaling up language models greatly improves task-agnostic, few-shot performance. GPT-3, an autoregressive language model with 175 billion parameters, achieves strong performance on many NLP datasets.",
        "metadata": {
            "title": "Language Models are Few-Shot Learners",
            "authors": ["Brown", "Mann", "Ryder", "Subbiah", "Kaplan"],
            "year": 2020,
            "tags": ["GPT-3", "few-shot learning", "language model", "scaling"],
            "venue": "NeurIPS"
        }
    },
    {
        "id": "paper-resnet",
        "text": "Deep Residual Learning for Image Recognition. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs.",
        "metadata": {
            "title": "Deep Residual Learning for Image Recognition",
            "authors": ["He", "Zhang", "Ren", "Sun"],
            "year": 2015,
            "tags": ["ResNet", "deep learning", "computer vision", "image classification"],
            "venue": "CVPR"
        }
    },
    {
        "id": "paper-vit",
        "text": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. We show that the reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.",
        "metadata": {
            "title": "Vision Transformer (ViT)",
            "authors": ["Dosovitskiy", "Beyer", "Kolesnikov", "Weissenborn", "Zhai"],
            "year": 2020,
            "tags": ["ViT", "vision transformer", "computer vision", "image classification"],
            "venue": "ICLR"
        }
    },
    {
        "id": "paper-adam",
        "text": "Adam: A Method for Stochastic Optimization. We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. Adam combines advantages of AdaGrad and RMSProp.",
        "metadata": {
            "title": "Adam: A Method for Stochastic Optimization",
            "authors": ["Kingma", "Ba"],
            "year": 2014,
            "tags": ["optimization", "Adam", "deep learning", "gradient descent"],
            "venue": "ICLR"
        }
    },
    {
        "id": "paper-dropout",
        "text": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting. We present dropout, a technique for addressing overfitting in neural networks. The key idea is to randomly drop units from the neural network during training to prevent units from co-adapting too much.",
        "metadata": {
            "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "authors": ["Srivastava", "Hinton", "Krizhevsky", "Sutskever", "Salakhutdinov"],
            "year": 2014,
            "tags": ["dropout", "regularization", "neural networks", "overfitting"],
            "venue": "JMLR"
        }
    },
    {
        "id": "paper-batch-norm",
        "text": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. We propose batch normalization, which allows us to use much higher learning rates and be less careful about initialization, and in some cases eliminates the need for Dropout.",
        "metadata": {
            "title": "Batch Normalization",
            "authors": ["Ioffe", "Szegedy"],
            "year": 2015,
            "tags": ["batch normalization", "training", "deep learning", "optimization"],
            "venue": "ICML"
        }
    },
    {
        "id": "paper-word2vec",
        "text": "Efficient Estimation of Word Representations in Vector Space. We propose two novel model architectures for computing continuous vector representations of words from very large data sets. Word2vec learns high-quality word vectors that encode semantic relationships.",
        "metadata": {
            "title": "Word2Vec",
            "authors": ["Mikolov", "Chen", "Corrado", "Dean"],
            "year": 2013,
            "tags": ["word2vec", "word embeddings", "NLP", "representation learning"],
            "venue": "ICLR"
        }
    },
    {
        "id": "paper-elmo",
        "text": "Deep contextualized word representations. We introduce a new type of deep contextualized word representation that models both complex characteristics of word use and how these uses vary across linguistic contexts. ELMo representations are learned from bidirectional LSTMs.",
        "metadata": {
            "title": "ELMo: Deep contextualized word representations",
            "authors": ["Peters", "Neumann", "Iyyer", "Gardner", "Clark", "Lee", "Zettlemoyer"],
            "year": 2018,
            "tags": ["ELMo", "contextual embeddings", "NLP", "LSTM"],
            "venue": "NAACL"
        }
    },
    {
        "id": "paper-gan",
        "text": "Generative Adversarial Nets. We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability.",
        "metadata": {
            "title": "Generative Adversarial Networks",
            "authors": ["Goodfellow", "Pouget-Abadie", "Mirza", "Xu", "Warde-Farley", "Ozair", "Courville", "Bengio"],
            "year": 2014,
            "tags": ["GAN", "generative models", "adversarial learning", "deep learning"],
            "venue": "NeurIPS"
        }
    },
    {
        "id": "paper-vae",
        "text": "Auto-Encoding Variational Bayes. We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. VAEs combine neural networks with variational inference.",
        "metadata": {
            "title": "Variational Autoencoders",
            "authors": ["Kingma", "Welling"],
            "year": 2013,
            "tags": ["VAE", "variational inference", "generative models", "latent variables"],
            "venue": "ICLR"
        }
    },
    {
        "id": "paper-seq2seq",
        "text": "Sequence to Sequence Learning with Neural Networks. We present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered LSTM to map the input sequence to a vector of fixed dimensionality.",
        "metadata": {
            "title": "Sequence to Sequence Learning with Neural Networks",
            "authors": ["Sutskever", "Vinyals", "Le"],
            "year": 2014,
            "tags": ["seq2seq", "LSTM", "machine translation", "encoder-decoder"],
            "venue": "NeurIPS"
        }
    },
    {
        "id": "paper-attention-mt",
        "text": "Neural Machine Translation by Jointly Learning to Align and Translate. We conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture. We introduce attention mechanism for neural machine translation.",
        "metadata": {
            "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
            "authors": ["Bahdanau", "Cho", "Bengio"],
            "year": 2014,
            "tags": ["attention", "machine translation", "NLP", "seq2seq"],
            "venue": "ICLR"
        }
    },
    {
        "id": "paper-roberta",
        "text": "RoBERTa: A Robustly Optimized BERT Pretraining Approach. We present a replication study of BERT pretraining that carefully measures the impact of many key hyperparameters and training data size. We find that BERT was significantly undertrained.",
        "metadata": {
            "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "authors": ["Liu", "Ott", "Goyal", "Du", "Joshi", "Chen", "Levy", "Lewis", "Zettlemoyer", "Stoyanov"],
            "year": 2019,
            "tags": ["RoBERTa", "BERT", "pre-training", "NLP"],
            "venue": "arXiv"
        }
    },
    {
        "id": "paper-t5",
        "text": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. We explore transfer learning techniques for NLP by converting every text processing task into a text-to-text format. T5 achieves state-of-the-art results on many benchmarks.",
        "metadata": {
            "title": "T5: Text-to-Text Transfer Transformer",
            "authors": ["Raffel", "Shazeer", "Roberts", "Lee", "Narang", "Matena", "Zhou", "Li", "Liu"],
            "year": 2019,
            "tags": ["T5", "transfer learning", "transformer", "NLP"],
            "venue": "JMLR"
        }
    },
    {
        "id": "paper-clip",
        "text": "Learning Transferable Visual Models From Natural Language Supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations. CLIP learns visual concepts from natural language.",
        "metadata": {
            "title": "CLIP: Learning Transferable Visual Models From Natural Language Supervision",
            "authors": ["Radford", "Kim", "Hallacy", "Ramesh", "Goh", "Agarwal"],
            "year": 2021,
            "tags": ["CLIP", "multimodal", "vision-language", "contrastive learning"],
            "venue": "ICML"
        }
    },
    {
        "id": "paper-llama",
        "text": "LLaMA: Open and Efficient Foundation Language Models. We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, showing that it is possible to train state-of-the-art models using publicly available datasets.",
        "metadata": {
            "title": "LLaMA: Open and Efficient Foundation Language Models",
            "authors": ["Touvron", "Lavril", "Izacard", "Martinet", "Lachaux"],
            "year": 2023,
            "tags": ["LLaMA", "language model", "foundation model", "open source"],
            "venue": "arXiv"
        }
    },
    {
        "id": "paper-rag",
        "text": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation. RAG models combine pre-trained parametric and non-parametric memory for language generation.",
        "metadata": {
            "title": "Retrieval-Augmented Generation (RAG)",
            "authors": ["Lewis", "Perez", "Piktus", "Petroni", "Karpukhin"],
            "year": 2020,
            "tags": ["RAG", "retrieval", "generation", "knowledge-intensive"],
            "venue": "NeurIPS"
        }
    }
]

# Citation relationships between papers
DEMO_EDGES = [
    # BERT cites Transformer
    {"source": "paper-bert", "target": "paper-transformer", "type": "cites", "weight": 1.0},
    # GPT-2 cites Transformer
    {"source": "paper-gpt2", "target": "paper-transformer", "type": "cites", "weight": 1.0},
    # GPT-3 cites GPT-2
    {"source": "paper-gpt3", "target": "paper-gpt2", "type": "cites", "weight": 1.0},
    # ViT cites Transformer
    {"source": "paper-vit", "target": "paper-transformer", "type": "cites", "weight": 1.0},
    # ViT cites ResNet
    {"source": "paper-vit", "target": "paper-resnet", "type": "cites", "weight": 0.8},
    # BERT cites ELMo
    {"source": "paper-bert", "target": "paper-elmo", "type": "cites", "weight": 0.9},
    # ELMo cites Word2Vec
    {"source": "paper-elmo", "target": "paper-word2vec", "type": "cites", "weight": 0.8},
    # RoBERTa cites BERT
    {"source": "paper-roberta", "target": "paper-bert", "type": "cites", "weight": 1.0},
    # T5 cites BERT
    {"source": "paper-t5", "target": "paper-bert", "type": "cites", "weight": 0.9},
    # T5 cites Transformer
    {"source": "paper-t5", "target": "paper-transformer", "type": "cites", "weight": 1.0},
    # Seq2Seq cites LSTM-related
    {"source": "paper-attention-mt", "target": "paper-seq2seq", "type": "cites", "weight": 1.0},
    # Transformer cites attention-mt
    {"source": "paper-transformer", "target": "paper-attention-mt", "type": "cites", "weight": 1.0},
    # ResNet cites Dropout
    {"source": "paper-resnet", "target": "paper-dropout", "type": "cites", "weight": 0.7},
    # ResNet cites Batch Norm
    {"source": "paper-resnet", "target": "paper-batch-norm", "type": "cites", "weight": 0.9},
    # CLIP cites ViT
    {"source": "paper-clip", "target": "paper-vit", "type": "cites", "weight": 0.9},
    # CLIP cites GPT-2
    {"source": "paper-clip", "target": "paper-gpt2", "type": "cites", "weight": 0.7},
    # RAG cites BERT
    {"source": "paper-rag", "target": "paper-bert", "type": "cites", "weight": 0.9},
    # LLaMA cites GPT-3
    {"source": "paper-llama", "target": "paper-gpt3", "type": "cites", "weight": 0.8},
    # LLaMA cites Transformer
    {"source": "paper-llama", "target": "paper-transformer", "type": "cites", "weight": 1.0},
    # VAE and GAN related
    {"source": "paper-vae", "target": "paper-dropout", "type": "cites", "weight": 0.5},
    {"source": "paper-gan", "target": "paper-dropout", "type": "cites", "weight": 0.5},
    # Same topic relationships
    {"source": "paper-bert", "target": "paper-roberta", "type": "same_topic", "weight": 1.0},
    {"source": "paper-gpt2", "target": "paper-gpt3", "type": "same_topic", "weight": 1.0},
    {"source": "paper-vit", "target": "paper-resnet", "type": "same_topic", "weight": 0.7},
    {"source": "paper-word2vec", "target": "paper-elmo", "type": "same_topic", "weight": 0.9},
    {"source": "paper-gan", "target": "paper-vae", "type": "same_topic", "weight": 0.8},
    {"source": "paper-adam", "target": "paper-batch-norm", "type": "same_topic", "weight": 0.7},
]


def load_demo_data():
    """Load demo dataset into HybridMind."""
    print("Initializing HybridMind...")
    db_manager = get_db_manager()
    
    sqlite_store = db_manager.sqlite_store
    vector_index = db_manager.vector_index
    graph_index = db_manager.graph_index
    embedding_engine = db_manager.embedding_engine
    
    # Check if data already exists
    existing_nodes = sqlite_store.count_nodes()
    if existing_nodes > 0:
        print(f"Database already contains {existing_nodes} nodes.")
        response = input("Clear existing data and reload? (y/n): ")
        if response.lower() != 'y':
            print("Keeping existing data.")
            return
        
        # Clear existing data
        print("Clearing existing data...")
        for node in sqlite_store.list_nodes(limit=10000):
            sqlite_store.delete_node(node["id"])
        vector_index.clear()
        graph_index.clear()
    
    print(f"\nLoading {len(DEMO_PAPERS)} papers...")
    
    # Load papers
    for i, paper in enumerate(DEMO_PAPERS, 1):
        print(f"  [{i}/{len(DEMO_PAPERS)}] {paper['metadata']['title'][:50]}...")
        
        # Generate embedding
        embedding = embedding_engine.embed(paper["text"])
        
        # Store in SQLite
        sqlite_store.create_node(
            node_id=paper["id"],
            text=paper["text"],
            metadata=paper["metadata"],
            embedding=embedding
        )
        
        # Add to vector index
        vector_index.add(paper["id"], embedding)
        
        # Add to graph index
        graph_index.add_node(paper["id"])
    
    print(f"\nLoading {len(DEMO_EDGES)} relationships...")
    
    # Load edges
    for i, edge in enumerate(DEMO_EDGES, 1):
        edge_id = f"edge-{edge['source']}-{edge['target']}-{edge['type']}"
        
        # Store in SQLite
        sqlite_store.create_edge(
            edge_id=edge_id,
            source_id=edge["source"],
            target_id=edge["target"],
            edge_type=edge["type"],
            weight=edge["weight"]
        )
        
        # Add to graph index
        graph_index.add_edge(
            source_id=edge["source"],
            target_id=edge["target"],
            edge_type=edge["type"],
            weight=edge["weight"],
            edge_id=edge_id
        )
    
    # Save indexes
    print("\nSaving indexes...")
    db_manager.save_indexes()
    
    # Print summary
    stats = db_manager.get_stats()
    print("\n" + "="*50)
    print("Demo Data Loaded Successfully!")
    print("="*50)
    print(f"  Nodes: {stats['total_nodes']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Edge Types: {stats['edge_types']}")
    print("\nTry these example queries:")
    print('  1. Vector search: "transformer attention mechanisms"')
    print('  2. Hybrid search: "deep learning optimization" with anchor=paper-adam')
    print('  3. Graph traversal: Start from paper-transformer, depth=2')


if __name__ == "__main__":
    load_demo_data()

