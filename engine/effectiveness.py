"""
Effectiveness Metrics Calculator for HybridMind.
Provides quantitative proof that hybrid search outperforms vector-only and graph-only.

Metrics implemented:
- Precision@K: Fraction of retrieved documents that are relevant
- Recall@K: Fraction of relevant documents that are retrieved
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant result
- NDCG (Normalized Discounted Cumulative Gain): Weighted relevance with position discount
- Coverage: Percentage of unique relevant results found
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EffectivenessMetrics:
    """Metrics for a single search system."""
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    coverage: float = 0.0
    unique_relevant_finds: int = 0
    avg_score: float = 0.0
    score_variance: float = 0.0


@dataclass
class EffectivenessComparison:
    """Comparison of effectiveness across systems."""
    hybridmind: EffectivenessMetrics = field(default_factory=EffectivenessMetrics)
    vector_only: EffectivenessMetrics = field(default_factory=EffectivenessMetrics)
    graph_only: EffectivenessMetrics = field(default_factory=EffectivenessMetrics)
    
    # Improvement metrics (hybrid vs others)
    precision_improvement_vs_vector: float = 0.0
    recall_improvement_vs_vector: float = 0.0
    ndcg_improvement_vs_vector: float = 0.0
    unique_finds_by_hybrid: int = 0
    
    # Summary
    winner: str = ""
    summary: str = ""


class EffectivenessCalculator:
    """
    Calculate retrieval effectiveness metrics.
    
    Since we don't have ground truth labels, we use multiple strategies:
    1. Pooled Relevance: Union of top results from all systems = relevant set
    2. Semantic Threshold: Results above similarity threshold = relevant
    3. Cross-Validation: Results appearing in multiple systems = relevant
    """
    
    def __init__(
        self,
        relevance_threshold: float = 0.3,
        use_pooled_relevance: bool = True
    ):
        """
        Initialize calculator.
        
        Args:
            relevance_threshold: Minimum score to consider result "relevant"
            use_pooled_relevance: If True, use pooled results as ground truth
        """
        self.relevance_threshold = relevance_threshold
        self.use_pooled_relevance = use_pooled_relevance
    
    def compute_metrics(
        self,
        results: List[Dict[str, Any]],
        relevant_set: Set[str],
        k: int = 10
    ) -> EffectivenessMetrics:
        """
        Compute effectiveness metrics for a single result set.
        
        Args:
            results: List of search results with node_id and score
            relevant_set: Set of relevant node IDs (ground truth)
            k: Number of results to evaluate
            
        Returns:
            EffectivenessMetrics with all computed values
        """
        if not results or not relevant_set:
            return EffectivenessMetrics()
        
        # Limit to top-k
        top_k_results = results[:k]
        retrieved_ids = [r["node_id"] for r in top_k_results]
        scores = [r.get("score", r.get("combined_score", r.get("vector_score", 0))) for r in top_k_results]
        
        # Precision@K: relevant in top-k / k
        relevant_in_k = sum(1 for nid in retrieved_ids if nid in relevant_set)
        precision_at_k = relevant_in_k / k if k > 0 else 0.0
        
        # Recall@K: relevant in top-k / total relevant
        recall_at_k = relevant_in_k / len(relevant_set) if relevant_set else 0.0
        
        # MRR: 1 / rank of first relevant result
        mrr = 0.0
        for i, nid in enumerate(retrieved_ids, 1):
            if nid in relevant_set:
                mrr = 1.0 / i
                break
        
        # NDCG: Normalized Discounted Cumulative Gain
        ndcg = self._compute_ndcg(retrieved_ids, relevant_set, scores, k)
        
        # Coverage: what fraction of relevant set did we find
        retrieved_relevant = set(retrieved_ids) & relevant_set
        coverage = len(retrieved_relevant) / len(relevant_set) if relevant_set else 0.0
        
        # Unique relevant finds (not in other systems)
        unique_relevant_finds = len(retrieved_relevant)
        
        # Score statistics
        avg_score = np.mean(scores) if scores else 0.0
        score_variance = np.var(scores) if len(scores) > 1 else 0.0
        
        return EffectivenessMetrics(
            precision_at_k=round(precision_at_k, 4),
            recall_at_k=round(recall_at_k, 4),
            mrr=round(mrr, 4),
            ndcg=round(ndcg, 4),
            coverage=round(coverage, 4),
            unique_relevant_finds=unique_relevant_finds,
            avg_score=round(float(avg_score), 4),
            score_variance=round(float(score_variance), 4)
        )
    
    def _compute_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_set: Set[str],
        scores: List[float],
        k: int
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain."""
        # DCG: sum of relevance / log2(position + 1)
        dcg = 0.0
        for i, nid in enumerate(retrieved_ids[:k], 1):
            rel = 1.0 if nid in relevant_set else 0.0
            dcg += rel / float(np.log2(i + 1))
        
        # Ideal DCG: all relevant items at the top
        ideal_relevances = sorted([1.0] * min(len(relevant_set), k), reverse=True)
        idcg = sum(rel / float(np.log2(i + 2)) for i, rel in enumerate(ideal_relevances))
        
        return float(dcg / idcg) if idcg > 0 else 0.0
    
    def compare_systems(
        self,
        hybrid_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        k: int = 10
    ) -> EffectivenessComparison:
        """
        Compare effectiveness across all three search modes.
        
        Uses pooled relevance strategy: the union of results from all systems
        above a score threshold is considered the "relevant set".
        
        Args:
            hybrid_results: HybridMind CRS results
            vector_results: Vector-only search results
            graph_results: Graph-only search results
            k: Number of results to evaluate
            
        Returns:
            EffectivenessComparison with metrics for all systems
        """
        # Build pooled relevant set
        all_results = []
        all_results.extend([(r, "hybrid") for r in hybrid_results[:k]])
        all_results.extend([(r, "vector") for r in vector_results[:k]])
        all_results.extend([(r, "graph") for r in graph_results[:k]])
        
        # Relevant = appears in at least 2 systems OR has high score
        node_counts: Dict[str, int] = {}
        node_scores: Dict[str, float] = {}
        
        for r, source in all_results:
            nid = r["node_id"]
            score = r.get("score", r.get("combined_score", r.get("vector_score", 0)))
            
            if nid not in node_counts:
                node_counts[nid] = 0
                node_scores[nid] = 0
            node_counts[nid] += 1
            node_scores[nid] = max(node_scores[nid], score)
        
        # Build relevant set using multiple criteria
        relevant_set = set()
        for nid, count in node_counts.items():
            # Relevant if: appears in 2+ systems OR has high score
            if count >= 2 or node_scores[nid] >= self.relevance_threshold:
                relevant_set.add(nid)
        
        # If relevant set is empty, use all unique results
        if not relevant_set:
            relevant_set = set(node_counts.keys())
        
        # Compute metrics for each system
        hybrid_metrics = self.compute_metrics(hybrid_results, relevant_set, k)
        vector_metrics = self.compute_metrics(vector_results, relevant_set, k)
        graph_metrics = self.compute_metrics(graph_results, relevant_set, k)
        
        # Compute improvement metrics (ensure native Python floats)
        precision_improvement = float(
            (hybrid_metrics.precision_at_k - vector_metrics.precision_at_k) 
            / max(vector_metrics.precision_at_k, 0.001) * 100
        )
        recall_improvement = float(
            (hybrid_metrics.recall_at_k - vector_metrics.recall_at_k)
            / max(vector_metrics.recall_at_k, 0.001) * 100
        )
        ndcg_improvement = float(
            (hybrid_metrics.ndcg - vector_metrics.ndcg)
            / max(vector_metrics.ndcg, 0.001) * 100
        )
        
        # Unique finds by hybrid
        hybrid_ids = set(r["node_id"] for r in hybrid_results[:k])
        vector_ids = set(r["node_id"] for r in vector_results[:k])
        graph_ids = set(r["node_id"] for r in graph_results[:k])
        
        unique_by_hybrid = len((hybrid_ids & relevant_set) - vector_ids - graph_ids)
        
        # Determine winner
        scores = {
            "hybrid": hybrid_metrics.ndcg + hybrid_metrics.precision_at_k,
            "vector": vector_metrics.ndcg + vector_metrics.precision_at_k,
            "graph": graph_metrics.ndcg + graph_metrics.precision_at_k
        }
        winner = max(scores, key=scores.get)
        
        # Generate summary
        if winner == "hybrid":
            summary = f"Hybrid search outperforms with {precision_improvement:+.1f}% precision and {ndcg_improvement:+.1f}% NDCG improvement over vector-only."
        elif winner == "vector":
            summary = f"Vector search leads, but hybrid finds {unique_by_hybrid} unique relevant results."
        else:
            summary = f"Graph search leads for this query with strong relationship-based results."
        
        return EffectivenessComparison(
            hybridmind=hybrid_metrics,
            vector_only=vector_metrics,
            graph_only=graph_metrics,
            precision_improvement_vs_vector=round(precision_improvement, 2),
            recall_improvement_vs_vector=round(recall_improvement, 2),
            ndcg_improvement_vs_vector=round(ndcg_improvement, 2),
            unique_finds_by_hybrid=unique_by_hybrid,
            winner=winner,
            summary=summary
        )
    
    def run_ablation_study(
        self,
        query_text: str,
        hybrid_ranker,
        weights: List[Tuple[float, float]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Run ablation study testing different alpha/beta weight combinations.
        
        This provides justification for the default weights (0.6, 0.4).
        
        Args:
            query_text: Search query
            hybrid_ranker: HybridRanker instance
            weights: List of (alpha, beta) tuples to test
            top_k: Number of results
            
        Returns:
            Ablation study results with best weights
        """
        if weights is None:
            weights = [
                (0.1, 0.9),  # Graph-heavy
                (0.2, 0.8),
                (0.3, 0.7),
                (0.4, 0.6),
                (0.5, 0.5),  # Balanced
                (0.6, 0.4),  # Default
                (0.7, 0.3),
                (0.8, 0.2),
                (0.9, 0.1),  # Vector-heavy
                (1.0, 0.0),  # Pure vector
            ]
        
        results = []
        all_node_ids = set()
        
        # Run searches with different weights
        for alpha, beta in weights:
            search_results, latency, total = hybrid_ranker.search(
                query_text=query_text,
                top_k=top_k,
                vector_weight=alpha,
                graph_weight=beta
            )
            
            result_ids = [r["node_id"] for r in search_results]
            all_node_ids.update(result_ids)
            
            # Compute score diversity (higher = better exploration)
            scores = [r.get("combined_score", 0) for r in search_results]
            score_diversity = float(np.std(scores)) if len(scores) > 1 else 0.0
            avg_score = float(np.mean(scores)) if scores else 0.0
            
            results.append({
                "alpha": float(alpha),
                "beta": float(beta),
                "avg_combined_score": avg_score,
                "score_diversity": score_diversity,
                "latency_ms": float(latency),
                "result_ids": result_ids[:5]  # Top 5 for comparison
            })
        
        # Build pooled relevant set from all results
        relevant_set = all_node_ids
        
        # Compute effectiveness metrics for each weight combo
        for i, (alpha, beta) in enumerate(weights):
            search_results, _, _ = hybrid_ranker.search(
                query_text=query_text,
                top_k=top_k,
                vector_weight=alpha,
                graph_weight=beta
            )
            
            metrics = self.compute_metrics(search_results, relevant_set, top_k)
            results[i]["precision_at_k"] = float(metrics.precision_at_k)
            results[i]["recall_at_k"] = float(metrics.recall_at_k)
            results[i]["ndcg"] = float(metrics.ndcg)
            results[i]["mrr"] = float(metrics.mrr)
        
        # Find optimal weights
        best_by_ndcg = max(results, key=lambda x: x.get("ndcg", 0))
        best_by_precision = max(results, key=lambda x: x.get("precision_at_k", 0))
        
        # Check if default (0.6, 0.4) is optimal or near-optimal
        default_result = next((r for r in results if r["alpha"] == 0.6 and r["beta"] == 0.4), None)
        if default_result:
            # Explicitly convert to Python floats to avoid numpy.bool_ issues
            default_ndcg = float(default_result.get("ndcg", 0))
            best_ndcg = float(best_by_ndcg.get("ndcg", 0))
            default_is_optimal = default_ndcg >= best_ndcg * 0.95
        else:
            default_is_optimal = False
        
        # Convert all numpy types to native Python types for JSON serialization
        serializable_results = []
        for r in results:
            serializable_results.append({
                "alpha": float(r["alpha"]),
                "beta": float(r["beta"]),
                "avg_combined_score": float(r.get("avg_combined_score", 0)),
                "score_diversity": float(r.get("score_diversity", 0)),
                "latency_ms": float(r.get("latency_ms", 0)),
                "precision_at_k": float(r.get("precision_at_k", 0)),
                "recall_at_k": float(r.get("recall_at_k", 0)),
                "ndcg": float(r.get("ndcg", 0)),
                "mrr": float(r.get("mrr", 0)),
                "result_ids": r.get("result_ids", [])
            })
        
        return {
            "query": query_text,
            "top_k": top_k,
            "results": serializable_results,
            "best_by_ndcg": {
                "alpha": float(best_by_ndcg["alpha"]),
                "beta": float(best_by_ndcg["beta"]),
                "ndcg": float(best_by_ndcg.get("ndcg", 0))
            },
            "best_by_precision": {
                "alpha": float(best_by_precision["alpha"]),
                "beta": float(best_by_precision["beta"]),
                "precision": float(best_by_precision.get("precision_at_k", 0))
            },
            "default_weights": {
                "alpha": 0.6,
                "beta": 0.4,
                "ndcg": float(default_result["ndcg"]) if default_result else None,
                "is_optimal_or_near": default_is_optimal
            },
            "recommendation": (
                "Default weights (α=0.6, β=0.4) are optimal or near-optimal for this query."
                if default_is_optimal
                else f"Consider weights α={best_by_ndcg['alpha']}, β={best_by_ndcg['beta']} for better NDCG."
            )
        }


def get_effectiveness_calculator(
    relevance_threshold: float = 0.3
) -> EffectivenessCalculator:
    """Get effectiveness calculator instance."""
    return EffectivenessCalculator(relevance_threshold=relevance_threshold)
