"""
Learned CRS Ranker for HybridMind.

Advanced ranking with:
- Edge-type specific weights
- Non-linear score transformations
- Hop-distance decay
- Harmony bonus for dual-channel relevance
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    """A candidate result with scores from multiple sources."""
    node_id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: float
    graph_score: float
    edge_types: List[str] = field(default_factory=list)
    hop_distance: Optional[int] = None
    path: Optional[List[str]] = None


class LearnedCRSRanker:
    """
    Advanced Contextual Relevance Score (CRS) ranker with learned weights.
    
    Features:
    1. Edge-type specific importance weights
    2. Non-linear score transformations (boost high-confidence matches)
    3. Hop-distance decay (closer connections = more relevant)
    4. Harmony bonus for nodes scoring well on both channels
    
    CRS Formula:
    ```
    final_score = (α × v_transformed) + (β × g_transformed) + harmony_bonus
    ```
    
    Where:
    - v_transformed = vector_score with confidence boost
    - g_transformed = graph_score × hop_decay × edge_weight
    - harmony_bonus = bonus if both scores are high
    """
    
    # Default edge-type importance weights
    DEFAULT_EDGE_WEIGHTS = {
        "cites": 1.0,
        "cited_by": 0.9,
        "same_author": 0.95,
        "co_authored": 0.95,
        "same_topic": 0.7,
        "related_to": 0.5,
        "extends": 0.85,
        "implements": 0.8,
        "similar_to": 0.6,
        "default": 0.4
    }
    
    def __init__(
        self,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        edge_weights: Optional[Dict[str, float]] = None,
        hop_decay: float = 0.7,
        vector_boost_threshold: float = 0.7,
        graph_boost_threshold: float = 0.6,
        harmony_factor: float = 0.1
    ):
        """
        Initialize learned ranker.
        
        Args:
            vector_weight: Base weight for vector similarity (α)
            graph_weight: Base weight for graph proximity (β)
            edge_weights: Importance weights per edge type
            hop_decay: Score multiplier per hop (0.7 = 30% decay per hop)
            vector_boost_threshold: Boost vector scores above this threshold
            graph_boost_threshold: Boost graph scores above this threshold
            harmony_factor: Bonus multiplier for dual-channel relevance
        """
        self.base_vector_weight = vector_weight
        self.base_graph_weight = graph_weight
        self.edge_weights = edge_weights or self.DEFAULT_EDGE_WEIGHTS.copy()
        self.hop_decay = hop_decay
        self.vector_boost_threshold = vector_boost_threshold
        self.graph_boost_threshold = graph_boost_threshold
        self.harmony_factor = harmony_factor
        
        logger.info(
            f"LearnedCRSRanker initialized: "
            f"α={vector_weight}, β={graph_weight}, "
            f"hop_decay={hop_decay}"
        )
    
    def compute_score(
        self,
        candidate: ScoredCandidate,
        vector_weight: Optional[float] = None,
        graph_weight: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute advanced CRS with non-linear transformations.
        
        Args:
            candidate: Candidate with raw scores
            vector_weight: Override vector weight
            graph_weight: Override graph weight
            
        Returns:
            Tuple of (final_score, explanation_dict)
        """
        vw = vector_weight if vector_weight is not None else self.base_vector_weight
        gw = graph_weight if graph_weight is not None else self.base_graph_weight
        
        # Normalize weights
        total = vw + gw
        if total > 0:
            vw, gw = vw / total, gw / total
        else:
            vw, gw = 0.5, 0.5
        
        # --- Vector Score Transformation ---
        v_raw = candidate.vector_score
        v_transformed = self._transform_vector_score(v_raw)
        
        # --- Graph Score Transformation ---
        g_raw = candidate.graph_score
        g_transformed = self._transform_graph_score(
            g_raw,
            hop_distance=candidate.hop_distance,
            edge_types=candidate.edge_types
        )
        
        # --- Harmony Bonus ---
        # Reward nodes that score well on BOTH channels
        harmony_bonus = self._compute_harmony_bonus(v_transformed, g_transformed)
        
        # --- Final CRS ---
        final_score = (vw * v_transformed) + (gw * g_transformed) + harmony_bonus
        
        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, final_score))
        
        # Build explanation
        explanation = {
            "raw_vector_score": round(v_raw, 4),
            "transformed_vector_score": round(v_transformed, 4),
            "raw_graph_score": round(g_raw, 4),
            "transformed_graph_score": round(g_transformed, 4),
            "hop_distance": candidate.hop_distance,
            "edge_types": candidate.edge_types,
            "harmony_bonus": round(harmony_bonus, 4),
            "weights": {
                "vector": round(vw, 3),
                "graph": round(gw, 3)
            },
            "final_crs": round(final_score, 4)
        }
        
        return final_score, explanation
    
    def _transform_vector_score(self, score: float) -> float:
        """
        Apply non-linear transformation to vector score.
        Boosts high-confidence semantic matches.
        """
        if score <= 0:
            return 0.0
        
        # Sigmoid-like boost for high-confidence matches
        if score > self.vector_boost_threshold:
            # Asymptotic boost: pushes score closer to 1.0
            boost_amount = (score - self.vector_boost_threshold) / (1.0 - self.vector_boost_threshold)
            boost = (1.0 - score) * 0.3 * boost_amount
            return score + boost
        
        return score
    
    def _transform_graph_score(
        self,
        score: float,
        hop_distance: Optional[int] = None,
        edge_types: Optional[List[str]] = None
    ) -> float:
        """
        Apply hop decay and edge-type weighting to graph score.
        """
        if score <= 0:
            return 0.0
        
        transformed = score
        
        # Apply hop decay (exponential decay with distance)
        if hop_distance is not None and hop_distance > 0:
            hop_multiplier = self.hop_decay ** hop_distance
            transformed *= hop_multiplier
        
        # Apply edge-type weight (use max weight if multiple types)
        if edge_types:
            edge_multiplier = max(
                self.edge_weights.get(et, self.edge_weights.get("default", 0.4))
                for et in edge_types
            )
            transformed *= edge_multiplier
        
        # Boost strongly connected nodes
        if transformed > self.graph_boost_threshold:
            boost_amount = (transformed - self.graph_boost_threshold) / (1.0 - self.graph_boost_threshold)
            boost = (1.0 - transformed) * 0.2 * boost_amount
            transformed += boost
        
        return transformed
    
    def _compute_harmony_bonus(
        self,
        v_score: float,
        g_score: float
    ) -> float:
        """
        Compute bonus for nodes with high scores on both channels.
        Uses harmonic mean scaled by harmony factor.
        """
        if v_score <= 0.5 or g_score <= 0.5:
            return 0.0
        
        # Harmonic mean gives higher weight to balanced scores
        harmonic_mean = 2 * v_score * g_score / (v_score + g_score)
        
        # Scale by harmony factor
        return harmonic_mean * self.harmony_factor
    
    def rank_candidates(
        self,
        candidates: List[ScoredCandidate],
        vector_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[ScoredCandidate, float, Dict[str, Any]]]:
        """
        Rank all candidates and return sorted results.
        
        Args:
            candidates: List of candidates with raw scores
            vector_weight: Override vector weight
            graph_weight: Override graph weight
            top_k: Limit results (None = return all)
            
        Returns:
            List of (candidate, final_score, explanation) sorted by score
        """
        scored = []
        
        for candidate in candidates:
            score, explanation = self.compute_score(
                candidate,
                vector_weight=vector_weight,
                graph_weight=graph_weight
            )
            scored.append((candidate, score, explanation))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            scored = scored[:top_k]
        
        return scored
    
    def generate_reasoning(
        self,
        candidate: ScoredCandidate,
        explanation: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation for a ranking.
        """
        parts = []
        
        v_score = explanation["raw_vector_score"]
        g_score = explanation["raw_graph_score"]
        harmony = explanation["harmony_bonus"]
        
        # Vector component
        if v_score >= 0.8:
            parts.append(f"High semantic match ({v_score:.0%})")
        elif v_score >= 0.5:
            parts.append(f"Moderate semantic match ({v_score:.0%})")
        else:
            parts.append(f"Weak semantic match ({v_score:.0%})")
        
        # Graph component
        if g_score > 0:
            hop_info = ""
            if candidate.hop_distance is not None:
                hop_info = f", {candidate.hop_distance} hop{'s' if candidate.hop_distance > 1 else ''} away"
            
            edge_info = ""
            if candidate.edge_types:
                edge_info = f" via {candidate.edge_types[0]}"
            
            if g_score >= 0.5:
                parts.append(f"strongly connected{edge_info}{hop_info}")
            else:
                parts.append(f"graph connection found{edge_info}{hop_info}")
        else:
            parts.append("no graph connection")
        
        # Harmony bonus
        if harmony > 0.01:
            parts.append(f"+{harmony:.0%} dual-channel bonus")
        
        return "; ".join(parts)
    
    def update_edge_weight(self, edge_type: str, weight: float):
        """Update weight for a specific edge type."""
        if 0.0 <= weight <= 1.0:
            self.edge_weights[edge_type] = weight
            logger.info(f"Updated edge weight: {edge_type} = {weight}")
    
    def save_config(self, path: str):
        """Save ranker configuration to JSON file."""
        config = {
            "base_vector_weight": self.base_vector_weight,
            "base_graph_weight": self.base_graph_weight,
            "edge_weights": self.edge_weights,
            "hop_decay": self.hop_decay,
            "vector_boost_threshold": self.vector_boost_threshold,
            "graph_boost_threshold": self.graph_boost_threshold,
            "harmony_factor": self.harmony_factor
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Ranker config saved to {path}")
    
    def load_config(self, path: str):
        """Load ranker configuration from JSON file."""
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {path}")
            return
        
        with open(path, "r") as f:
            config = json.load(f)
        
        self.base_vector_weight = config.get("base_vector_weight", self.base_vector_weight)
        self.base_graph_weight = config.get("base_graph_weight", self.base_graph_weight)
        self.edge_weights = config.get("edge_weights", self.edge_weights)
        self.hop_decay = config.get("hop_decay", self.hop_decay)
        self.vector_boost_threshold = config.get("vector_boost_threshold", self.vector_boost_threshold)
        self.graph_boost_threshold = config.get("graph_boost_threshold", self.graph_boost_threshold)
        self.harmony_factor = config.get("harmony_factor", self.harmony_factor)
        
        logger.info(f"Ranker config loaded from {path}")


def create_candidate_from_result(
    result: Dict[str, Any],
    graph_score: float = 0.0,
    edge_types: Optional[List[str]] = None,
    hop_distance: Optional[int] = None,
    path: Optional[List[str]] = None
) -> ScoredCandidate:
    """
    Helper to create ScoredCandidate from search result dict.
    """
    return ScoredCandidate(
        node_id=result.get("node_id", ""),
        text=result.get("text", ""),
        metadata=result.get("metadata", {}),
        vector_score=result.get("vector_score", 0.0),
        graph_score=graph_score,
        edge_types=edge_types or [],
        hop_distance=hop_distance,
        path=path
    )

