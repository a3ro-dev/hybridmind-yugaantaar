"""
Performance metrics utilities for HybridMind.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from statistics import mean, median, stdev


@dataclass
class QueryMetric:
    """Single query metric."""
    operation: str
    latency_ms: float
    result_count: int
    timestamp: float = field(default_factory=time.time)


class PerformanceMetrics:
    """
    Collects and analyzes performance metrics.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metrics to keep
        """
        self.max_history = max_history
        self._metrics: deque = deque(maxlen=max_history)
        self._operation_counts: Dict[str, int] = {}
    
    def record(
        self,
        operation: str,
        latency_ms: float,
        result_count: int = 0
    ):
        """Record a query metric."""
        metric = QueryMetric(
            operation=operation,
            latency_ms=latency_ms,
            result_count=result_count
        )
        self._metrics.append(metric)
        self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1
    
    def get_summary(self, operation: Optional[str] = None) -> dict:
        """
        Get performance summary.
        
        Args:
            operation: Filter by operation type (optional)
            
        Returns:
            Summary statistics
        """
        metrics = list(self._metrics)
        
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        if not metrics:
            return {
                "count": 0,
                "latency_ms": {"mean": 0, "median": 0, "min": 0, "max": 0}
            }
        
        latencies = [m.latency_ms for m in metrics]
        
        return {
            "count": len(metrics),
            "latency_ms": {
                "mean": round(mean(latencies), 2),
                "median": round(median(latencies), 2),
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
                "stdev": round(stdev(latencies), 2) if len(latencies) > 1 else 0
            },
            "operations": dict(self._operation_counts)
        }
    
    def get_recent(self, count: int = 10) -> List[dict]:
        """Get recent metrics."""
        recent = list(self._metrics)[-count:]
        return [
            {
                "operation": m.operation,
                "latency_ms": round(m.latency_ms, 2),
                "result_count": m.result_count,
                "timestamp": m.timestamp
            }
            for m in recent
        ]
    
    def check_performance_targets(self) -> dict:
        """
        Check if performance meets targets from PRD.
        
        Targets:
        - Node Create: < 50ms
        - Node Read: < 5ms
        - Vector Search: < 50ms
        - Graph Traversal: < 30ms
        - Hybrid Search: < 100ms
        """
        targets = {
            "create_node": 50,
            "read_node": 5,
            "vector_search": 50,
            "graph_search": 30,
            "hybrid_search": 100
        }
        
        results = {}
        
        for operation, target in targets.items():
            summary = self.get_summary(operation)
            if summary["count"] > 0:
                avg_latency = summary["latency_ms"]["mean"]
                results[operation] = {
                    "target_ms": target,
                    "actual_ms": avg_latency,
                    "meets_target": avg_latency <= target,
                    "sample_count": summary["count"]
                }
            else:
                results[operation] = {
                    "target_ms": target,
                    "actual_ms": None,
                    "meets_target": None,
                    "sample_count": 0
                }
        
        return results
    
    def clear(self):
        """Clear all metrics."""
        self._metrics.clear()
        self._operation_counts.clear()


# Global metrics instance
_metrics: Optional[PerformanceMetrics] = None


def get_metrics() -> PerformanceMetrics:
    """Get global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = PerformanceMetrics()
    return _metrics

