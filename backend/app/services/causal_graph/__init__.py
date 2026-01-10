"""
Causal Graph System for Root Cause Analysis.

This module provides a graph-based approach to identifying root causes
by modeling services and their dependencies, then computing causality scores
based on metric correlations, log patterns, and temporal alignment.
"""

from app.services.causal_graph.models import (
    CausalNode,
    CausalEdge,
    CausalGraph,
    NodeType,
    EdgeType,
)
from app.services.causal_graph.correlations import (
    CorrelationCalculator,
    pearson_correlation,
    spearman_correlation,
    log_cooccurrence_score,
)
from app.services.causal_graph.graph_builder import CausalGraphBuilder
from app.services.causal_graph.scorer import RootCauseScorer, compute_root_cause_scores

__all__ = [
    # Models
    "CausalNode",
    "CausalEdge",
    "CausalGraph",
    "NodeType",
    "EdgeType",
    # Correlations
    "CorrelationCalculator",
    "pearson_correlation",
    "spearman_correlation",
    "log_cooccurrence_score",
    # Builder & Scorer
    "CausalGraphBuilder",
    "RootCauseScorer",
    "compute_root_cause_scores",
]

