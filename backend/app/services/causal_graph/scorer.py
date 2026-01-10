"""
Root Cause Scorer for Causal Graphs.

Computes per-node root cause scores by analyzing:
- Node anomaly scores
- Graph topology (upstream/downstream relationships)
- Edge weights (correlations)
- Temporal ordering (earlier = more likely cause)

Uses a propagation algorithm similar to PageRank to distribute
"blame" through the causal graph.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from app.services.causal_graph.models import (
    CausalGraph,
    CausalNode,
    CausalEdge,
    NodeType,
)


@dataclass
class ScoringConfig:
    """Configuration for the root cause scoring algorithm."""
    
    # Weight factors for different score components
    anomaly_weight: float = 0.35       # Local anomaly score
    topology_weight: float = 0.25      # Graph structure (sources vs sinks)
    propagation_weight: float = 0.25   # Blame propagation
    temporal_weight: float = 0.15      # Time ordering
    
    # Propagation parameters
    damping_factor: float = 0.85       # Similar to PageRank damping
    max_iterations: int = 20           # Maximum iterations for convergence
    convergence_threshold: float = 1e-4  # When to stop iterating
    
    # Temporal scoring parameters
    max_time_bonus: float = 0.3        # Maximum bonus for earliest anomaly
    
    # Node type weights (some types more likely to be root causes)
    node_type_weights: Dict[NodeType, float] = None
    
    def __post_init__(self):
        if self.node_type_weights is None:
            self.node_type_weights = {
                NodeType.DATABASE: 1.2,
                NodeType.EXTERNAL_DEPENDENCY: 1.3,
                NodeType.CACHE: 1.1,
                NodeType.MESSAGE_QUEUE: 1.1,
                NodeType.SERVICE: 1.0,
                NodeType.LOAD_BALANCER: 0.8,
                NodeType.UNKNOWN: 0.9,
            }


class RootCauseScorer:
    """
    Computes root cause scores for nodes in a causal graph.
    
    The scoring algorithm combines multiple signals:
    
    1. **Local Anomaly Score**: How anomalous is the node's own metrics/logs?
    
    2. **Topology Score**: Nodes with more outgoing edges (affecting many downstream
       components) are more likely to be root causes. Computed using graph structure.
    
    3. **Propagation Score**: Uses iterative blame propagation where nodes
       receive blame from their downstream effects. Similar to reverse PageRank.
    
    4. **Temporal Score**: Nodes whose anomalies appeared first are more likely
       to be the root cause.
    
    The final score is a weighted combination of these factors.
    """
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize the scorer.
        
        Args:
            config: Scoring configuration parameters
        """
        self.config = config or ScoringConfig()
    
    def compute_scores(self, graph: CausalGraph) -> Dict[str, float]:
        """
        Compute root cause scores for all nodes in the graph.
        
        This is the main entry point. It modifies the graph in place,
        setting root_cause_score on each node, and returns a dictionary
        of node_id -> score.
        
        Args:
            graph: The causal graph to score
            
        Returns:
            Dictionary mapping node IDs to root cause scores
        """
        if not graph.nodes:
            return {}
        
        # 1. Compute local anomaly scores (already set during graph building)
        anomaly_scores = {
            nid: node.anomaly_score 
            for nid, node in graph.nodes.items()
        }
        
        # 2. Compute topology scores
        topology_scores = self._compute_topology_scores(graph)
        
        # 3. Compute propagation scores
        propagation_scores = self._compute_propagation_scores(graph)
        
        # 4. Compute temporal scores
        temporal_scores = self._compute_temporal_scores(graph)
        
        # 5. Combine scores with weights
        final_scores = {}
        for node_id, node in graph.nodes.items():
            # Get component scores
            anomaly = anomaly_scores.get(node_id, 0)
            topology = topology_scores.get(node_id, 0)
            propagation = propagation_scores.get(node_id, 0)
            temporal = temporal_scores.get(node_id, 0)
            
            # Apply node type weight
            type_weight = self.config.node_type_weights.get(node.node_type, 1.0)
            
            # Weighted combination
            score = (
                self.config.anomaly_weight * anomaly +
                self.config.topology_weight * topology +
                self.config.propagation_weight * propagation +
                self.config.temporal_weight * temporal
            ) * type_weight
            
            # Normalize to [0, 1] - ensures all scores are comparable
            # regardless of type weights or component score distributions
            final_scores[node_id] = min(1.0, max(0.0, score))
            
            # Update node
            node.root_cause_score = final_scores[node_id]
        
        return final_scores
    
    def _compute_topology_scores(self, graph: CausalGraph) -> Dict[str, float]:
        """
        Compute topology-based scores.
        
        Nodes that:
        - Have many outgoing edges (affect many components) → higher score
        - Have few incoming edges (not caused by others) → higher score
        - Are connected to high-anomaly nodes → higher score
        """
        scores = {}
        
        for node_id, node in graph.nodes.items():
            outgoing = graph.get_outgoing_edges(node_id)
            incoming = graph.get_incoming_edges(node_id)
            
            # Base score from edge counts
            # More outgoing = more impact = more likely root cause
            out_factor = min(1.0, len(outgoing) / max(len(graph.nodes) - 1, 1))
            
            # Fewer incoming = less likely to be caused by others
            in_factor = 1.0 - min(1.0, len(incoming) / max(len(graph.nodes) - 1, 1))
            
            # Edge weight contribution
            edge_weight_sum = sum(e.final_weight for e in outgoing)
            edge_factor = edge_weight_sum / max(len(outgoing), 1) if outgoing else 0
            
            # Downstream anomaly contribution
            downstream_anomaly = 0
            for edge in outgoing:
                target = graph.get_node(edge.target_id)
                if target:
                    downstream_anomaly += target.anomaly_score * edge.final_weight
            downstream_factor = downstream_anomaly / max(len(outgoing), 1) if outgoing else 0
            
            # Combine factors
            score = (
                0.3 * out_factor +
                0.2 * in_factor +
                0.2 * edge_factor +
                0.3 * downstream_factor
            )
            
            scores[node_id] = score
        
        return scores
    
    def _compute_propagation_scores(self, graph: CausalGraph) -> Dict[str, float]:
        """
        Compute blame propagation scores using reverse PageRank-like algorithm.
        
        Algorithm Overview:
        ------------------
        This implements a "reverse blame propagation" inspired by PageRank:
        
        1. INITIALIZATION: Start with anomaly-weighted scores. Nodes with higher
           anomalies get more initial "blame budget".
        
        2. PROPAGATION: For each iteration, blame flows BACKWARDS through edges.
           If A → B (A affects B), and B has problems, some blame propagates to A.
           This is "reverse" because we're finding causes, not effects.
        
        3. DAMPING: With probability (1 - damping_factor), we "teleport" to uniform
           distribution. This prevents blame from concentrating in cycles and ensures
           all nodes retain some baseline score.
        
        4. CONVERGENCE: Stop when scores change less than threshold between iterations.
        
        Why Reverse Direction:
        ---------------------
        In incident analysis, symptoms (high anomaly) appear in downstream services.
        We want to trace blame BACK to the root cause, so we propagate scores in
        the reverse direction of the dependency edges.
        
        Determinism:
        -----------
        This algorithm is deterministic given the same graph structure and anomaly scores.
        Node iteration order doesn't affect the final converged result.
        """
        n = len(graph.nodes)
        if n == 0:
            return {}
        
        # Initialize scores uniformly
        scores = {nid: 1.0 / n for nid in graph.nodes}
        
        # Get anomaly-weighted initial distribution
        # Nodes with higher anomalies start with more "blame"
        total_anomaly = sum(node.anomaly_score for node in graph.nodes.values())
        if total_anomaly > 0:
            for node_id, node in graph.nodes.items():
                scores[node_id] = node.anomaly_score / total_anomaly
        
        # Iterative propagation until convergence
        for iteration in range(self.config.max_iterations):
            new_scores = {}
            max_delta = 0
            
            for node_id in graph.nodes:
                # Collect incoming blame from downstream nodes
                # (reverse of dependency direction - blame flows back to causes)
                incoming_blame = 0
                downstream = graph.get_outgoing_edges(node_id)
                
                for edge in downstream:
                    target_id = edge.target_id
                    if target_id in scores:
                        # Blame flows back from target, weighted by:
                        # - target's current blame score
                        # - edge strength (correlation/dependency weight)
                        # - edge confidence
                        incoming_blame += (
                            scores[target_id] * 
                            edge.final_weight * 
                            edge.confidence
                        )
                
                # Apply damping factor (prevents over-concentration)
                # Random teleport component: (1 - d) / n
                # Propagated component: d * incoming_blame
                new_score = (
                    (1 - self.config.damping_factor) / n +
                    self.config.damping_factor * incoming_blame
                )
                
                # Also incorporate node's own anomaly (self-blame)
                # This ensures high-anomaly nodes retain some score even if
                # they don't receive blame from downstream
                node = graph.get_node(node_id)
                if node:
                    new_score += self.config.damping_factor * 0.5 * node.anomaly_score
                
                new_scores[node_id] = new_score
                max_delta = max(max_delta, abs(new_score - scores[node_id]))
            
            scores = new_scores
            
            # Check convergence - stop if scores stabilized
            if max_delta < self.config.convergence_threshold:
                break
        
        # Normalize to [0, 1] for comparable scores
        max_score = max(scores.values()) if scores else 1
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def _compute_temporal_scores(self, graph: CausalGraph) -> Dict[str, float]:
        """
        Compute temporal scores based on anomaly timing.
        
        Nodes whose anomalies appeared first are more likely to be
        root causes (cause precedes effect).
        """
        scores = {}
        
        # Collect nodes with known first anomaly times
        timed_nodes = [
            (node_id, node.first_anomaly_time)
            for node_id, node in graph.nodes.items()
            if node.first_anomaly_time is not None
        ]
        
        if not timed_nodes:
            # No temporal information available
            return {nid: 0.5 for nid in graph.nodes}
        
        # Sort by time (earliest first)
        timed_nodes.sort(key=lambda x: x[1])
        
        earliest_time = timed_nodes[0][1]
        latest_time = timed_nodes[-1][1]
        time_range = (latest_time - earliest_time).total_seconds()
        
        for node_id, node in graph.nodes.items():
            if node.first_anomaly_time is None:
                # Unknown time, assign middle score
                scores[node_id] = 0.5
            elif time_range == 0:
                # All anomalies at same time
                scores[node_id] = 1.0
            else:
                # Score based on relative timing (earlier = higher)
                seconds_from_start = (node.first_anomaly_time - earliest_time).total_seconds()
                scores[node_id] = 1.0 - (seconds_from_start / time_range)
        
        return scores
    
    def get_root_cause_summary(
        self,
        graph: CausalGraph,
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get a summary of the top root cause candidates.
        
        Args:
            graph: Scored causal graph
            top_n: Number of candidates to return
            
        Returns:
            List of candidate summaries with scores and evidence
        """
        # Ensure scores are computed
        if not any(n.root_cause_score > 0 for n in graph.nodes.values()):
            self.compute_scores(graph)
        
        # Get ranked nodes
        ranked = sorted(
            graph.nodes.values(),
            key=lambda n: n.root_cause_score,
            reverse=True
        )[:top_n]
        
        summaries = []
        for node in ranked:
            # Gather evidence
            evidence = []
            
            if node.anomaly_score > 0.5:
                evidence.append(f"High anomaly score: {node.anomaly_score:.2f}")
            
            if node.logs.error_count > 0:
                evidence.append(f"{node.logs.error_count} error logs detected")
            
            if node.first_anomaly_time:
                evidence.append(f"First anomaly at: {node.first_anomaly_time.isoformat()}")
            
            # Check downstream impact
            downstream = graph.get_outgoing_edges(node.id)
            if downstream:
                impact_nodes = [graph.get_node(e.target_id) for e in downstream]
                impact_names = [n.name for n in impact_nodes if n]
                if impact_names:
                    evidence.append(f"Impacts: {', '.join(impact_names)}")
            
            # Get relevant metrics
            metrics_summary = {}
            if node.metrics.error_rate is not None:
                metrics_summary["error_rate"] = f"{node.metrics.error_rate:.1%}"
            if node.metrics.latency_p95_ms is not None:
                metrics_summary["latency_p95"] = f"{node.metrics.latency_p95_ms:.0f}ms"
            
            summary = {
                "node_id": node.id,
                "name": node.name,
                "type": node.node_type.value,
                "root_cause_score": node.root_cause_score,
                "confidence": self._score_to_confidence(node.root_cause_score),
                "anomaly_score": node.anomaly_score,
                "evidence": evidence,
                "metrics": metrics_summary,
                "error_count": node.logs.error_count,
            }
            summaries.append(summary)
        
        return summaries
    
    def _score_to_confidence(self, score: float) -> str:
        """Convert numeric score to confidence level."""
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very low"
    
    def explain_score(
        self,
        graph: CausalGraph,
        node_id: str
    ) -> Dict[str, Any]:
        """
        Explain how a node's root cause score was computed.
        
        Args:
            graph: The causal graph
            node_id: ID of node to explain
            
        Returns:
            Dictionary with score breakdown and explanation
        """
        node = graph.get_node(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}
        
        # Recompute individual components
        anomaly_scores = {nid: n.anomaly_score for nid, n in graph.nodes.items()}
        topology_scores = self._compute_topology_scores(graph)
        propagation_scores = self._compute_propagation_scores(graph)
        temporal_scores = self._compute_temporal_scores(graph)
        
        # Get type weight
        type_weight = self.config.node_type_weights.get(node.node_type, 1.0)
        
        breakdown = {
            "node_id": node_id,
            "name": node.name,
            "type": node.node_type.value,
            "final_score": node.root_cause_score,
            "type_weight": type_weight,
            "components": {
                "anomaly": {
                    "weight": self.config.anomaly_weight,
                    "score": anomaly_scores.get(node_id, 0),
                    "weighted": self.config.anomaly_weight * anomaly_scores.get(node_id, 0),
                    "explanation": "Based on metric anomalies (error rate, latency, etc.)"
                },
                "topology": {
                    "weight": self.config.topology_weight,
                    "score": topology_scores.get(node_id, 0),
                    "weighted": self.config.topology_weight * topology_scores.get(node_id, 0),
                    "explanation": "Based on graph structure (upstream/downstream edges)"
                },
                "propagation": {
                    "weight": self.config.propagation_weight,
                    "score": propagation_scores.get(node_id, 0),
                    "weighted": self.config.propagation_weight * propagation_scores.get(node_id, 0),
                    "explanation": "Based on blame propagation from affected nodes"
                },
                "temporal": {
                    "weight": self.config.temporal_weight,
                    "score": temporal_scores.get(node_id, 0),
                    "weighted": self.config.temporal_weight * temporal_scores.get(node_id, 0),
                    "explanation": "Based on when anomaly first appeared (earlier = higher)"
                }
            },
            "graph_position": {
                "incoming_edges": len(graph.get_incoming_edges(node_id)),
                "outgoing_edges": len(graph.get_outgoing_edges(node_id)),
                "upstream_nodes": [n.name for n in graph.get_upstream_nodes(node_id)],
                "downstream_nodes": [n.name for n in graph.get_downstream_nodes(node_id)],
            }
        }
        
        return breakdown


def compute_root_cause_scores(
    signals: Dict[str, Any],
    incident_id: Optional[str] = None,
    config: Optional[ScoringConfig] = None
) -> Tuple[CausalGraph, List[Dict[str, Any]]]:
    """
    Convenience function to build graph and compute scores in one step.
    
    This function can be directly consumed by RootCauseAgent.
    
    Args:
        signals: Incident signals dictionary
        incident_id: Optional incident ID
        config: Optional scoring configuration
        
    Returns:
        Tuple of (CausalGraph, root_cause_summaries)
        
    Example:
        >>> signals = {"type": "error_rate", "error_rate": 0.15, ...}
        >>> graph, causes = compute_root_cause_scores(signals)
        >>> print(causes[0]["name"], causes[0]["confidence"])
        "Database" "high"
    """
    from app.services.causal_graph.graph_builder import CausalGraphBuilder
    
    # Build graph
    builder = CausalGraphBuilder()
    graph = builder.build_from_signals(
        signals=signals,
        incident_id=incident_id
    )
    
    # Score nodes
    scorer = RootCauseScorer(config=config)
    scorer.compute_scores(graph)
    
    # Get summaries
    summaries = scorer.get_root_cause_summary(graph)
    
    return graph, summaries

