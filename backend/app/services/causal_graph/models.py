"""
Data models for the Causal Graph system.

Defines nodes (services, databases, caches, external dependencies)
and edges (dependencies, correlations) for root cause analysis.

All scores (anomaly_score, root_cause_score, confidence) are normalized to [0, 1].
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class NodeType(str, Enum):
    """Types of nodes in the causal graph."""
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_DEPENDENCY = "external_dependency"
    MESSAGE_QUEUE = "message_queue"
    LOAD_BALANCER = "load_balancer"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    """Types of edges representing relationships between nodes."""
    DEPENDENCY = "dependency"          # A depends on B
    CORRELATION = "correlation"        # Metrics correlate during incident
    LOG_COOCCURRENCE = "log_cooccurrence"  # Errors appear together in logs
    TEMPORAL = "temporal"              # Temporal causation (A precedes B)
    INFERRED = "inferred"              # Inferred from signals


@dataclass
class MetricTimeSeries:
    """Time series data for a metric."""
    name: str
    values: List[float]
    timestamps: List[datetime]
    
    def __len__(self) -> int:
        return len(self.values)
    
    def to_list(self) -> List[float]:
        """Return values as a list for correlation calculations."""
        return self.values


@dataclass
class NodeMetrics:
    """Metrics associated with a node during an incident window."""
    error_rate: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    request_count: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Time series data for correlation analysis
    error_rate_series: Optional[MetricTimeSeries] = None
    latency_series: Optional[MetricTimeSeries] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in {
            "error_rate": self.error_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "request_count": self.request_count,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
        }.items() if v is not None}


@dataclass
class NodeLogs:
    """Log information associated with a node."""
    error_count: int = 0
    warning_count: int = 0
    error_messages: List[str] = field(default_factory=list)
    first_error_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    
    # Keywords extracted from logs
    keywords: List[str] = field(default_factory=list)
    
    def add_error(self, message: str, timestamp: datetime) -> None:
        """Add an error log entry."""
        self.error_count += 1
        self.error_messages.append(message)
        
        if self.first_error_time is None or timestamp < self.first_error_time:
            self.first_error_time = timestamp
        if self.last_error_time is None or timestamp > self.last_error_time:
            self.last_error_time = timestamp


@dataclass
class CausalNode:
    """
    A node in the causal graph representing a system component.
    
    Nodes can be services, databases, caches, or external dependencies.
    Each node tracks its metrics and logs during an incident window.
    """
    id: str
    name: str
    node_type: NodeType
    
    # Observability data
    metrics: NodeMetrics = field(default_factory=NodeMetrics)
    logs: NodeLogs = field(default_factory=NodeLogs)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed scores (populated by scorer)
    anomaly_score: float = 0.0
    root_cause_score: float = 0.0
    
    # Temporal information
    first_anomaly_time: Optional[datetime] = None
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, CausalNode):
            return self.id == other.id
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "metrics": self.metrics.to_dict(),
            "logs": {
                "error_count": self.logs.error_count,
                "warning_count": self.logs.warning_count,
                "first_error_time": self.logs.first_error_time.isoformat() if self.logs.first_error_time else None,
            },
            "anomaly_score": self.anomaly_score,
            "root_cause_score": self.root_cause_score,
            "first_anomaly_time": self.first_anomaly_time.isoformat() if self.first_anomaly_time else None,
            "metadata": self.metadata,
        }


@dataclass
class EdgeWeight:
    """
    Composite weight for a causal edge.
    
    The final weight is computed from multiple factors:
    - metric_correlation: Pearson/Spearman correlation between node metrics
    - log_cooccurrence: Frequency of log errors appearing together
    - temporal_weight: Earlier events weighted higher (causation direction)
    """
    metric_correlation: float = 0.0      # -1 to 1, typically we use abs()
    log_cooccurrence: float = 0.0        # 0 to 1, normalized frequency
    temporal_weight: float = 0.0         # 0 to 1, based on time ordering
    
    # Individual correlation scores for debugging
    pearson: Optional[float] = None
    spearman: Optional[float] = None
    
    def compute_final_weight(
        self,
        metric_weight: float = 0.4,
        log_weight: float = 0.3,
        temporal_weight_factor: float = 0.3
    ) -> float:
        """
        Compute the final edge weight as a weighted combination.
        
        Args:
            metric_weight: Weight for metric correlation (default 0.4)
            log_weight: Weight for log co-occurrence (default 0.3)
            temporal_weight_factor: Weight for temporal alignment (default 0.3)
            
        Returns:
            Combined weight between 0 and 1
        """
        # Use absolute correlation (both positive and negative are meaningful)
        metric_score = abs(self.metric_correlation)
        
        final = (
            metric_weight * metric_score +
            log_weight * self.log_cooccurrence +
            temporal_weight_factor * self.temporal_weight
        )
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, final))
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "metric_correlation": self.metric_correlation,
            "log_cooccurrence": self.log_cooccurrence,
            "temporal_weight": self.temporal_weight,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "final_weight": self.compute_final_weight(),
        }


@dataclass
class CausalEdge:
    """
    An edge in the causal graph representing a relationship between nodes.
    
    Edges can represent:
    - Direct dependencies (A calls B)
    - Metric correlations during an incident
    - Log co-occurrence patterns
    - Temporal causation (A's anomaly preceded B's)
    """
    id: str
    source_id: str       # Source node ID
    target_id: str       # Target node ID
    edge_type: EdgeType
    weight: EdgeWeight = field(default_factory=EdgeWeight)
    
    # Direction of causality: True if source is likely cause of target
    is_causal: bool = True
    
    # Confidence in this edge (0 to 1)
    confidence: float = 0.5
    
    # Evidence supporting this edge
    evidence: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, CausalEdge):
            return self.id == other.id
        return False
    
    @property
    def final_weight(self) -> float:
        """Get the computed final weight."""
        return self.weight.compute_final_weight()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight.to_dict(),
            "is_causal": self.is_causal,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class CausalGraph:
    """
    A directed graph of causal relationships for root cause analysis.
    
    The graph contains nodes (system components) and edges (relationships),
    with methods for traversal and root cause computation.
    """
    incident_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: Dict[str, CausalEdge] = field(default_factory=dict)
    
    # Adjacency lists for efficient traversal
    _outgoing: Dict[str, List[str]] = field(default_factory=dict)  # node_id -> [edge_ids]
    _incoming: Dict[str, List[str]] = field(default_factory=dict)  # node_id -> [edge_ids]
    
    def add_node(self, node: CausalNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self._outgoing:
            self._outgoing[node.id] = []
        if node.id not in self._incoming:
            self._incoming[node.id] = []
    
    def add_edge(self, edge: CausalEdge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.id] = edge
        
        # Update adjacency lists
        if edge.source_id not in self._outgoing:
            self._outgoing[edge.source_id] = []
        self._outgoing[edge.source_id].append(edge.id)
        
        if edge.target_id not in self._incoming:
            self._incoming[edge.target_id] = []
        self._incoming[edge.target_id].append(edge.id)
    
    def get_node(self, node_id: str) -> Optional[CausalNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[CausalEdge]:
        """Get an edge by ID."""
        return self.edges.get(edge_id)
    
    def get_outgoing_edges(self, node_id: str) -> List[CausalEdge]:
        """Get all edges originating from a node."""
        edge_ids = self._outgoing.get(node_id, [])
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]
    
    def get_incoming_edges(self, node_id: str) -> List[CausalEdge]:
        """Get all edges pointing to a node."""
        edge_ids = self._incoming.get(node_id, [])
        return [self.edges[eid] for eid in edge_ids if eid in self.edges]
    
    def get_upstream_nodes(self, node_id: str) -> List[CausalNode]:
        """Get nodes that are upstream (potential causes) of this node."""
        incoming = self.get_incoming_edges(node_id)
        return [self.nodes[e.source_id] for e in incoming if e.source_id in self.nodes]
    
    def get_downstream_nodes(self, node_id: str) -> List[CausalNode]:
        """Get nodes that are downstream (potential effects) of this node."""
        outgoing = self.get_outgoing_edges(node_id)
        return [self.nodes[e.target_id] for e in outgoing if e.target_id in self.nodes]
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[CausalNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]
    
    def get_root_candidates(self) -> List[CausalNode]:
        """
        Get potential root cause candidates.
        
        These are nodes with:
        - High anomaly scores
        - Early first_anomaly_time
        - Few or no incoming causal edges
        """
        candidates = []
        for node in self.nodes.values():
            incoming = self.get_incoming_edges(node.id)
            # Nodes with anomalies and limited incoming edges are candidates
            if node.anomaly_score > 0 or node.logs.error_count > 0:
                candidates.append(node)
        
        # Sort by anomaly time (earliest first), then by anomaly score (highest first)
        def sort_key(n: CausalNode):
            time_key = n.first_anomaly_time.timestamp() if n.first_anomaly_time else float('inf')
            return (time_key, -n.anomaly_score)
        
        return sorted(candidates, key=sort_key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "incident_id": self.incident_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": {eid: e.to_dict() for eid, e in self.edges.items()},
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }
    
    def get_ranked_root_causes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top N ranked root cause candidates.
        
        Returns:
            List of dicts with node info and scores, sorted by root_cause_score
        """
        ranked = sorted(
            self.nodes.values(),
            key=lambda n: n.root_cause_score,
            reverse=True
        )
        
        return [
            {
                "node_id": n.id,
                "name": n.name,
                "node_type": n.node_type.value,
                "root_cause_score": n.root_cause_score,
                "anomaly_score": n.anomaly_score,
                "first_anomaly_time": n.first_anomaly_time.isoformat() if n.first_anomaly_time else None,
                "error_count": n.logs.error_count,
                "upstream_count": len(self.get_incoming_edges(n.id)),
                "downstream_count": len(self.get_outgoing_edges(n.id)),
            }
            for n in ranked[:top_n]
        ]

