"""
Causal Graph Builder.

Constructs a causal graph from incident signals, metrics, and logs.
Automatically discovers nodes (components) and infers edges (relationships)
based on correlations and temporal patterns.

Determinism Note:
- Node IDs for discovered components use deterministic naming (e.g., "node_database")
- Custom nodes use UUIDs for uniqueness, which are non-deterministic between runs
- For reproducible results, use fixed node IDs when adding custom nodes
"""

import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

from app.services.causal_graph.models import (
    CausalGraph,
    CausalNode,
    CausalEdge,
    NodeType,
    EdgeType,
    NodeMetrics,
    NodeLogs,
    MetricTimeSeries,
    EdgeWeight,
)
from app.services.causal_graph.correlations import (
    CorrelationCalculator,
    temporal_alignment_score,
)


# Common component patterns for node discovery
COMPONENT_PATTERNS = {
    NodeType.DATABASE: [
        r"database", r"db", r"postgres", r"mysql", r"redis",
        r"mongo", r"sql", r"query", r"connection.*pool"
    ],
    NodeType.CACHE: [
        r"cache", r"redis", r"memcached", r"cdn"
    ],
    NodeType.EXTERNAL_DEPENDENCY: [
        r"external", r"api", r"third.?party", r"upstream",
        r"downstream", r"dependency", r"service"
    ],
    NodeType.MESSAGE_QUEUE: [
        r"queue", r"kafka", r"rabbitmq", r"sqs", r"pubsub"
    ],
    NodeType.LOAD_BALANCER: [
        r"load.?balancer", r"lb", r"nginx", r"haproxy", r"ingress"
    ],
}


class CausalGraphBuilder:
    """
    Builds a causal graph from incident signals.
    
    The builder:
    1. Discovers nodes from metrics and log patterns
    2. Creates edges based on known dependencies
    3. Infers additional edges from correlations
    4. Assigns edge weights based on multiple factors
    """
    
    def __init__(
        self,
        correlation_calculator: Optional[CorrelationCalculator] = None,
        min_edge_weight: float = 0.1,
        infer_edges: bool = True
    ):
        """
        Initialize the graph builder.
        
        Args:
            correlation_calculator: Calculator for edge weights
            min_edge_weight: Minimum weight to include an edge
            infer_edges: Whether to infer edges from correlations
        """
        self.calculator = correlation_calculator or CorrelationCalculator()
        self.min_edge_weight = min_edge_weight
        self.infer_edges = infer_edges
    
    def build_from_signals(
        self,
        signals: Dict[str, Any],
        incident_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> CausalGraph:
        """
        Build a causal graph from incident signals.
        
        Args:
            signals: Dictionary containing metrics, logs, and other signals
            incident_id: Optional incident ID
            start_time: Incident start time
            end_time: Incident end time (or current time if ongoing)
            
        Returns:
            Constructed CausalGraph
        """
        graph = CausalGraph(
            incident_id=incident_id,
            start_time=start_time,
            end_time=end_time or datetime.utcnow()
        )
        
        # Extract components from signals
        metrics_snapshot = signals.get("metrics_snapshot", {})
        logs = signals.get("logs", [])
        
        # 1. Create nodes from discovered components
        nodes = self._discover_nodes(signals, metrics_snapshot, logs)
        for node in nodes:
            graph.add_node(node)
        
        # 2. Create edges from known dependencies
        known_edges = self._create_dependency_edges(signals, graph)
        for edge in known_edges:
            graph.add_edge(edge)
        
        # 3. Infer additional edges from correlations
        if self.infer_edges and len(graph.nodes) > 1:
            inferred_edges = self._infer_correlation_edges(graph, logs)
            for edge in inferred_edges:
                if edge.final_weight >= self.min_edge_weight:
                    graph.add_edge(edge)
        
        return graph
    
    def _discover_nodes(
        self,
        signals: Dict[str, Any],
        metrics: Dict[str, Any],
        logs: List[Dict[str, Any]]
    ) -> List[CausalNode]:
        """
        Discover nodes (components) from signals.
        
        Looks for:
        - Explicitly mentioned services
        - Components inferred from error patterns
        - Default application node
        """
        nodes = []
        discovered_names: Set[str] = set()
        
        # Always add the primary application node
        app_node = self._create_application_node(signals, metrics, logs)
        nodes.append(app_node)
        discovered_names.add("application")
        
        # Discover components from signal type
        signal_type = signals.get("type", "")
        if "database" in signal_type or "db" in signal_type:
            db_node = self._create_database_node(signals, metrics, logs)
            nodes.append(db_node)
            discovered_names.add("database")
        
        if "dependency" in signal_type or "external" in signal_type:
            ext_node = self._create_external_dependency_node(signals, logs)
            nodes.append(ext_node)
            discovered_names.add("external_dependency")
        
        if "memory" in signal_type or "cache" in signal_type:
            cache_node = self._create_cache_node(signals, metrics)
            nodes.append(cache_node)
            discovered_names.add("cache")
        
        # Discover components from log patterns
        # Note: We process node types in sorted order for deterministic results
        for log in logs:
            message = log.get("message", "").lower()
            
            for node_type in sorted(COMPONENT_PATTERNS.keys(), key=lambda x: x.value):
                patterns = COMPONENT_PATTERNS[node_type]
                for pattern in patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        component_name = f"{node_type.value}"
                        if component_name not in discovered_names:
                            # Use deterministic ID based on node type
                            node = CausalNode(
                                id=f"node_{node_type.value}",
                                name=component_name.replace("_", " ").title(),
                                node_type=node_type
                            )
                            self._populate_node_from_logs(node, logs, patterns)
                            nodes.append(node)
                            discovered_names.add(component_name)
                        break
        
        return nodes
    
    def _create_application_node(
        self,
        signals: Dict[str, Any],
        metrics: Dict[str, Any],
        logs: List[Dict[str, Any]]
    ) -> CausalNode:
        """Create the primary application node."""
        node = CausalNode(
            id="node_application",
            name="Application",
            node_type=NodeType.SERVICE
        )
        
        # Populate metrics
        node.metrics = NodeMetrics(
            error_rate=metrics.get("error_rate") or signals.get("error_rate"),
            latency_p50_ms=metrics.get("latency_p50_ms"),
            latency_p95_ms=metrics.get("latency_p95_ms") or signals.get("latency_p95_ms"),
            latency_p99_ms=metrics.get("latency_p99_ms"),
            request_count=metrics.get("request_count"),
            memory_usage_mb=metrics.get("memory_usage_mb"),
            cpu_usage_percent=metrics.get("cpu_usage_percent"),
        )
        
        # Calculate anomaly score based on metrics
        node.anomaly_score = self._calculate_anomaly_score(node.metrics, signals)
        
        # Populate logs
        self._populate_node_logs(node, logs)
        
        # Set first anomaly time
        if node.logs.first_error_time:
            node.first_anomaly_time = node.logs.first_error_time
        elif signals.get("timestamp"):
            try:
                node.first_anomaly_time = datetime.fromisoformat(
                    signals["timestamp"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                node.first_anomaly_time = datetime.utcnow()
        
        return node
    
    def _create_database_node(
        self,
        signals: Dict[str, Any],
        metrics: Dict[str, Any],
        logs: List[Dict[str, Any]]
    ) -> CausalNode:
        """Create a database node."""
        node = CausalNode(
            id="node_database",
            name="Database",
            node_type=NodeType.DATABASE
        )
        
        # Database-specific metrics
        node.metrics = NodeMetrics(
            latency_p95_ms=signals.get("query_latency_ms"),
        )
        
        # Look for database-related logs
        db_patterns = COMPONENT_PATTERNS[NodeType.DATABASE]
        self._populate_node_from_logs(node, logs, db_patterns)
        
        # High anomaly if this is a DB-related incident
        if "db" in signals.get("type", "").lower():
            node.anomaly_score = 0.8
        else:
            node.anomaly_score = 0.3
        
        return node
    
    def _create_external_dependency_node(
        self,
        signals: Dict[str, Any],
        logs: List[Dict[str, Any]]
    ) -> CausalNode:
        """Create an external dependency node."""
        node = CausalNode(
            id="node_external",
            name="External Service",
            node_type=NodeType.EXTERNAL_DEPENDENCY
        )
        
        # Look for dependency-related logs
        dep_patterns = COMPONENT_PATTERNS[NodeType.EXTERNAL_DEPENDENCY]
        self._populate_node_from_logs(node, logs, dep_patterns)
        
        # High anomaly if this is a dependency-related incident
        if "dependency" in signals.get("type", "").lower():
            node.anomaly_score = 0.9
        else:
            node.anomaly_score = 0.2
        
        return node
    
    def _create_cache_node(
        self,
        signals: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> CausalNode:
        """Create a cache node."""
        node = CausalNode(
            id="node_cache",
            name="Cache",
            node_type=NodeType.CACHE
        )
        
        node.metrics = NodeMetrics(
            memory_usage_mb=metrics.get("memory_usage_mb"),
        )
        
        if "memory" in signals.get("type", "").lower():
            node.anomaly_score = 0.7
        else:
            node.anomaly_score = 0.2
        
        return node
    
    def _populate_node_logs(
        self,
        node: CausalNode,
        logs: List[Dict[str, Any]]
    ) -> None:
        """Populate a node with all logs."""
        for log in logs:
            level = log.get("level", "").lower()
            message = log.get("message", "")
            timestamp = log.get("timestamp")
            
            if level == "error":
                if timestamp:
                    if isinstance(timestamp, str):
                        try:
                            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        except ValueError:
                            ts = datetime.utcnow()
                    else:
                        ts = timestamp
                    node.logs.add_error(message, ts)
                else:
                    node.logs.error_count += 1
                    node.logs.error_messages.append(message)
            elif level == "warning":
                node.logs.warning_count += 1
            
            # Extract keywords
            keywords = self._extract_keywords(message)
            node.logs.keywords.extend(keywords)
    
    def _populate_node_from_logs(
        self,
        node: CausalNode,
        logs: List[Dict[str, Any]],
        patterns: List[str]
    ) -> None:
        """Populate a node with logs matching specific patterns."""
        for log in logs:
            message = log.get("message", "")
            
            # Check if log matches any pattern
            matches = any(
                re.search(pattern, message, re.IGNORECASE)
                for pattern in patterns
            )
            
            if matches:
                level = log.get("level", "").lower()
                timestamp = log.get("timestamp")
                
                if level == "error":
                    if timestamp:
                        if isinstance(timestamp, str):
                            try:
                                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            except ValueError:
                                ts = datetime.utcnow()
                        else:
                            ts = timestamp
                        node.logs.add_error(message, ts)
                    else:
                        node.logs.error_count += 1
                        node.logs.error_messages.append(message)
                elif level == "warning":
                    node.logs.warning_count += 1
                
                # Extract keywords
                keywords = self._extract_keywords(message)
                node.logs.keywords.extend(keywords)
        
        # Set first anomaly time from logs
        if node.logs.first_error_time:
            node.first_anomaly_time = node.logs.first_error_time
    
    def _extract_keywords(self, message: str) -> List[str]:
        """Extract meaningful keywords from a log message."""
        # Common error-related keywords
        keywords = []
        patterns = [
            r"timeout", r"connection", r"refused", r"error", r"failed",
            r"exception", r"null", r"undefined", r"unavailable", r"slow",
            r"memory", r"oom", r"disk", r"cpu", r"network", r"socket",
            r"database", r"query", r"transaction", r"deadlock", r"lock"
        ]
        
        message_lower = message.lower()
        for pattern in patterns:
            if re.search(pattern, message_lower):
                keywords.append(pattern)
        
        return keywords
    
    def _calculate_anomaly_score(
        self,
        metrics: NodeMetrics,
        signals: Dict[str, Any]
    ) -> float:
        """Calculate an anomaly score based on metrics."""
        score = 0.0
        factors = 0
        
        # Error rate (high = anomalous)
        if metrics.error_rate is not None:
            if metrics.error_rate > 0.1:
                score += 1.0
            elif metrics.error_rate > 0.05:
                score += 0.7
            elif metrics.error_rate > 0.01:
                score += 0.3
            factors += 1
        
        # Latency (high = anomalous)
        if metrics.latency_p95_ms is not None:
            if metrics.latency_p95_ms > 2000:
                score += 1.0
            elif metrics.latency_p95_ms > 1000:
                score += 0.7
            elif metrics.latency_p95_ms > 500:
                score += 0.4
            factors += 1
        
        # Memory (high = potentially anomalous)
        if metrics.memory_usage_mb is not None:
            if metrics.memory_usage_mb > 800:
                score += 0.8
            elif metrics.memory_usage_mb > 500:
                score += 0.5
            factors += 1
        
        # Signal type boost
        signal_type = signals.get("type", "")
        if "error" in signal_type:
            score += 0.5
            factors += 1
        
        return score / max(factors, 1)
    
    def _create_dependency_edges(
        self,
        signals: Dict[str, Any],
        graph: CausalGraph
    ) -> List[CausalEdge]:
        """Create edges based on known/inferred dependencies."""
        edges = []
        
        # Standard dependency patterns
        dependencies = [
            ("node_application", "node_database", "calls database"),
            ("node_application", "node_cache", "uses cache"),
            ("node_application", "node_external", "calls external service"),
        ]
        
        for source_id, target_id, description in dependencies:
            if source_id in graph.nodes and target_id in graph.nodes:
                edge = CausalEdge(
                    id=f"edge_{source_id}_{target_id}",
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=EdgeType.DEPENDENCY,
                    is_causal=True,
                    confidence=0.8,
                    evidence=[f"Application {description}"]
                )
                
                # Compute weight based on nodes
                source = graph.get_node(source_id)
                target = graph.get_node(target_id)
                if source and target:
                    edge.weight = self.calculator.compute_edge_weight(source, target)
                
                edges.append(edge)
        
        return edges
    
    def _infer_correlation_edges(
        self,
        graph: CausalGraph,
        logs: List[Dict[str, Any]]
    ) -> List[CausalEdge]:
        """
        Infer edges based on correlations between all node pairs.
        
        Determinism Note:
        -----------------
        Python 3.7+ guarantees dict insertion order, so graph.nodes.values()
        returns nodes in the order they were added. Since node discovery
        follows a deterministic pattern (application first, then signal-type
        based, then sorted log patterns), the iteration order here is stable.
        """
        edges = []
        node_list = list(graph.nodes.values())
        
        for i, node_a in enumerate(node_list):
            for node_b in node_list[i + 1:]:
                # Skip if edge already exists
                existing_edge_id = f"edge_{node_a.id}_{node_b.id}"
                reverse_edge_id = f"edge_{node_b.id}_{node_a.id}"
                if existing_edge_id in graph.edges or reverse_edge_id in graph.edges:
                    continue
                
                # Compute weight
                weight = self.calculator.compute_edge_weight(node_a, node_b, logs, logs)
                
                # Determine causality direction
                a_causes_b, confidence = self.calculator.compute_causality_direction(
                    node_a, node_b
                )
                
                # Only create edge if weight is significant
                if weight.compute_final_weight() >= self.min_edge_weight:
                    if a_causes_b:
                        source, target = node_a, node_b
                    else:
                        source, target = node_b, node_a
                    
                    edge = CausalEdge(
                        id=f"edge_{source.id}_{target.id}_inferred",
                        source_id=source.id,
                        target_id=target.id,
                        edge_type=EdgeType.CORRELATION,
                        weight=weight,
                        is_causal=True,
                        confidence=confidence,
                        evidence=[
                            f"Metric correlation: {weight.metric_correlation:.2f}",
                            f"Log co-occurrence: {weight.log_cooccurrence:.2f}",
                            f"Temporal alignment: {weight.temporal_weight:.2f}",
                        ]
                    )
                    edges.append(edge)
        
        return edges
    
    def add_custom_node(
        self,
        graph: CausalGraph,
        name: str,
        node_type: NodeType,
        metrics: Optional[Dict[str, Any]] = None,
        logs: Optional[List[Dict[str, Any]]] = None,
        node_id: Optional[str] = None
    ) -> CausalNode:
        """
        Add a custom node to an existing graph.
        
        Args:
            graph: The graph to add to
            name: Node name
            node_type: Type of node
            metrics: Optional metrics dictionary
            logs: Optional log entries
            node_id: Optional fixed node ID for deterministic behavior.
                     If not provided, generates a deterministic ID from name + type.
            
        Returns:
            The created node
        """
        # Generate deterministic ID from name and type if not provided
        if node_id is None:
            # Use hash for deterministic ID generation
            hash_input = f"{name}:{node_type.value}".encode()
            node_id = f"node_{hashlib.md5(hash_input).hexdigest()[:8]}"
        
        node = CausalNode(
            id=node_id,
            name=name,
            node_type=node_type
        )
        
        if metrics:
            node.metrics = NodeMetrics(
                error_rate=metrics.get("error_rate"),
                latency_p95_ms=metrics.get("latency_p95_ms"),
                memory_usage_mb=metrics.get("memory_usage_mb"),
            )
            node.anomaly_score = self._calculate_anomaly_score(node.metrics, {})
        
        if logs:
            self._populate_node_logs(node, logs)
        
        graph.add_node(node)
        return node
    
    def add_custom_edge(
        self,
        graph: CausalGraph,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DEPENDENCY,
        evidence: Optional[List[str]] = None,
        edge_id: Optional[str] = None
    ) -> Optional[CausalEdge]:
        """
        Add a custom edge to an existing graph.
        
        Args:
            graph: The graph to add to
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
            evidence: Optional evidence strings
            edge_id: Optional fixed edge ID for deterministic behavior.
                     If not provided, generates a deterministic ID from source + target + type.
            
        Returns:
            The created edge, or None if nodes don't exist
        """
        source = graph.get_node(source_id)
        target = graph.get_node(target_id)
        
        if not source or not target:
            return None
        
        # Generate deterministic ID from source, target, and type
        if edge_id is None:
            hash_input = f"{source_id}:{target_id}:{edge_type.value}".encode()
            edge_id = f"edge_{hashlib.md5(hash_input).hexdigest()[:8]}"
        
        edge = CausalEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            evidence=evidence or []
        )
        
        # Compute weight
        edge.weight = self.calculator.compute_edge_weight(source, target)
        
        graph.add_edge(edge)
        return edge

