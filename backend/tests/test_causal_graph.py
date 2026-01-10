"""
Tests for the Causal Graph system.

Tests cover:
- Data models (CausalNode, CausalEdge, CausalGraph)
- Correlation calculations (Pearson, Spearman, log co-occurrence)
- Graph building from signals
- Root cause scoring algorithm
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.services.causal_graph.models import (
    CausalNode,
    CausalEdge,
    CausalGraph,
    NodeType,
    EdgeType,
    NodeMetrics,
    NodeLogs,
    MetricTimeSeries,
    EdgeWeight,
)
from app.services.causal_graph.correlations import (
    pearson_correlation,
    spearman_correlation,
    log_cooccurrence_score,
    temporal_alignment_score,
    keyword_overlap_score,
    CorrelationCalculator,
)
from app.services.causal_graph.graph_builder import CausalGraphBuilder
from app.services.causal_graph.scorer import (
    RootCauseScorer,
    ScoringConfig,
    compute_root_cause_scores,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_node():
    """Create a sample causal node."""
    return CausalNode(
        id="test_node_1",
        name="Test Service",
        node_type=NodeType.SERVICE,
        metrics=NodeMetrics(
            error_rate=0.15,
            latency_p95_ms=500,
            memory_usage_mb=256
        )
    )


@pytest.fixture
def sample_graph():
    """Create a sample causal graph with nodes and edges."""
    graph = CausalGraph(
        incident_id="inc_001",
        start_time=datetime.utcnow() - timedelta(minutes=10),
        end_time=datetime.utcnow()
    )
    
    # Add nodes
    app_node = CausalNode(
        id="app",
        name="Application",
        node_type=NodeType.SERVICE,
        metrics=NodeMetrics(error_rate=0.2, latency_p95_ms=1500),
        anomaly_score=0.8,
        first_anomaly_time=datetime.utcnow() - timedelta(minutes=5)
    )
    
    db_node = CausalNode(
        id="db",
        name="Database",
        node_type=NodeType.DATABASE,
        metrics=NodeMetrics(latency_p95_ms=2000),
        anomaly_score=0.9,
        first_anomaly_time=datetime.utcnow() - timedelta(minutes=8)  # Earlier
    )
    
    cache_node = CausalNode(
        id="cache",
        name="Cache",
        node_type=NodeType.CACHE,
        metrics=NodeMetrics(memory_usage_mb=600),
        anomaly_score=0.3,
        first_anomaly_time=datetime.utcnow() - timedelta(minutes=2)  # Later
    )
    
    graph.add_node(app_node)
    graph.add_node(db_node)
    graph.add_node(cache_node)
    
    # Add edges
    edge1 = CausalEdge(
        id="edge_app_db",
        source_id="app",
        target_id="db",
        edge_type=EdgeType.DEPENDENCY,
        weight=EdgeWeight(metric_correlation=0.85, temporal_weight=0.6),
        confidence=0.9
    )
    
    edge2 = CausalEdge(
        id="edge_app_cache",
        source_id="app",
        target_id="cache",
        edge_type=EdgeType.DEPENDENCY,
        weight=EdgeWeight(metric_correlation=0.4),
        confidence=0.7
    )
    
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    
    return graph


@pytest.fixture
def sample_signals():
    """Create sample incident signals."""
    return {
        "type": "error_rate",
        "error_rate": 0.15,
        "latency_p95_ms": 1200,
        "memory_usage_mb": 450,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics_snapshot": {
            "error_rate": 0.15,
            "latency_p95_ms": 1200,
            "memory_usage_mb": 450,
            "request_count": 5000
        },
        "logs": [
            {"level": "error", "message": "Database connection timeout", "timestamp": datetime.utcnow().isoformat()},
            {"level": "error", "message": "HTTP 500 Internal Server Error", "timestamp": datetime.utcnow().isoformat()},
            {"level": "warning", "message": "High memory usage detected", "timestamp": datetime.utcnow().isoformat()},
        ]
    }


# =============================================================================
# Test Models
# =============================================================================

class TestCausalNode:
    """Tests for CausalNode model."""
    
    def test_create_node(self, sample_node):
        """Test creating a basic node."""
        assert sample_node.id == "test_node_1"
        assert sample_node.name == "Test Service"
        assert sample_node.node_type == NodeType.SERVICE
        assert sample_node.metrics.error_rate == 0.15
    
    def test_node_to_dict(self, sample_node):
        """Test node serialization."""
        data = sample_node.to_dict()
        assert data["id"] == "test_node_1"
        assert data["node_type"] == "service"
        assert "metrics" in data
    
    def test_node_equality(self):
        """Test node equality based on ID."""
        node1 = CausalNode(id="same_id", name="Node 1", node_type=NodeType.SERVICE)
        node2 = CausalNode(id="same_id", name="Node 2", node_type=NodeType.DATABASE)
        node3 = CausalNode(id="different_id", name="Node 3", node_type=NodeType.SERVICE)
        
        assert node1 == node2  # Same ID
        assert node1 != node3  # Different ID
    
    def test_node_logs(self):
        """Test NodeLogs functionality."""
        logs = NodeLogs()
        now = datetime.utcnow()
        
        logs.add_error("Error 1", now - timedelta(minutes=5))
        logs.add_error("Error 2", now)
        
        assert logs.error_count == 2
        assert logs.first_error_time == now - timedelta(minutes=5)
        assert logs.last_error_time == now


class TestCausalEdge:
    """Tests for CausalEdge model."""
    
    def test_create_edge(self):
        """Test creating an edge."""
        edge = CausalEdge(
            id="edge_1",
            source_id="node_a",
            target_id="node_b",
            edge_type=EdgeType.DEPENDENCY
        )
        assert edge.source_id == "node_a"
        assert edge.target_id == "node_b"
    
    def test_edge_weight_computation(self):
        """Test EdgeWeight.compute_final_weight()."""
        weight = EdgeWeight(
            metric_correlation=0.8,
            log_cooccurrence=0.6,
            temporal_weight=0.4
        )
        
        # Default weights: 0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.4 = 0.62
        final = weight.compute_final_weight()
        assert 0.61 < final < 0.63
    
    def test_edge_weight_clamping(self):
        """Test that edge weight is clamped to [0, 1]."""
        weight = EdgeWeight(
            metric_correlation=1.0,
            log_cooccurrence=1.0,
            temporal_weight=1.0
        )
        assert weight.compute_final_weight() <= 1.0


class TestCausalGraph:
    """Tests for CausalGraph model."""
    
    def test_add_and_get_nodes(self, sample_graph):
        """Test adding and retrieving nodes."""
        assert len(sample_graph.nodes) == 3
        assert sample_graph.get_node("app") is not None
        assert sample_graph.get_node("nonexistent") is None
    
    def test_add_and_get_edges(self, sample_graph):
        """Test adding and retrieving edges."""
        assert len(sample_graph.edges) == 2
        assert sample_graph.get_edge("edge_app_db") is not None
    
    def test_get_outgoing_edges(self, sample_graph):
        """Test getting outgoing edges from a node."""
        outgoing = sample_graph.get_outgoing_edges("app")
        assert len(outgoing) == 2
    
    def test_get_incoming_edges(self, sample_graph):
        """Test getting incoming edges to a node."""
        incoming = sample_graph.get_incoming_edges("db")
        assert len(incoming) == 1
        assert incoming[0].source_id == "app"
    
    def test_get_upstream_downstream_nodes(self, sample_graph):
        """Test getting upstream and downstream nodes."""
        upstream = sample_graph.get_upstream_nodes("db")
        downstream = sample_graph.get_downstream_nodes("app")
        
        assert len(upstream) == 1
        assert upstream[0].id == "app"
        assert len(downstream) == 2
    
    def test_get_nodes_by_type(self, sample_graph):
        """Test filtering nodes by type."""
        databases = sample_graph.get_nodes_by_type(NodeType.DATABASE)
        assert len(databases) == 1
        assert databases[0].id == "db"
    
    def test_graph_to_dict(self, sample_graph):
        """Test graph serialization."""
        data = sample_graph.to_dict()
        assert data["node_count"] == 3
        assert data["edge_count"] == 2
        assert "nodes" in data
        assert "edges" in data


# =============================================================================
# Test Correlations
# =============================================================================

class TestCorrelations:
    """Tests for correlation calculations."""
    
    def test_pearson_perfect_positive(self):
        """Test Pearson correlation with perfect positive correlation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        corr = pearson_correlation(x, y)
        assert corr is not None
        assert abs(corr - 1.0) < 0.001
    
    def test_pearson_perfect_negative(self):
        """Test Pearson correlation with perfect negative correlation."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        corr = pearson_correlation(x, y)
        assert corr is not None
        assert abs(corr + 1.0) < 0.001
    
    def test_pearson_no_correlation(self):
        """Test Pearson correlation with no correlation."""
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 1, 5]  # Random-ish
        corr = pearson_correlation(x, y)
        assert corr is not None
        assert abs(corr) < 0.5  # Weak correlation
    
    def test_pearson_insufficient_data(self):
        """Test Pearson with insufficient data."""
        assert pearson_correlation([1, 2], [2, 4]) is None  # Less than 3 points
        assert pearson_correlation([], []) is None
    
    def test_spearman_monotonic(self):
        """Test Spearman correlation with monotonic relationship."""
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 9, 16, 25]  # Quadratic but monotonic
        corr = spearman_correlation(x, y)
        assert corr is not None
        assert abs(corr - 1.0) < 0.001  # Perfect monotonic
    
    def test_spearman_with_ties(self):
        """Test Spearman correlation handles ties correctly."""
        x = [1, 1, 2, 3, 3]
        y = [1, 2, 2, 4, 4]
        corr = spearman_correlation(x, y)
        assert corr is not None
        assert corr > 0.5  # Still positive correlation
    
    def test_log_cooccurrence_score(self):
        """Test log co-occurrence scoring."""
        now = datetime.utcnow()
        
        logs_a = [
            {"level": "error", "message": "Error A", "timestamp": now.isoformat()},
            {"level": "error", "message": "Error A2", "timestamp": (now + timedelta(seconds=30)).isoformat()},
        ]
        logs_b = [
            {"level": "error", "message": "Error B", "timestamp": (now + timedelta(seconds=10)).isoformat()},
        ]
        
        score = log_cooccurrence_score(logs_a, logs_b, time_window_seconds=60)
        assert 0 < score <= 1.0
    
    def test_log_cooccurrence_no_overlap(self):
        """Test log co-occurrence with no temporal overlap."""
        now = datetime.utcnow()
        
        logs_a = [
            {"level": "error", "message": "Error A", "timestamp": now.isoformat()},
        ]
        logs_b = [
            {"level": "error", "message": "Error B", "timestamp": (now + timedelta(minutes=10)).isoformat()},
        ]
        
        score = log_cooccurrence_score(logs_a, logs_b, time_window_seconds=60)
        assert score == 0.0
    
    def test_temporal_alignment_score(self):
        """Test temporal alignment scoring."""
        now = datetime.utcnow()
        earlier = now - timedelta(minutes=2)
        
        score, a_precedes = temporal_alignment_score(earlier, now, max_lag_seconds=300)
        
        assert a_precedes is True
        assert 0 < score < 1.0
    
    def test_temporal_alignment_simultaneous(self):
        """Test temporal alignment for nearly simultaneous events."""
        now = datetime.utcnow()
        
        score, _ = temporal_alignment_score(now, now + timedelta(seconds=2))
        assert score > 0.9  # Very close in time
    
    def test_keyword_overlap(self):
        """Test keyword overlap scoring."""
        keywords_a = ["database", "timeout", "connection"]
        keywords_b = ["timeout", "connection", "refused"]
        
        score = keyword_overlap_score(keywords_a, keywords_b)
        assert 0.3 < score < 0.7  # Partial overlap


class TestCorrelationCalculator:
    """Tests for CorrelationCalculator class."""
    
    def test_compute_edge_weight(self):
        """Test computing edge weight between nodes."""
        calculator = CorrelationCalculator()
        
        node_a = CausalNode(
            id="a", name="A", node_type=NodeType.SERVICE,
            first_anomaly_time=datetime.utcnow() - timedelta(minutes=5)
        )
        node_b = CausalNode(
            id="b", name="B", node_type=NodeType.DATABASE,
            first_anomaly_time=datetime.utcnow() - timedelta(minutes=2)
        )
        
        weight = calculator.compute_edge_weight(node_a, node_b)
        
        assert isinstance(weight, EdgeWeight)
        assert weight.temporal_weight > 0  # Should have temporal component
    
    def test_compute_causality_direction(self):
        """Test determining causality direction."""
        calculator = CorrelationCalculator()
        
        early = CausalNode(
            id="early", name="Early", node_type=NodeType.DATABASE,
            first_anomaly_time=datetime.utcnow() - timedelta(minutes=10)
        )
        late = CausalNode(
            id="late", name="Late", node_type=NodeType.SERVICE,
            first_anomaly_time=datetime.utcnow() - timedelta(minutes=2)
        )
        
        a_causes_b, confidence = calculator.compute_causality_direction(early, late)
        
        assert a_causes_b is True  # Early anomaly is likely cause
        assert confidence > 0.5


# =============================================================================
# Test Graph Builder
# =============================================================================

class TestCausalGraphBuilder:
    """Tests for CausalGraphBuilder."""
    
    def test_build_from_signals(self, sample_signals):
        """Test building a graph from signals."""
        builder = CausalGraphBuilder()
        graph = builder.build_from_signals(sample_signals, incident_id="test_inc")
        
        assert len(graph.nodes) >= 1  # At least application node
        assert graph.incident_id == "test_inc"
    
    def test_discovers_database_node(self):
        """Test that database-related signals create a database node."""
        signals = {
            "type": "db_slow",
            "query_latency_ms": 5000,
            "logs": [
                {"level": "error", "message": "Database query timeout", "timestamp": datetime.utcnow().isoformat()},
            ]
        }
        
        builder = CausalGraphBuilder()
        graph = builder.build_from_signals(signals)
        
        db_nodes = graph.get_nodes_by_type(NodeType.DATABASE)
        assert len(db_nodes) >= 1
    
    def test_discovers_dependency_node(self):
        """Test that dependency-related signals create an external node."""
        signals = {
            "type": "dependency_down",
            "logs": [
                {"level": "error", "message": "External API connection refused", "timestamp": datetime.utcnow().isoformat()},
            ]
        }
        
        builder = CausalGraphBuilder()
        graph = builder.build_from_signals(signals)
        
        ext_nodes = graph.get_nodes_by_type(NodeType.EXTERNAL_DEPENDENCY)
        assert len(ext_nodes) >= 1
    
    def test_creates_dependency_edges(self, sample_signals):
        """Test that dependency edges are created."""
        builder = CausalGraphBuilder()
        graph = builder.build_from_signals(sample_signals)
        
        # Should have at least some edges if multiple nodes exist
        if len(graph.nodes) > 1:
            assert len(graph.edges) >= 1
    
    def test_infer_edges_disabled(self, sample_signals):
        """Test building without edge inference."""
        builder = CausalGraphBuilder(infer_edges=False)
        graph = builder.build_from_signals(sample_signals)
        
        # Should still have known dependency edges
        assert isinstance(graph, CausalGraph)
    
    def test_add_custom_node(self, sample_signals):
        """Test adding a custom node to a graph."""
        builder = CausalGraphBuilder()
        graph = builder.build_from_signals(sample_signals)
        
        custom = builder.add_custom_node(
            graph,
            name="Custom Service",
            node_type=NodeType.SERVICE,
            metrics={"error_rate": 0.05}
        )
        
        assert custom.id in graph.nodes
        assert custom.name == "Custom Service"


# =============================================================================
# Test Scorer
# =============================================================================

class TestRootCauseScorer:
    """Tests for RootCauseScorer."""
    
    def test_compute_scores(self, sample_graph):
        """Test computing root cause scores."""
        scorer = RootCauseScorer()
        scores = scorer.compute_scores(sample_graph)
        
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores.values())
    
    def test_scores_reflect_topology(self, sample_graph):
        """Test that scores reflect graph topology."""
        scorer = RootCauseScorer()
        scorer.compute_scores(sample_graph)
        
        # Database node has highest anomaly and earliest time
        # It should have a high score
        db_node = sample_graph.get_node("db")
        assert db_node.root_cause_score > 0.3
    
    def test_temporal_scoring(self, sample_graph):
        """Test that earlier anomalies get higher temporal scores."""
        config = ScoringConfig(temporal_weight=1.0, anomaly_weight=0, topology_weight=0, propagation_weight=0)
        scorer = RootCauseScorer(config=config)
        scores = scorer.compute_scores(sample_graph)
        
        # Database anomaly was earliest, cache was latest
        assert scores["db"] > scores["cache"]
    
    def test_get_root_cause_summary(self, sample_graph):
        """Test getting root cause summary."""
        scorer = RootCauseScorer()
        scorer.compute_scores(sample_graph)
        
        summary = scorer.get_root_cause_summary(sample_graph, top_n=2)
        
        assert len(summary) == 2
        assert "node_id" in summary[0]
        assert "root_cause_score" in summary[0]
        assert "confidence" in summary[0]
    
    def test_explain_score(self, sample_graph):
        """Test score explanation."""
        scorer = RootCauseScorer()
        scorer.compute_scores(sample_graph)
        
        explanation = scorer.explain_score(sample_graph, "db")
        
        assert "components" in explanation
        assert "anomaly" in explanation["components"]
        assert "topology" in explanation["components"]
    
    def test_empty_graph(self):
        """Test scorer handles empty graph."""
        scorer = RootCauseScorer()
        graph = CausalGraph()
        scores = scorer.compute_scores(graph)
        
        assert scores == {}


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_end_to_end(self, sample_signals):
        """Test the complete pipeline from signals to root causes."""
        graph, summaries = compute_root_cause_scores(
            signals=sample_signals,
            incident_id="int_test_001"
        )
        
        assert isinstance(graph, CausalGraph)
        assert len(graph.nodes) >= 1
        assert len(summaries) >= 1
        assert summaries[0]["root_cause_score"] > 0
    
    def test_different_signal_types(self):
        """Test handling different signal types."""
        signal_types = [
            {"type": "error_rate", "error_rate": 0.25},
            {"type": "latency_spike", "latency_p95_ms": 3000},
            {"type": "memory_leak", "memory_usage_mb": 800},
            {"type": "dependency_down", "logs": [{"level": "error", "message": "Connection refused"}]},
        ]
        
        for signals in signal_types:
            graph, summaries = compute_root_cause_scores(signals)
            assert len(graph.nodes) >= 1
            assert len(summaries) >= 1
    
    def test_graph_serialization(self, sample_signals):
        """Test that the graph can be serialized to dict."""
        graph, _ = compute_root_cause_scores(sample_signals)
        data = graph.to_dict()
        
        assert "nodes" in data
        assert "edges" in data
        assert "node_count" in data

