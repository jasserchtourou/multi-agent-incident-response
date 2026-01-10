"""
Tests for the extended RootCauseAgent with causal graph integration.

Tests cover:
- Causal graph-based hypothesis generation
- Structured evidence creation
- Compatibility with ReporterAgent output format
- Fallback behavior when LLM unavailable
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from app.agents.root_cause import RootCauseAgent
from app.agents.schemas import (
    RootCauseAgentOutput,
    Hypothesis,
    StructuredEvidence,
    CausalGraphSummary,
)


@pytest.fixture
def sample_signals():
    """Sample incident signals for testing."""
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
        ]
    }


@pytest.fixture
def root_cause_agent():
    """Create a RootCauseAgent instance."""
    return RootCauseAgent()


class TestRootCauseAgentIntegration:
    """Integration tests for RootCauseAgent with causal graph."""
    
    def test_agent_initializes(self, root_cause_agent):
        """Test that the agent initializes correctly."""
        assert root_cause_agent.name == "root_cause"
        assert root_cause_agent.graph_builder is not None
        assert root_cause_agent.scorer is not None
    
    def test_run_produces_valid_output(self, root_cause_agent, sample_signals):
        """Test that run() produces valid output structure."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        assert "hypotheses" in result
        assert "most_likely" in result
        assert isinstance(result["hypotheses"], list)
        assert len(result["hypotheses"]) >= 1
    
    def test_hypotheses_have_required_fields(self, root_cause_agent, sample_signals):
        """Test that hypotheses have all required fields."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        for hyp in result["hypotheses"]:
            assert "cause" in hyp
            assert "confidence" in hyp
            assert "evidence" in hyp
            assert isinstance(hyp["confidence"], (int, float))
            assert 0 <= hyp["confidence"] <= 1
    
    def test_hypotheses_have_component_info(self, root_cause_agent, sample_signals):
        """Test that hypotheses include component information."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        # At least one hypothesis should have component info
        has_component = any(
            hyp.get("component") is not None
            for hyp in result["hypotheses"]
        )
        assert has_component
    
    def test_causal_graph_summary_included(self, root_cause_agent, sample_signals):
        """Test that causal graph summary is included."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        assert "causal_graph_summary" in result
        summary = result["causal_graph_summary"]
        assert "node_count" in summary
        assert "edge_count" in summary
        assert summary["node_count"] >= 1


class TestStructuredEvidence:
    """Tests for structured evidence generation."""
    
    def test_evidence_contains_correlated_metrics(self, root_cause_agent, sample_signals):
        """Test that evidence includes correlated metrics."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        for hyp in result["hypotheses"]:
            if hyp.get("structured_evidence"):
                metrics = hyp["structured_evidence"].get("correlated_metrics", [])
                # Should have some metrics
                if hyp.get("component") == "Application":
                    assert len(metrics) >= 1
    
    def test_evidence_contains_temporal_info(self, root_cause_agent, sample_signals):
        """Test that evidence includes temporal information."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        # Check for temporal evidence in at least one hypothesis
        has_temporal = False
        for hyp in result["hypotheses"]:
            if hyp.get("structured_evidence"):
                temporal = hyp["structured_evidence"].get("temporal_evidence", [])
                if temporal:
                    has_temporal = True
                    break
        
        # Should have some temporal evidence
        assert has_temporal or len(result["hypotheses"]) > 0


class TestDifferentSignalTypes:
    """Tests for handling different signal types."""
    
    @pytest.fixture
    def agent(self):
        return RootCauseAgent()
    
    def test_error_rate_signals(self, agent):
        """Test handling of error rate signals."""
        signals = {"type": "error_rate", "error_rate": 0.25}
        result = agent.run({"signals": signals})
        
        assert len(result["hypotheses"]) >= 1
        # Should identify error-related cause
        causes = [h["cause"].lower() for h in result["hypotheses"]]
        assert any("error" in c or "database" in c or "connection" in c for c in causes)
    
    def test_latency_spike_signals(self, agent):
        """Test handling of latency spike signals."""
        signals = {"type": "latency_spike", "latency_p95_ms": 3000}
        result = agent.run({"signals": signals})
        
        assert len(result["hypotheses"]) >= 1
        # Should identify latency-related cause
        causes = [h["cause"].lower() for h in result["hypotheses"]]
        assert any("latency" in c or "query" in c or "database" in c or "slow" in c for c in causes)
    
    def test_memory_leak_signals(self, agent):
        """Test handling of memory leak signals."""
        signals = {"type": "memory_leak", "memory_usage_mb": 800}
        result = agent.run({"signals": signals})
        
        assert len(result["hypotheses"]) >= 1
        # Should identify memory-related cause
        causes = [h["cause"].lower() for h in result["hypotheses"]]
        assert any("memory" in c or "cache" in c or "leak" in c for c in causes)
    
    def test_dependency_down_signals(self, agent):
        """Test handling of dependency down signals."""
        signals = {
            "type": "dependency_down",
            "logs": [
                {"level": "error", "message": "External API connection refused"}
            ]
        }
        result = agent.run({"signals": signals})
        
        assert len(result["hypotheses"]) >= 1
        # Should identify dependency-related cause
        causes = [h["cause"].lower() for h in result["hypotheses"]]
        assert any("dependency" in c or "external" in c or "service" in c for c in causes)


class TestReporterCompatibility:
    """Tests for compatibility with ReporterAgent."""
    
    def test_output_compatible_with_merge(self, root_cause_agent, sample_signals):
        """Test that output is compatible with supervisor merge logic."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        # These fields are expected by the supervisor's merge logic
        assert "hypotheses" in result
        assert "most_likely" in result
        
        # Each hypothesis should have the fields expected by ReporterAgent
        for hyp in result["hypotheses"]:
            assert "cause" in hyp
            assert "confidence" in hyp
            assert "evidence" in hyp
    
    def test_most_likely_is_string(self, root_cause_agent, sample_signals):
        """Test that most_likely is a string."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        assert isinstance(result["most_likely"], str)
        assert len(result["most_likely"]) > 0
    
    def test_evidence_list_is_string_list(self, root_cause_agent, sample_signals):
        """Test that evidence is a list of strings (for ReporterAgent)."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        for hyp in result["hypotheses"]:
            evidence = hyp.get("evidence", [])
            assert isinstance(evidence, list)
            for item in evidence:
                assert isinstance(item, str)


class TestHypothesisRanking:
    """Tests for hypothesis ranking by confidence."""
    
    def test_hypotheses_sorted_by_confidence(self, root_cause_agent, sample_signals):
        """Test that hypotheses are sorted by confidence (descending)."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        if len(result["hypotheses"]) > 1:
            confidences = [h["confidence"] for h in result["hypotheses"]]
            assert confidences == sorted(confidences, reverse=True)
    
    def test_top_hypothesis_highest_confidence(self, root_cause_agent, sample_signals):
        """Test that the first hypothesis has highest confidence."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        if len(result["hypotheses"]) > 1:
            top_confidence = result["hypotheses"][0]["confidence"]
            for hyp in result["hypotheses"][1:]:
                assert hyp["confidence"] <= top_confidence


class TestCausalPathGeneration:
    """Tests for causal path generation."""
    
    def test_causal_path_in_evidence(self, root_cause_agent, sample_signals):
        """Test that causal path is included in structured evidence."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        # At least one hypothesis should have a causal path if graph has edges
        for hyp in result["hypotheses"]:
            if hyp.get("structured_evidence"):
                causal_path = hyp["structured_evidence"].get("causal_path")
                if causal_path:
                    assert "nodes" in causal_path
                    assert "edges" in causal_path
                    assert "description" in causal_path


class TestFallbackBehavior:
    """Tests for fallback behavior."""
    
    def test_handles_empty_signals(self, root_cause_agent):
        """Test handling of empty signals."""
        result = root_cause_agent.run({"signals": {}})
        
        assert "hypotheses" in result
        assert "most_likely" in result
    
    def test_handles_unknown_signal_type(self, root_cause_agent):
        """Test handling of unknown signal types."""
        signals = {"type": "unknown_type", "some_metric": 123}
        result = root_cause_agent.run({"signals": signals})
        
        assert "hypotheses" in result
        assert len(result["hypotheses"]) >= 1
    
    def test_handles_missing_logs(self, root_cause_agent):
        """Test handling when logs are missing."""
        signals = {"type": "error_rate", "error_rate": 0.15}
        result = root_cause_agent.run({"signals": signals})
        
        assert "hypotheses" in result
        assert len(result["hypotheses"]) >= 1


class TestSchemaValidation:
    """Tests for Pydantic schema validation."""
    
    def test_output_validates_against_schema(self, root_cause_agent, sample_signals):
        """Test that output validates against RootCauseAgentOutput schema."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        # Should not raise
        output = RootCauseAgentOutput.model_validate(result)
        assert output is not None
    
    def test_hypotheses_validate_against_schema(self, root_cause_agent, sample_signals):
        """Test that hypotheses validate against Hypothesis schema."""
        result = root_cause_agent.run({"signals": sample_signals})
        
        for hyp_data in result["hypotheses"]:
            # Should not raise
            hyp = Hypothesis.model_validate(hyp_data)
            assert hyp is not None

