"""Tests for schema validation."""
import pytest
from pydantic import ValidationError

from app.agents.schemas import (
    MonitoringAgentOutput,
    LogAnalysisAgentOutput,
    RootCauseAgentOutput,
    MitigationAgentOutput,
    ReporterAgentOutput,
    MetricInfo,
    Hypothesis,
    Action
)


class TestMonitoringAgentSchema:
    """Test MonitoringAgentOutput schema validation."""
    
    def test_valid_output(self):
        """Test valid monitoring agent output."""
        data = {
            "anomaly_summary": "High error rate detected",
            "key_metrics": [
                {"name": "error_rate", "value": 0.15, "unit": "%", "status": "critical"}
            ],
            "timeline": [
                {"timestamp": "2024-01-01T00:00:00Z", "event": "Error spike", "severity": "critical"}
            ]
        }
        
        output = MonitoringAgentOutput.model_validate(data)
        assert output.anomaly_summary == "High error rate detected"
        assert len(output.key_metrics) == 1
        assert len(output.timeline) == 1
    
    def test_empty_lists_allowed(self):
        """Test that empty lists are allowed."""
        data = {
            "anomaly_summary": "No anomalies",
            "key_metrics": [],
            "timeline": []
        }
        
        output = MonitoringAgentOutput.model_validate(data)
        assert output.key_metrics == []
        assert output.timeline == []


class TestRootCauseAgentSchema:
    """Test RootCauseAgentOutput schema validation."""
    
    def test_valid_output(self):
        """Test valid root cause output."""
        data = {
            "hypotheses": [
                {
                    "cause": "Database connection exhaustion",
                    "confidence": 0.85,
                    "evidence": ["High connection count", "Timeout errors"]
                },
                {
                    "cause": "Memory leak",
                    "confidence": 0.45,
                    "evidence": ["Rising memory usage"]
                }
            ],
            "most_likely": "Database connection exhaustion"
        }
        
        output = RootCauseAgentOutput.model_validate(data)
        assert len(output.hypotheses) == 2
        assert output.hypotheses[0].confidence == 0.85
        assert output.most_likely == "Database connection exhaustion"
    
    def test_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        data = {
            "hypotheses": [
                {"cause": "Test", "confidence": 1.5, "evidence": []}
            ],
            "most_likely": "Test"
        }
        
        with pytest.raises(ValidationError):
            RootCauseAgentOutput.model_validate(data)
        
        data["hypotheses"][0]["confidence"] = -0.1
        with pytest.raises(ValidationError):
            RootCauseAgentOutput.model_validate(data)


class TestMitigationAgentSchema:
    """Test MitigationAgentOutput schema validation."""
    
    def test_valid_output(self):
        """Test valid mitigation output."""
        data = {
            "immediate_actions": [
                {
                    "action": "Restart service",
                    "priority": "high",
                    "estimated_impact": "Restores availability"
                }
            ],
            "longer_term_fixes": [
                {
                    "action": "Add circuit breaker",
                    "priority": "medium",
                    "estimated_impact": "Prevents cascading failures"
                }
            ],
            "risk_notes": ["Service restart may cause brief downtime"]
        }
        
        output = MitigationAgentOutput.model_validate(data)
        assert len(output.immediate_actions) == 1
        assert len(output.longer_term_fixes) == 1
        assert len(output.risk_notes) == 1


class TestReporterAgentSchema:
    """Test ReporterAgentOutput schema validation."""
    
    def test_valid_output(self):
        """Test valid reporter output."""
        data = {
            "markdown_report": "# RCA Report\n\nIncident analysis...",
            "executive_summary": "Service disruption due to database issues.",
            "action_items": ["Restart database", "Add monitoring"]
        }
        
        output = ReporterAgentOutput.model_validate(data)
        assert "RCA Report" in output.markdown_report
        assert len(output.action_items) == 2


class TestMetricInfoSchema:
    """Test MetricInfo schema."""
    
    def test_valid_metric(self):
        """Test valid metric info."""
        metric = MetricInfo(
            name="error_rate",
            value=0.05,
            unit="%",
            status="warning"
        )
        
        assert metric.name == "error_rate"
        assert metric.value == 0.05
    
    def test_default_status(self):
        """Test that status defaults correctly."""
        data = {"name": "test", "value": 100, "unit": "ms"}
        # MetricInfo has default for status
        metric = MetricInfo.model_validate(data)
        assert metric.status is not None

