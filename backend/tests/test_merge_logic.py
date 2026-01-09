"""Tests for agent output merge logic."""
import pytest
from app.services.orchestration.supervisor import Supervisor


class TestMergeLogic:
    """Test cases for Supervisor._merge_outputs method."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.supervisor = Supervisor()
    
    def test_merge_empty_outputs(self):
        """Test merging empty agent outputs."""
        outputs = {}
        
        merged = self.supervisor._merge_outputs(outputs)
        
        assert merged["anomaly_summary"] == ""
        assert merged["key_metrics"] == []
        assert merged["hypotheses"] == []
        assert merged["immediate_actions"] == []
    
    def test_merge_monitoring_output(self):
        """Test merging monitoring agent output."""
        outputs = {
            "monitoring": {
                "anomaly_summary": "High error rate detected",
                "key_metrics": [
                    {"name": "error_rate", "value": 0.15, "unit": "%", "status": "critical"}
                ],
                "timeline": [
                    {"timestamp": "T-5m", "event": "Error spike started", "severity": "warning"}
                ]
            }
        }
        
        merged = self.supervisor._merge_outputs(outputs)
        
        assert merged["anomaly_summary"] == "High error rate detected"
        assert len(merged["key_metrics"]) == 1
        assert len(merged["timeline"]) == 1
    
    def test_merge_log_analysis_output(self):
        """Test merging log analysis agent output."""
        outputs = {
            "log_analysis": {
                "top_errors": [
                    {"message": "HTTP 500", "count": 45}
                ],
                "patterns": [
                    {"pattern": "DB connection failures", "frequency": 30, "significance": "high"}
                ],
                "probable_component": "Database",
                "evidence": ["Connection errors in logs"]
            }
        }
        
        merged = self.supervisor._merge_outputs(outputs)
        
        assert len(merged["top_errors"]) == 1
        assert len(merged["error_patterns"]) == 1
        assert merged["probable_component"] == "Database"
    
    def test_merge_root_cause_output(self):
        """Test merging root cause agent output."""
        outputs = {
            "root_cause": {
                "hypotheses": [
                    {"cause": "DB exhaustion", "confidence": 0.85, "evidence": ["Test"]}
                ],
                "most_likely": "DB exhaustion"
            }
        }
        
        merged = self.supervisor._merge_outputs(outputs)
        
        assert len(merged["hypotheses"]) == 1
        assert merged["most_likely_cause"] == "DB exhaustion"
    
    def test_merge_mitigation_output(self):
        """Test merging mitigation agent output."""
        outputs = {
            "mitigation": {
                "immediate_actions": [
                    {"action": "Restart DB", "priority": "high", "estimated_impact": "Restores service"}
                ],
                "longer_term_fixes": [
                    {"action": "Add monitoring", "priority": "medium", "estimated_impact": "Early detection"}
                ],
                "risk_notes": ["Brief downtime during restart"]
            }
        }
        
        merged = self.supervisor._merge_outputs(outputs)
        
        assert len(merged["immediate_actions"]) == 1
        assert len(merged["longer_term_fixes"]) == 1
        assert len(merged["risk_notes"]) == 1
    
    def test_merge_all_outputs(self):
        """Test merging all agent outputs together."""
        outputs = {
            "monitoring": {
                "anomaly_summary": "Error spike",
                "key_metrics": [{"name": "error_rate", "value": 0.15, "unit": "%", "status": "critical"}],
                "timeline": [{"timestamp": "T-5m", "event": "Started", "severity": "warning"}]
            },
            "log_analysis": {
                "top_errors": [{"message": "HTTP 500", "count": 45}],
                "patterns": [],
                "probable_component": "API Gateway",
                "evidence": []
            },
            "root_cause": {
                "hypotheses": [{"cause": "DB issue", "confidence": 0.9, "evidence": []}],
                "most_likely": "DB issue"
            },
            "mitigation": {
                "immediate_actions": [{"action": "Restart", "priority": "high", "estimated_impact": "Fix"}],
                "longer_term_fixes": [],
                "risk_notes": []
            }
        }
        
        merged = self.supervisor._merge_outputs(outputs)
        
        assert merged["anomaly_summary"] == "Error spike"
        assert len(merged["key_metrics"]) == 1
        assert merged["probable_component"] == "API Gateway"
        assert merged["most_likely_cause"] == "DB issue"
        assert len(merged["immediate_actions"]) == 1
    
    def test_merge_handles_errors(self):
        """Test that merge handles agent errors gracefully."""
        outputs = {
            "monitoring": {"error": "Agent failed"},
            "log_analysis": {
                "top_errors": [{"message": "Test", "count": 1}],
                "patterns": [],
                "probable_component": "Test",
                "evidence": []
            }
        }
        
        merged = self.supervisor._merge_outputs(outputs)
        
        # Should skip monitoring due to error, but include log_analysis
        assert merged["anomaly_summary"] == ""  # From error agent
        assert len(merged["top_errors"]) == 1  # From working agent
    
    def test_merge_extends_timeline(self):
        """Test that timeline events are extended not replaced."""
        outputs = {
            "monitoring": {
                "anomaly_summary": "Test",
                "key_metrics": [],
                "timeline": [
                    {"timestamp": "T-5m", "event": "Event 1", "severity": "warning"},
                    {"timestamp": "T-3m", "event": "Event 2", "severity": "critical"}
                ]
            }
        }
        
        merged = self.supervisor._merge_outputs(outputs)
        
        assert len(merged["timeline"]) == 2

