"""Tests for incident detection rules."""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from app.services.detection.detector import IncidentDetector


class TestIncidentDetector:
    """Test cases for IncidentDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = IncidentDetector()
    
    def test_detect_error_rate_spike(self):
        """Test detection of high error rates."""
        metrics = {
            "error_rate": 0.15,  # 15% error rate, above 5% threshold
            "latency_p95_ms": 100,
            "memory_usage_mb": 256
        }
        logs = [
            {"level": "error", "message": "HTTP 500"},
            {"level": "error", "message": "Database error"},
            {"level": "info", "message": "Request processed"}
        ]
        now = datetime.utcnow()
        
        anomaly = self.detector._detect_error_rate_spike(metrics, logs, now)
        
        assert anomaly is not None
        assert "Error Rate" in anomaly["title"]
        assert anomaly["severity"] == "SEV2"  # > 10%
        assert anomaly["signals"]["type"] == "error_rate"
    
    def test_no_error_rate_anomaly_when_below_threshold(self):
        """Test that no anomaly is detected when error rate is normal."""
        metrics = {
            "error_rate": 0.01,  # 1% error rate, below 5% threshold
            "latency_p95_ms": 100,
            "memory_usage_mb": 256
        }
        logs = []
        now = datetime.utcnow()
        
        anomaly = self.detector._detect_error_rate_spike(metrics, logs, now)
        
        assert anomaly is None
    
    def test_detect_latency_spike(self):
        """Test detection of high latency."""
        metrics = {
            "error_rate": 0.01,
            "latency_p95_ms": 2500,  # 2.5s, above 1s threshold
            "latency_avg_ms": 800,
            "request_count": 1000,
            "memory_usage_mb": 256
        }
        now = datetime.utcnow()
        
        anomaly = self.detector._detect_latency_spike(metrics, now)
        
        assert anomaly is not None
        assert "Latency" in anomaly["title"]
        assert anomaly["severity"] == "SEV2"  # > 2000ms
        assert anomaly["signals"]["type"] == "latency_spike"
    
    def test_detect_memory_issue(self):
        """Test detection of memory issues."""
        metrics = {
            "memory_usage_mb": 600,  # Above 500MB threshold
        }
        now = datetime.utcnow()
        
        anomaly = self.detector._detect_memory_issue(metrics, now)
        
        assert anomaly is not None
        assert "Memory" in anomaly["title"]
        assert anomaly["signals"]["type"] == "memory_leak"
    
    def test_detect_dependency_failure(self):
        """Test detection of dependency failures."""
        logs = [
            {"level": "error", "message": "Connection timeout to external API"},
            {"level": "error", "message": "Dependency service connection refused"},
            {"level": "error", "message": "Timeout waiting for dependency response"},
            {"level": "info", "message": "Request processed"}
        ]
        now = datetime.utcnow()
        
        anomaly = self.detector._detect_dependency_failure(logs, now)
        
        assert anomaly is not None
        assert "Dependency" in anomaly["title"]
        assert anomaly["signals"]["type"] == "dependency_down"
    
    def test_baseline_anomaly_detection(self):
        """Test moving average baseline anomaly detection."""
        historical = [100, 105, 98, 102, 101, 99, 103, 100]
        
        # Normal value should not be anomalous
        assert not self.detector._calculate_baseline_anomaly(102, historical)
        
        # Extremely high value should be anomalous
        assert self.detector._calculate_baseline_anomaly(200, historical)
        
        # Extremely low value should be anomalous
        assert self.detector._calculate_baseline_anomaly(20, historical)
    
    def test_baseline_anomaly_insufficient_data(self):
        """Test baseline detection with insufficient historical data."""
        # With less than 3 data points, should return False
        assert not self.detector._calculate_baseline_anomaly(100, [50, 60])
        assert not self.detector._calculate_baseline_anomaly(100, [])


class TestDetectionIntegration:
    """Integration tests for detection with mocked clients."""
    
    @patch('app.services.detection.detector.MetricsClient')
    @patch('app.services.detection.detector.LogsClient')
    def test_detect_anomalies_with_multiple_issues(self, MockLogsClient, MockMetricsClient):
        """Test detection when multiple issues are present."""
        # Mock metrics client
        mock_metrics = Mock()
        mock_metrics.get_recent_metrics.return_value = {
            "error_rate": 0.2,  # High
            "latency_p95_ms": 3000,  # High
            "memory_usage_mb": 256  # Normal
        }
        MockMetricsClient.return_value = mock_metrics
        
        # Mock logs client
        mock_logs = Mock()
        mock_logs.get_recent_logs.return_value = [
            {"level": "error", "message": "HTTP 500"},
            {"level": "error", "message": "Database timeout"}
        ]
        MockLogsClient.return_value = mock_logs
        
        detector = IncidentDetector()
        anomalies = detector.detect_anomalies()
        
        # Should detect at least error rate and latency anomalies
        assert len(anomalies) >= 2
        anomaly_types = [a["signals"]["type"] for a in anomalies]
        assert "error_rate" in anomaly_types
        assert "latency_spike" in anomaly_types

