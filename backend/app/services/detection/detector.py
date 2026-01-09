"""Incident detection service."""
import structlog
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import statistics

from app.config import settings
from app.services.metrics.client import MetricsClient
from app.services.logs.client import LogsClient

logger = structlog.get_logger()


class IncidentDetector:
    """
    Detects incidents from metrics and logs using threshold rules
    and simple anomaly detection.
    """
    
    def __init__(self):
        self.metrics_client = MetricsClient()
        self.logs_client = LogsClient()
        
        # Thresholds from config
        self.error_rate_threshold = settings.ERROR_RATE_THRESHOLD
        self.latency_p95_threshold = settings.LATENCY_P95_THRESHOLD_MS
        self.memory_threshold = settings.MEMORY_THRESHOLD_MB
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Run all detection rules and return list of detected anomalies.
        
        Returns:
            List of anomaly dictionaries with title, severity, start_time, signals
        """
        anomalies = []
        now = datetime.utcnow()
        
        # Get recent metrics and logs
        metrics = self.metrics_client.get_recent_metrics(minutes=5)
        logs = self.logs_client.get_recent_logs(minutes=5)
        
        # Run detection rules
        error_rate_anomaly = self._detect_error_rate_spike(metrics, logs, now)
        if error_rate_anomaly:
            anomalies.append(error_rate_anomaly)
        
        latency_anomaly = self._detect_latency_spike(metrics, now)
        if latency_anomaly:
            anomalies.append(latency_anomaly)
        
        memory_anomaly = self._detect_memory_issue(metrics, now)
        if memory_anomaly:
            anomalies.append(memory_anomaly)
        
        dependency_anomaly = self._detect_dependency_failure(logs, now)
        if dependency_anomaly:
            anomalies.append(dependency_anomaly)
        
        logger.info("Detection complete", anomaly_count=len(anomalies))
        return anomalies
    
    def _detect_error_rate_spike(
        self,
        metrics: Dict[str, Any],
        logs: List[Dict[str, Any]],
        now: datetime
    ) -> Optional[Dict[str, Any]]:
        """Detect high error rates."""
        error_rate = metrics.get("error_rate", 0)
        
        if error_rate > self.error_rate_threshold:
            # Count error logs
            error_logs = [l for l in logs if l.get("level") == "error"]
            
            severity = "SEV2" if error_rate > 0.1 else "SEV3"
            
            return {
                "title": f"High Error Rate Detected ({error_rate:.1%})",
                "severity": severity,
                "start_time": now,
                "signals": {
                    "type": "error_rate",
                    "error_rate": error_rate,
                    "error_count": len(error_logs),
                    "sample_errors": error_logs[:5],
                    "metrics_snapshot": metrics
                }
            }
        return None
    
    def _detect_latency_spike(
        self,
        metrics: Dict[str, Any],
        now: datetime
    ) -> Optional[Dict[str, Any]]:
        """Detect high latency (p95)."""
        latency_p95 = metrics.get("latency_p95_ms", 0)
        
        if latency_p95 > self.latency_p95_threshold:
            severity = "SEV2" if latency_p95 > 2000 else "SEV3"
            
            return {
                "title": f"High Latency Detected (p95: {latency_p95:.0f}ms)",
                "severity": severity,
                "start_time": now,
                "signals": {
                    "type": "latency_spike",
                    "latency_p95_ms": latency_p95,
                    "latency_avg_ms": metrics.get("latency_avg_ms", 0),
                    "request_count": metrics.get("request_count", 0),
                    "metrics_snapshot": metrics
                }
            }
        return None
    
    def _detect_memory_issue(
        self,
        metrics: Dict[str, Any],
        now: datetime
    ) -> Optional[Dict[str, Any]]:
        """Detect memory issues (potential leak)."""
        memory_mb = metrics.get("memory_usage_mb", 0)
        
        if memory_mb > self.memory_threshold:
            severity = "SEV2" if memory_mb > 800 else "SEV3"
            
            return {
                "title": f"High Memory Usage Detected ({memory_mb:.0f}MB)",
                "severity": severity,
                "start_time": now,
                "signals": {
                    "type": "memory_leak",
                    "memory_usage_mb": memory_mb,
                    "memory_threshold_mb": self.memory_threshold,
                    "metrics_snapshot": metrics
                }
            }
        return None
    
    def _detect_dependency_failure(
        self,
        logs: List[Dict[str, Any]],
        now: datetime
    ) -> Optional[Dict[str, Any]]:
        """Detect external dependency failures."""
        dependency_errors = [
            l for l in logs
            if l.get("level") == "error" and (
                "connection" in l.get("message", "").lower() or
                "timeout" in l.get("message", "").lower() or
                "dependency" in l.get("message", "").lower() or
                "external" in l.get("message", "").lower()
            )
        ]
        
        if len(dependency_errors) >= 3:  # At least 3 dependency errors
            return {
                "title": "Dependency Failure Detected",
                "severity": "SEV2",
                "start_time": now,
                "signals": {
                    "type": "dependency_down",
                    "error_count": len(dependency_errors),
                    "sample_errors": dependency_errors[:5],
                }
            }
        return None
    
    def _calculate_baseline_anomaly(
        self,
        current_value: float,
        historical_values: List[float],
        std_multiplier: float = 2.0
    ) -> bool:
        """
        Simple baseline anomaly detection using moving average + std deviation.
        
        Args:
            current_value: The current metric value
            historical_values: List of historical values for baseline
            std_multiplier: Number of standard deviations for threshold
        
        Returns:
            True if current value is anomalous
        """
        if len(historical_values) < 3:
            return False
        
        mean = statistics.mean(historical_values)
        std = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
        
        upper_bound = mean + (std_multiplier * std)
        lower_bound = mean - (std_multiplier * std)
        
        return current_value > upper_bound or current_value < lower_bound

