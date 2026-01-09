"""Metrics client for fetching observability data."""
import structlog
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from app.config import settings

logger = structlog.get_logger()


class BaseMetricsClient(ABC):
    """Abstract base class for metrics clients."""
    
    @abstractmethod
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get recent metrics summary."""
        pass
    
    @abstractmethod
    def get_metric_history(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get metric history over a time range."""
        pass


class SimulatedMetricsClient(BaseMetricsClient):
    """Simulated metrics client for demo mode."""
    
    def __init__(self):
        self.demo_service_url = settings.DEMO_SERVICE_URL
    
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Fetch metrics from demo service."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.demo_service_url}/metrics/summary")
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            logger.warning("Failed to fetch metrics from demo service", error=str(e))
            return self._get_default_metrics()
        except Exception as e:
            logger.error("Metrics fetch error", error=str(e))
            return self._get_default_metrics()
    
    def get_metric_history(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get metric history from demo service."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    f"{self.demo_service_url}/metrics/history",
                    params={
                        "metric": metric_name,
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    }
                )
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception as e:
            logger.error("Metric history fetch error", error=str(e))
            return []
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default healthy metrics."""
        return {
            "error_rate": 0.01,
            "latency_p95_ms": 150,
            "latency_avg_ms": 50,
            "request_count": 1000,
            "memory_usage_mb": 256,
            "cpu_usage_percent": 25,
            "timestamp": datetime.utcnow().isoformat()
        }


class PrometheusMetricsClient(BaseMetricsClient):
    """Prometheus metrics client for production mode."""
    
    def __init__(self):
        self.prometheus_url = settings.PROMETHEUS_URL
    
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        """Query Prometheus for recent metrics."""
        try:
            with httpx.Client(timeout=10.0) as client:
                metrics = {}
                
                # Error rate query
                error_rate = self._query_prometheus(
                    client,
                    f'rate(http_requests_total{{status=~"5.."}}[{minutes}m]) / rate(http_requests_total[{minutes}m])'
                )
                metrics["error_rate"] = error_rate or 0
                
                # Latency p95 query
                latency_p95 = self._query_prometheus(
                    client,
                    f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[{minutes}m])) * 1000'
                )
                metrics["latency_p95_ms"] = latency_p95 or 0
                
                # Request count
                request_count = self._query_prometheus(
                    client,
                    f'increase(http_requests_total[{minutes}m])'
                )
                metrics["request_count"] = request_count or 0
                
                # Memory usage
                memory = self._query_prometheus(
                    client,
                    'process_resident_memory_bytes / 1024 / 1024'
                )
                metrics["memory_usage_mb"] = memory or 0
                
                metrics["timestamp"] = datetime.utcnow().isoformat()
                
                return metrics
        
        except Exception as e:
            logger.error("Prometheus query error", error=str(e))
            return {"error": str(e)}
    
    def get_metric_history(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Query Prometheus range API for metric history."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.prometheus_url}/api/v1/query_range",
                    params={
                        "query": metric_name,
                        "start": start_time.timestamp(),
                        "end": end_time.timestamp(),
                        "step": "60s"
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                results = []
                if data.get("status") == "success":
                    for result in data.get("data", {}).get("result", []):
                        for value in result.get("values", []):
                            results.append({
                                "timestamp": datetime.fromtimestamp(value[0]),
                                "value": float(value[1])
                            })
                return results
        
        except Exception as e:
            logger.error("Prometheus range query error", error=str(e))
            return []
    
    def _query_prometheus(self, client: httpx.Client, query: str) -> Optional[float]:
        """Execute a Prometheus instant query."""
        try:
            response = client.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query}
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success":
                results = data.get("data", {}).get("result", [])
                if results:
                    return float(results[0].get("value", [0, 0])[1])
            return None
        
        except Exception as e:
            logger.warning("Prometheus query failed", query=query, error=str(e))
            return None


class MetricsClient(BaseMetricsClient):
    """
    Factory class that returns the appropriate metrics client based on config.
    """
    
    def __init__(self):
        if settings.DATA_MODE == "prometheus_loki":
            self._client = PrometheusMetricsClient()
        else:
            self._client = SimulatedMetricsClient()
    
    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, Any]:
        return self._client.get_recent_metrics(minutes)
    
    def get_metric_history(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        return self._client.get_metric_history(metric_name, start_time, end_time)

