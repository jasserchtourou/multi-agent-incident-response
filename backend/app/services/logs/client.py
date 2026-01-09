"""Logs client for fetching application logs."""
import structlog
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any, List
from abc import ABC, abstractmethod

from app.config import settings

logger = structlog.get_logger()


class BaseLogsClient(ABC):
    """Abstract base class for logs clients."""
    
    @abstractmethod
    def get_recent_logs(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent log entries."""
        pass
    
    @abstractmethod
    def search_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search logs with a query."""
        pass


class SimulatedLogsClient(BaseLogsClient):
    """Simulated logs client for demo mode."""
    
    def __init__(self):
        self.demo_service_url = settings.DEMO_SERVICE_URL
    
    def get_recent_logs(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Fetch logs from demo service."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    f"{self.demo_service_url}/logs/recent",
                    params={"minutes": minutes}
                )
                response.raise_for_status()
                return response.json().get("logs", [])
        except httpx.RequestError as e:
            logger.warning("Failed to fetch logs from demo service", error=str(e))
            return []
        except Exception as e:
            logger.error("Logs fetch error", error=str(e))
            return []
    
    def search_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search logs from demo service."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    f"{self.demo_service_url}/logs/search",
                    params={
                        "query": query,
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "limit": limit
                    }
                )
                response.raise_for_status()
                return response.json().get("logs", [])
        except Exception as e:
            logger.error("Log search error", error=str(e))
            return []


class LokiLogsClient(BaseLogsClient):
    """Loki logs client for production mode."""
    
    def __init__(self):
        self.loki_url = settings.LOKI_URL
    
    def get_recent_logs(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Query Loki for recent logs."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=minutes)
        
        return self.search_logs("{job=\"demo_service\"}", start_time, end_time)
    
    def search_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search logs using Loki's query API."""
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(
                    f"{self.loki_url}/loki/api/v1/query_range",
                    params={
                        "query": query,
                        "start": int(start_time.timestamp() * 1e9),  # Nanoseconds
                        "end": int(end_time.timestamp() * 1e9),
                        "limit": limit
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                logs = []
                if data.get("status") == "success":
                    for stream in data.get("data", {}).get("result", []):
                        labels = stream.get("stream", {})
                        for value in stream.get("values", []):
                            timestamp_ns, log_line = value
                            logs.append({
                                "timestamp": datetime.fromtimestamp(int(timestamp_ns) / 1e9).isoformat(),
                                "message": log_line,
                                "labels": labels,
                                "level": self._extract_level(log_line)
                            })
                
                return logs
        
        except Exception as e:
            logger.error("Loki query error", error=str(e))
            return []
    
    def _extract_level(self, log_line: str) -> str:
        """Extract log level from log line."""
        log_lower = log_line.lower()
        if "error" in log_lower:
            return "error"
        elif "warn" in log_lower:
            return "warning"
        elif "debug" in log_lower:
            return "debug"
        return "info"


class LogsClient(BaseLogsClient):
    """
    Factory class that returns the appropriate logs client based on config.
    """
    
    def __init__(self):
        if settings.DATA_MODE == "prometheus_loki":
            self._client = LokiLogsClient()
        else:
            self._client = SimulatedLogsClient()
    
    def get_recent_logs(self, minutes: int = 5) -> List[Dict[str, Any]]:
        return self._client.get_recent_logs(minutes)
    
    def search_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        return self._client.search_logs(query, start_time, end_time, limit)

