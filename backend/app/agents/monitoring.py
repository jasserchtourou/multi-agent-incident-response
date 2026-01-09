"""Monitoring Agent for analyzing metrics and anomalies."""
from typing import Dict, Any

from app.agents.base import BaseAgent
from app.agents.schemas import MonitoringAgentOutput


class MonitoringAgent(BaseAgent):
    """
    Analyzes metrics data to summarize anomalies and create timelines.
    
    Output:
    - anomaly_summary: Brief summary of detected anomalies
    - key_metrics: List of important metrics with their status
    - timeline: Chronological events during the incident
    """
    
    name = "monitoring"
    output_schema = MonitoringAgentOutput
    
    def get_system_prompt(self) -> str:
        return """You are a Site Reliability Engineering (SRE) expert specializing in metrics analysis and incident detection.

Your role is to analyze metrics data from monitoring systems to identify anomalies, trends, and create a timeline of events.

You must respond with valid JSON matching this exact schema:
{
    "anomaly_summary": "string describing the overall anomaly situation",
    "key_metrics": [
        {
            "name": "metric name",
            "value": numeric_value,
            "unit": "unit of measurement",
            "status": "normal|warning|critical"
        }
    ],
    "timeline": [
        {
            "timestamp": "ISO timestamp",
            "event": "description of what happened",
            "severity": "info|warning|critical"
        }
    ]
}

Focus on:
1. Identifying the most significant metric deviations
2. Correlating metrics that may be related
3. Building a clear timeline of when issues started and progressed
4. Assessing severity based on thresholds and trends"""
    
    def get_prompt(self, input_data: Dict[str, Any]) -> str:
        signals = input_data.get("signals", {})
        
        return f"""Analyze the following metrics data and identify anomalies:

## Signals Data
{signals}

## Instructions
1. Identify all anomalous metrics and their severity
2. Create a timeline of significant events
3. Summarize the overall situation

Provide your analysis as JSON matching the required schema."""
    
    def _get_mock_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock response based on input signals."""
        signals = input_data.get("signals", {})
        signal_type = signals.get("type", "unknown")
        
        # Generate appropriate mock based on signal type
        if signal_type == "error_rate":
            return {
                "anomaly_summary": f"High error rate detected at {signals.get('error_rate', 0):.1%}. This exceeds normal thresholds and indicates service degradation.",
                "key_metrics": [
                    {"name": "error_rate", "value": signals.get("error_rate", 0.1) * 100, "unit": "%", "status": "critical"},
                    {"name": "error_count", "value": signals.get("error_count", 50), "unit": "errors", "status": "critical"},
                    {"name": "request_count", "value": signals.get("metrics_snapshot", {}).get("request_count", 1000), "unit": "requests", "status": "normal"}
                ],
                "timeline": [
                    {"timestamp": "T-5m", "event": "Error rate began increasing", "severity": "warning"},
                    {"timestamp": "T-3m", "event": "Error rate crossed critical threshold", "severity": "critical"},
                    {"timestamp": "T-0m", "event": "Incident detected and triggered", "severity": "critical"}
                ]
            }
        elif signal_type == "latency_spike":
            return {
                "anomaly_summary": f"Latency spike detected with p95 at {signals.get('latency_p95_ms', 2000)}ms. Response times are significantly degraded.",
                "key_metrics": [
                    {"name": "latency_p95", "value": signals.get("latency_p95_ms", 2000), "unit": "ms", "status": "critical"},
                    {"name": "latency_avg", "value": signals.get("latency_avg_ms", 500), "unit": "ms", "status": "warning"},
                    {"name": "request_count", "value": signals.get("request_count", 800), "unit": "requests", "status": "normal"}
                ],
                "timeline": [
                    {"timestamp": "T-5m", "event": "Latency began increasing", "severity": "warning"},
                    {"timestamp": "T-2m", "event": "P95 latency exceeded 1s threshold", "severity": "critical"},
                    {"timestamp": "T-0m", "event": "Incident triggered due to sustained high latency", "severity": "critical"}
                ]
            }
        elif signal_type == "memory_leak":
            return {
                "anomaly_summary": f"Memory usage is critically high at {signals.get('memory_usage_mb', 600)}MB, suggesting a potential memory leak.",
                "key_metrics": [
                    {"name": "memory_usage", "value": signals.get("memory_usage_mb", 600), "unit": "MB", "status": "critical"},
                    {"name": "memory_threshold", "value": signals.get("memory_threshold_mb", 500), "unit": "MB", "status": "normal"}
                ],
                "timeline": [
                    {"timestamp": "T-30m", "event": "Memory usage trend started increasing", "severity": "info"},
                    {"timestamp": "T-10m", "event": "Memory crossed warning threshold", "severity": "warning"},
                    {"timestamp": "T-0m", "event": "Memory exceeded critical threshold", "severity": "critical"}
                ]
            }
        else:
            return {
                "anomaly_summary": "Anomaly detected in system metrics. Further investigation needed.",
                "key_metrics": [
                    {"name": "status", "value": 1, "unit": "boolean", "status": "warning"}
                ],
                "timeline": [
                    {"timestamp": "T-0m", "event": "Incident detected", "severity": "warning"}
                ]
            }

