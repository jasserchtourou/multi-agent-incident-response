"""Log Analysis Agent for analyzing error logs and patterns."""
from typing import Dict, Any, List

from app.agents.base import BaseAgent
from app.agents.schemas import LogAnalysisAgentOutput


class LogAnalysisAgent(BaseAgent):
    """
    Analyzes log data to identify error patterns and affected components.
    
    Output:
    - top_errors: Most frequent errors with counts
    - patterns: Identified error patterns
    - probable_component: Most likely affected component
    - evidence: Supporting evidence from logs
    """
    
    name = "log_analysis"
    output_schema = LogAnalysisAgentOutput
    
    def get_system_prompt(self) -> str:
        return """You are an expert log analyst specializing in distributed systems and error pattern recognition.

Your role is to analyze application logs to identify error patterns, cluster similar errors, and determine the most likely affected component.

You must respond with valid JSON matching this exact schema:
{
    "top_errors": [
        {
            "message": "error message or pattern",
            "count": number_of_occurrences,
            "first_seen": "ISO timestamp or null",
            "last_seen": "ISO timestamp or null"
        }
    ],
    "patterns": [
        {
            "pattern": "description of the pattern",
            "frequency": occurrence_count,
            "significance": "high|medium|low"
        }
    ],
    "probable_component": "name of the most likely affected component",
    "evidence": ["list of evidence supporting the analysis"]
}

Focus on:
1. Identifying the most critical and frequent errors
2. Finding patterns that suggest root cause
3. Determining which component(s) are affected
4. Providing clear evidence for your conclusions"""
    
    def get_prompt(self, input_data: Dict[str, Any]) -> str:
        signals = input_data.get("signals", {})
        sample_errors = signals.get("sample_errors", [])
        
        return f"""Analyze the following log data and identify error patterns:

## Signals Data
{signals}

## Sample Error Logs
{sample_errors}

## Instructions
1. Identify and rank the most significant errors
2. Find patterns that may indicate the root cause
3. Determine the most likely affected component
4. Provide evidence for your conclusions

Provide your analysis as JSON matching the required schema."""
    
    def _get_mock_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock response based on input signals."""
        signals = input_data.get("signals", {})
        signal_type = signals.get("type", "unknown")
        sample_errors = signals.get("sample_errors", [])
        
        # Extract error messages if available
        error_messages = []
        for err in sample_errors:
            if isinstance(err, dict):
                error_messages.append(err.get("message", "Unknown error"))
            elif isinstance(err, str):
                error_messages.append(err)
        
        if signal_type == "error_rate":
            return {
                "top_errors": [
                    {"message": "HTTP 500 Internal Server Error", "count": 45, "first_seen": "T-5m", "last_seen": "T-0m"},
                    {"message": "Connection timeout to database", "count": 23, "first_seen": "T-4m", "last_seen": "T-0m"},
                    {"message": "Request handler exception", "count": 12, "first_seen": "T-3m", "last_seen": "T-1m"}
                ],
                "patterns": [
                    {"pattern": "Database connection failures correlate with HTTP 500s", "frequency": 68, "significance": "high"},
                    {"pattern": "Errors concentrated in /api/work endpoint", "frequency": 50, "significance": "high"},
                    {"pattern": "Retry storms following initial failures", "frequency": 30, "significance": "medium"}
                ],
                "probable_component": "Database connection pool or database service",
                "evidence": [
                    "68% of errors mention database or connection",
                    "Error spike correlates with increased latency",
                    "All affected requests go through database layer"
                ]
            }
        elif signal_type == "dependency_down":
            return {
                "top_errors": [
                    {"message": "Connection refused to external service", "count": 89, "first_seen": "T-5m", "last_seen": "T-0m"},
                    {"message": "Timeout waiting for response from dependency", "count": 34, "first_seen": "T-5m", "last_seen": "T-0m"},
                    {"message": "Circuit breaker opened", "count": 12, "first_seen": "T-3m", "last_seen": "T-0m"}
                ],
                "patterns": [
                    {"pattern": "External API dependency is unreachable", "frequency": 123, "significance": "high"},
                    {"pattern": "Cascading failures through dependent services", "frequency": 45, "significance": "high"}
                ],
                "probable_component": "External API dependency",
                "evidence": [
                    "All errors reference external service connections",
                    "Circuit breaker triggered after repeated failures",
                    "No internal service errors before dependency failures"
                ]
            }
        elif signal_type == "latency_spike":
            return {
                "top_errors": [
                    {"message": "Request timeout exceeded", "count": 34, "first_seen": "T-3m", "last_seen": "T-0m"},
                    {"message": "Slow query warning", "count": 56, "first_seen": "T-5m", "last_seen": "T-0m"}
                ],
                "patterns": [
                    {"pattern": "Database queries taking abnormally long", "frequency": 90, "significance": "high"},
                    {"pattern": "Request queue building up", "frequency": 45, "significance": "medium"}
                ],
                "probable_component": "Database query execution",
                "evidence": [
                    "Slow query logs show queries taking 5-10x normal time",
                    "Connection pool exhaustion warnings",
                    "Latency spike correlates with query performance degradation"
                ]
            }
        else:
            return {
                "top_errors": [
                    {"message": "Error detected in application", "count": 10, "first_seen": "T-5m", "last_seen": "T-0m"}
                ],
                "patterns": [
                    {"pattern": "General application errors", "frequency": 10, "significance": "medium"}
                ],
                "probable_component": "Application service",
                "evidence": ["Errors detected in application logs"]
            }

