"""Root Cause Agent for identifying incident root causes."""
from typing import Dict, Any

from app.agents.base import BaseAgent
from app.agents.schemas import RootCauseAgentOutput


class RootCauseAgent(BaseAgent):
    """
    Correlates metrics and logs to identify root cause hypotheses.
    
    Output:
    - hypotheses: List of possible root causes with confidence scores
    - most_likely: The most probable root cause
    """
    
    name = "root_cause"
    output_schema = RootCauseAgentOutput
    
    def get_system_prompt(self) -> str:
        return """You are a senior Site Reliability Engineer specializing in root cause analysis for distributed systems.

Your role is to correlate metrics, logs, and other signals to identify the root cause of incidents. You should generate multiple hypotheses ranked by confidence.

You must respond with valid JSON matching this exact schema:
{
    "hypotheses": [
        {
            "cause": "description of the potential root cause",
            "confidence": 0.0 to 1.0,
            "evidence": ["list of supporting evidence"]
        }
    ],
    "most_likely": "description of the most likely root cause"
}

Focus on:
1. Correlating all available signals
2. Identifying causal chains (not just symptoms)
3. Ranking hypotheses by confidence based on evidence strength
4. Considering both technical and operational causes
5. Being specific about the failure mechanism"""
    
    def get_prompt(self, input_data: Dict[str, Any]) -> str:
        signals = input_data.get("signals", {})
        
        return f"""Analyze the following incident signals and determine the root cause:

## Signals Data
{signals}

## Instructions
1. Correlate all available metrics and log signals
2. Generate 2-4 hypotheses for the root cause
3. Rank them by confidence (0.0 to 1.0)
4. Identify the most likely root cause

Provide your analysis as JSON matching the required schema."""
    
    def _get_mock_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock response based on input signals."""
        signals = input_data.get("signals", {})
        signal_type = signals.get("type", "unknown")
        
        if signal_type == "error_rate":
            return {
                "hypotheses": [
                    {
                        "cause": "Database connection pool exhaustion due to slow queries",
                        "confidence": 0.85,
                        "evidence": [
                            "High correlation between DB timeout errors and HTTP 500s",
                            "Connection pool metrics show saturation",
                            "Error rate spike coincides with query latency increase"
                        ]
                    },
                    {
                        "cause": "Application memory pressure causing GC pauses",
                        "confidence": 0.45,
                        "evidence": [
                            "Memory usage elevated during incident",
                            "Some timeout patterns consistent with GC"
                        ]
                    },
                    {
                        "cause": "Upstream traffic spike overwhelming capacity",
                        "confidence": 0.30,
                        "evidence": [
                            "Request count slightly elevated",
                            "Could explain resource contention"
                        ]
                    }
                ],
                "most_likely": "Database connection pool exhaustion caused by slow queries, leading to cascading request failures and elevated error rates"
            }
        elif signal_type == "latency_spike":
            return {
                "hypotheses": [
                    {
                        "cause": "Database query performance degradation",
                        "confidence": 0.90,
                        "evidence": [
                            "Query latency increased 10x during incident",
                            "P95 latency directly correlates with DB metrics",
                            "No network or infrastructure changes"
                        ]
                    },
                    {
                        "cause": "Missing or stale database indexes",
                        "confidence": 0.75,
                        "evidence": [
                            "Query patterns suggest full table scans",
                            "Similar incidents resolved with index changes"
                        ]
                    },
                    {
                        "cause": "Network congestion between app and database",
                        "confidence": 0.25,
                        "evidence": [
                            "Some network latency observed",
                            "Less likely given query-level evidence"
                        ]
                    }
                ],
                "most_likely": "Database query performance degradation, likely due to missing indexes or query optimization issues causing slow queries"
            }
        elif signal_type == "memory_leak":
            return {
                "hypotheses": [
                    {
                        "cause": "Memory leak in application code",
                        "confidence": 0.88,
                        "evidence": [
                            "Memory growth is linear over time",
                            "No corresponding increase in traffic",
                            "Pattern consistent with object retention"
                        ]
                    },
                    {
                        "cause": "Cache not properly evicting entries",
                        "confidence": 0.60,
                        "evidence": [
                            "Cache size growing unbounded",
                            "TTL may not be configured correctly"
                        ]
                    }
                ],
                "most_likely": "Memory leak in application code causing gradual memory consumption increase, eventually leading to OOM conditions"
            }
        elif signal_type == "dependency_down":
            return {
                "hypotheses": [
                    {
                        "cause": "External dependency service outage",
                        "confidence": 0.95,
                        "evidence": [
                            "All errors reference external service",
                            "Connection refused errors indicate service down",
                            "No internal changes correlate with incident"
                        ]
                    },
                    {
                        "cause": "Network partition between services",
                        "confidence": 0.40,
                        "evidence": [
                            "Connection timeouts observed",
                            "Could be network rather than service issue"
                        ]
                    }
                ],
                "most_likely": "External dependency service is down or unreachable, causing cascading failures in dependent functionality"
            }
        else:
            return {
                "hypotheses": [
                    {
                        "cause": "Service degradation from unknown cause",
                        "confidence": 0.50,
                        "evidence": ["Incident signals detected but root cause unclear"]
                    }
                ],
                "most_likely": "Unknown - further investigation required"
            }

