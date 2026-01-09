"""Mitigation Agent for recommending incident response actions."""
from typing import Dict, Any

from app.agents.base import BaseAgent
from app.agents.schemas import MitigationAgentOutput


class MitigationAgent(BaseAgent):
    """
    Recommends mitigation actions based on incident analysis.
    
    Output:
    - immediate_actions: Actions to take right now
    - longer_term_fixes: Actions to prevent recurrence
    - risk_notes: Important considerations and risks
    """
    
    name = "mitigation"
    output_schema = MitigationAgentOutput
    
    def get_system_prompt(self) -> str:
        return """You are an experienced Site Reliability Engineer specializing in incident response and mitigation strategies.

Your role is to recommend actionable mitigation steps for incidents, both immediate fixes and longer-term improvements.

You must respond with valid JSON matching this exact schema:
{
    "immediate_actions": [
        {
            "action": "specific action to take",
            "priority": "high|medium|low",
            "estimated_impact": "description of expected impact"
        }
    ],
    "longer_term_fixes": [
        {
            "action": "specific improvement or fix",
            "priority": "high|medium|low",
            "estimated_impact": "description of expected improvement"
        }
    ],
    "risk_notes": ["list of risks or considerations"]
}

Action types to consider:
- Rollback: Revert recent deployments
- Scale: Add capacity (horizontal or vertical)
- Circuit breaker: Enable/configure circuit breakers
- Retry policy: Adjust retry behavior
- Cache: Enable or adjust caching
- DB index: Add or optimize database indexes
- Feature flag: Disable problematic features
- Rate limiting: Adjust rate limits
- Restart: Restart affected services

Focus on:
1. Immediate actions to restore service
2. Prioritizing by impact and urgency
3. Longer-term fixes to prevent recurrence
4. Noting any risks or dependencies"""
    
    def get_prompt(self, input_data: Dict[str, Any]) -> str:
        signals = input_data.get("signals", {})
        
        return f"""Based on the following incident signals, recommend mitigation actions:

## Signals Data
{signals}

## Instructions
1. Identify 2-4 immediate actions to restore service
2. Identify 2-3 longer-term fixes
3. Note any risks or considerations
4. Prioritize by impact and feasibility

Provide your recommendations as JSON matching the required schema."""
    
    def _get_mock_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock response based on input signals."""
        signals = input_data.get("signals", {})
        signal_type = signals.get("type", "unknown")
        
        if signal_type == "error_rate":
            return {
                "immediate_actions": [
                    {
                        "action": "Increase database connection pool size",
                        "priority": "high",
                        "estimated_impact": "Should restore connection availability within 2-3 minutes"
                    },
                    {
                        "action": "Enable circuit breaker for database calls",
                        "priority": "high",
                        "estimated_impact": "Prevents cascade failures, allows partial functionality"
                    },
                    {
                        "action": "Scale up database read replicas",
                        "priority": "medium",
                        "estimated_impact": "Reduces load on primary, improves query performance"
                    }
                ],
                "longer_term_fixes": [
                    {
                        "action": "Implement connection pool monitoring and auto-scaling",
                        "priority": "high",
                        "estimated_impact": "Prevents future pool exhaustion incidents"
                    },
                    {
                        "action": "Add query timeout limits and retry with backoff",
                        "priority": "medium",
                        "estimated_impact": "Improves resilience to slow queries"
                    },
                    {
                        "action": "Review and optimize slow database queries",
                        "priority": "medium",
                        "estimated_impact": "Reduces baseline load and latency"
                    }
                ],
                "risk_notes": [
                    "Increasing connection pool may impact database performance if connections aren't used efficiently",
                    "Scaling should be monitored to prevent over-provisioning",
                    "Verify circuit breaker thresholds are appropriate for traffic patterns"
                ]
            }
        elif signal_type == "latency_spike":
            return {
                "immediate_actions": [
                    {
                        "action": "Add or optimize missing database indexes",
                        "priority": "high",
                        "estimated_impact": "Query performance improvement of 5-10x expected"
                    },
                    {
                        "action": "Enable query result caching",
                        "priority": "high",
                        "estimated_impact": "Reduces database load for repeated queries"
                    },
                    {
                        "action": "Increase request timeout thresholds temporarily",
                        "priority": "medium",
                        "estimated_impact": "Prevents timeout errors while fixing root cause"
                    }
                ],
                "longer_term_fixes": [
                    {
                        "action": "Implement query performance monitoring and alerting",
                        "priority": "high",
                        "estimated_impact": "Early detection of query performance degradation"
                    },
                    {
                        "action": "Review database schema and query patterns",
                        "priority": "medium",
                        "estimated_impact": "Identifies optimization opportunities"
                    }
                ],
                "risk_notes": [
                    "Adding indexes requires database maintenance window for large tables",
                    "Cache invalidation strategy must be considered",
                    "Extended timeouts may mask underlying issues"
                ]
            }
        elif signal_type == "memory_leak":
            return {
                "immediate_actions": [
                    {
                        "action": "Perform rolling restart of affected service instances",
                        "priority": "high",
                        "estimated_impact": "Immediately frees leaked memory"
                    },
                    {
                        "action": "Scale out to distribute load and reduce per-instance memory",
                        "priority": "medium",
                        "estimated_impact": "Buys time while investigating root cause"
                    }
                ],
                "longer_term_fixes": [
                    {
                        "action": "Profile application to identify memory leak source",
                        "priority": "high",
                        "estimated_impact": "Enables permanent fix"
                    },
                    {
                        "action": "Implement memory usage alerts and automatic restarts",
                        "priority": "medium",
                        "estimated_impact": "Prevents future OOM incidents"
                    },
                    {
                        "action": "Review object lifecycle and caching policies",
                        "priority": "medium",
                        "estimated_impact": "May identify unnecessary object retention"
                    }
                ],
                "risk_notes": [
                    "Rolling restarts may briefly impact availability",
                    "Memory profiling may impact production performance",
                    "Root cause may be in a dependency rather than application code"
                ]
            }
        elif signal_type == "dependency_down":
            return {
                "immediate_actions": [
                    {
                        "action": "Enable circuit breaker to fail fast on dependency calls",
                        "priority": "high",
                        "estimated_impact": "Prevents cascade failures, improves response times"
                    },
                    {
                        "action": "Switch to fallback/cached data if available",
                        "priority": "high",
                        "estimated_impact": "Maintains partial functionality"
                    },
                    {
                        "action": "Contact dependency service team for status",
                        "priority": "high",
                        "estimated_impact": "Enables coordination and ETA for resolution"
                    }
                ],
                "longer_term_fixes": [
                    {
                        "action": "Implement comprehensive fallback strategy",
                        "priority": "high",
                        "estimated_impact": "Reduces dependency on external services"
                    },
                    {
                        "action": "Add dependency health monitoring and alerting",
                        "priority": "medium",
                        "estimated_impact": "Earlier detection of dependency issues"
                    },
                    {
                        "action": "Consider caching layer for dependency data",
                        "priority": "medium",
                        "estimated_impact": "Reduces coupling to external service availability"
                    }
                ],
                "risk_notes": [
                    "Fallback data may be stale and lead to inconsistencies",
                    "Circuit breaker timeout values need tuning",
                    "Some functionality may not work without the dependency"
                ]
            }
        else:
            return {
                "immediate_actions": [
                    {
                        "action": "Investigate incident signals and gather more data",
                        "priority": "high",
                        "estimated_impact": "Enables targeted mitigation"
                    }
                ],
                "longer_term_fixes": [
                    {
                        "action": "Improve monitoring and alerting coverage",
                        "priority": "medium",
                        "estimated_impact": "Better detection and diagnosis of future incidents"
                    }
                ],
                "risk_notes": ["Root cause not fully determined - proceed with caution"]
            }

