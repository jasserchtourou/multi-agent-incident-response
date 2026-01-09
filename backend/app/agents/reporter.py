"""Reporter Agent for generating RCA reports."""
from typing import Dict, Any
from datetime import datetime

from app.agents.base import BaseAgent
from app.agents.schemas import ReporterAgentOutput


class ReporterAgent(BaseAgent):
    """
    Generates comprehensive RCA reports from agent outputs.
    
    Output:
    - markdown_report: Full RCA report in Markdown format
    - executive_summary: Brief summary for stakeholders
    - action_items: List of follow-up actions
    """
    
    name = "reporter"
    output_schema = ReporterAgentOutput
    
    def get_system_prompt(self) -> str:
        return """You are a technical writer specializing in incident Root Cause Analysis (RCA) reports.

Your role is to synthesize analysis from multiple sources into a clear, comprehensive RCA report.

You must respond with valid JSON matching this exact schema:
{
    "markdown_report": "full RCA report in Markdown format",
    "executive_summary": "2-3 sentence summary for executives",
    "action_items": ["list of follow-up action items"]
}

The Markdown report should include:
1. Executive Summary
2. Incident Timeline
3. Impact Assessment
4. Root Cause Analysis
5. Mitigation Actions Taken
6. Prevention Recommendations
7. Action Items

Focus on:
1. Clear, professional language
2. Actionable insights
3. Well-organized structure
4. Both technical and business audience"""
    
    def get_prompt(self, input_data: Dict[str, Any]) -> str:
        incident_title = input_data.get("incident_title", "Incident")
        signals = input_data.get("signals", {})
        agent_outputs = input_data.get("agent_outputs", {})
        
        return f"""Generate a comprehensive RCA report for the following incident:

## Incident Title
{incident_title}

## Signals
{signals}

## Agent Analysis Outputs
{agent_outputs}

## Instructions
1. Create a complete RCA report in Markdown
2. Write a brief executive summary
3. List specific action items

Provide the report as JSON matching the required schema."""
    
    def _get_mock_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock RCA report based on input data."""
        incident_title = input_data.get("incident_title", "Incident")
        signals = input_data.get("signals", {})
        agent_outputs = input_data.get("agent_outputs", {})
        signal_type = signals.get("type", "unknown")
        
        now = datetime.utcnow().isoformat()
        
        # Extract data from agent outputs
        anomaly_summary = agent_outputs.get("anomaly_summary", "Anomaly detected")
        most_likely_cause = agent_outputs.get("most_likely_cause", "Under investigation")
        immediate_actions = agent_outputs.get("immediate_actions", [])
        longer_term_fixes = agent_outputs.get("longer_term_fixes", [])
        hypotheses = agent_outputs.get("hypotheses", [])
        timeline = agent_outputs.get("timeline", [])
        
        # Build timeline section
        timeline_md = "\n".join([
            f"- **{event.get('timestamp', 'T-0')}**: {event.get('event', 'Event')} ({event.get('severity', 'info')})"
            for event in timeline
        ]) if timeline else "- Timeline data not available"
        
        # Build hypotheses section
        hypotheses_md = "\n".join([
            f"- **{h.get('cause', 'Unknown')}** (Confidence: {h.get('confidence', 0):.0%})"
            for h in hypotheses
        ]) if hypotheses else "- Hypothesis data not available"
        
        # Build immediate actions section
        actions_md = "\n".join([
            f"- {a.get('action', 'Action')} (Priority: {a.get('priority', 'medium')})"
            for a in immediate_actions
        ]) if immediate_actions else "- Actions pending investigation"
        
        # Build prevention recommendations
        prevention_md = "\n".join([
            f"- {f.get('action', 'Fix')} (Priority: {f.get('priority', 'medium')})"
            for f in longer_term_fixes
        ]) if longer_term_fixes else "- Prevention measures to be determined"
        
        # Build action items
        action_items = []
        for a in immediate_actions[:3]:
            action_items.append(a.get("action", "Complete immediate action"))
        for f in longer_term_fixes[:2]:
            action_items.append(f"[Follow-up] {f.get('action', 'Implement long-term fix')}")
        action_items.append("Schedule post-incident review meeting")
        action_items.append("Update runbooks based on learnings")
        
        # Generate markdown report
        markdown_report = f"""# Root Cause Analysis Report

## Incident: {incident_title}

**Generated**: {now}
**Status**: Resolved

---

## Executive Summary

{anomaly_summary}

The root cause was identified as: **{most_likely_cause}**

Immediate mitigation actions were taken to restore service, and longer-term prevention measures have been identified.

---

## Incident Timeline

{timeline_md}

---

## Impact Assessment

- **Severity**: {signals.get('severity', 'SEV3') if isinstance(signals.get('severity'), str) else 'SEV3'}
- **Duration**: Approximately 5-10 minutes
- **Affected Services**: Primary API endpoints
- **User Impact**: Elevated error rates and degraded response times

---

## Root Cause Analysis

### Hypotheses Evaluated

{hypotheses_md}

### Confirmed Root Cause

{most_likely_cause}

---

## Mitigation Actions Taken

{actions_md}

---

## Prevention Recommendations

{prevention_md}

---

## Action Items

{chr(10).join([f'{i+1}. {item}' for i, item in enumerate(action_items)])}

---

## Appendix

### Signals Data
- Type: {signal_type}
- See attached metrics and log data for details.

---

*This report was generated by the Multi-Agent Incident Response System.*
"""
        
        executive_summary = f"{incident_title}: {anomaly_summary} Root cause identified as {most_likely_cause}. Service has been restored and prevention measures identified."
        
        return {
            "markdown_report": markdown_report,
            "executive_summary": executive_summary,
            "action_items": action_items
        }

