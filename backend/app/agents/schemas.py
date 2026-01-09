"""Pydantic schemas for agent inputs and outputs."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ============== Monitoring Agent ==============

class MetricInfo(BaseModel):
    """Information about a key metric."""
    name: str
    value: float
    unit: str
    status: str = Field(description="normal, warning, or critical")


class TimelineEvent(BaseModel):
    """A timeline event during the incident."""
    timestamp: str
    event: str
    severity: str = "info"


class MonitoringAgentOutput(BaseModel):
    """Output schema for MonitoringAgent."""
    anomaly_summary: str = Field(description="Summary of detected anomalies")
    key_metrics: List[MetricInfo] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)


# ============== Log Analysis Agent ==============

class ErrorInfo(BaseModel):
    """Information about an error."""
    message: str
    count: int
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


class PatternInfo(BaseModel):
    """Information about an error pattern."""
    pattern: str
    frequency: int
    significance: str


class LogAnalysisAgentOutput(BaseModel):
    """Output schema for LogAnalysisAgent."""
    top_errors: List[ErrorInfo] = Field(default_factory=list)
    patterns: List[PatternInfo] = Field(default_factory=list)
    probable_component: str = Field(description="Most likely affected component")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")


# ============== Root Cause Agent ==============

class Hypothesis(BaseModel):
    """A root cause hypothesis."""
    cause: str
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    evidence: List[str] = Field(default_factory=list)


class RootCauseAgentOutput(BaseModel):
    """Output schema for RootCauseAgent."""
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    most_likely: str = Field(description="Most likely root cause")


# ============== Mitigation Agent ==============

class Action(BaseModel):
    """A mitigation action."""
    action: str
    priority: str = Field(description="high, medium, or low")
    estimated_impact: str


class MitigationAgentOutput(BaseModel):
    """Output schema for MitigationAgent."""
    immediate_actions: List[Action] = Field(default_factory=list)
    longer_term_fixes: List[Action] = Field(default_factory=list)
    risk_notes: List[str] = Field(default_factory=list)


# ============== Reporter Agent ==============

class ReporterAgentOutput(BaseModel):
    """Output schema for ReporterAgent."""
    markdown_report: str = Field(description="Full RCA report in Markdown")
    executive_summary: str = Field(description="Brief executive summary")
    action_items: List[str] = Field(default_factory=list)

