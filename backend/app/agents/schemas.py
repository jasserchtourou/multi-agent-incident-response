"""Pydantic schemas for agent inputs and outputs."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ============== Monitoring Agent ==============

class MetricInfo(BaseModel):
    """Information about a key metric."""
    name: str
    value: float
    unit: str
    status: str = Field(default="normal", description="normal, warning, or critical")


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

class CorrelatedMetric(BaseModel):
    """A metric correlated with the root cause."""
    metric_name: str
    value: float
    correlation_strength: float = Field(ge=0, le=1, description="How strongly correlated 0-1")
    direction: str = Field(default="positive", description="positive or negative correlation")


class LogPattern(BaseModel):
    """A log pattern supporting the hypothesis."""
    pattern: str
    occurrences: int = 0
    first_seen: Optional[str] = None
    component: Optional[str] = None


class TemporalEvidence(BaseModel):
    """Temporal ordering evidence for causality."""
    event: str
    timestamp: Optional[str] = None
    precedes: List[str] = Field(default_factory=list, description="Events this preceded")
    lag_seconds: Optional[float] = None


class CausalPathNode(BaseModel):
    """A node in the causal path."""
    node_id: str
    name: str
    node_type: str
    anomaly_score: float = Field(ge=0, le=1)
    is_root: bool = False


class CausalPathEdge(BaseModel):
    """An edge in the causal path."""
    source: str
    target: str
    weight: float = Field(ge=0, le=1)
    edge_type: str


class CausalPath(BaseModel):
    """A causal path from root cause to symptoms."""
    nodes: List[CausalPathNode] = Field(default_factory=list)
    edges: List[CausalPathEdge] = Field(default_factory=list)
    description: str = ""


class StructuredEvidence(BaseModel):
    """Structured evidence supporting a hypothesis."""
    correlated_metrics: List[CorrelatedMetric] = Field(default_factory=list)
    log_patterns: List[LogPattern] = Field(default_factory=list)
    temporal_evidence: List[TemporalEvidence] = Field(default_factory=list)
    causal_path: Optional[CausalPath] = None


class Hypothesis(BaseModel):
    """A root cause hypothesis with structured evidence."""
    cause: str
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    evidence: List[str] = Field(default_factory=list, description="Human-readable evidence")
    structured_evidence: Optional[StructuredEvidence] = None
    component: Optional[str] = Field(default=None, description="Affected component name")
    component_type: Optional[str] = Field(default=None, description="service, database, cache, etc.")


class CausalGraphSummary(BaseModel):
    """Summary of the causal graph analysis."""
    node_count: int = 0
    edge_count: int = 0
    root_candidates: List[str] = Field(default_factory=list)
    primary_causal_chain: Optional[str] = None


class RootCauseAgentOutput(BaseModel):
    """Output schema for RootCauseAgent."""
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    most_likely: str = Field(description="Most likely root cause")
    causal_graph_summary: Optional[CausalGraphSummary] = None


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

