"""API request/response schemas."""
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field
from app.db.models import IncidentStatus, Severity


# ============== Incident Schemas ==============

class IncidentBase(BaseModel):
    """Base incident schema."""
    title: str = Field(..., min_length=1, max_length=500)
    severity: Severity = Field(default=Severity.SEV3)
    start_time: datetime
    end_time: Optional[datetime] = None
    signals_json: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IncidentCreate(IncidentBase):
    """Schema for creating an incident."""
    pass


class IncidentUpdate(BaseModel):
    """Schema for updating an incident."""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    status: Optional[IncidentStatus] = None
    severity: Optional[Severity] = None
    end_time: Optional[datetime] = None
    signals_json: Optional[Dict[str, Any]] = None
    final_summary_json: Optional[Dict[str, Any]] = None
    rca_markdown: Optional[str] = None


class AgentRunResponse(BaseModel):
    """Response schema for agent runs."""
    id: UUID
    incident_id: UUID
    agent_name: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    input_json: Optional[Dict[str, Any]] = None
    output_json: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class IncidentResponse(BaseModel):
    """Response schema for incidents."""
    id: UUID
    created_at: datetime
    updated_at: datetime
    status: IncidentStatus
    title: str
    severity: Severity
    start_time: datetime
    end_time: Optional[datetime] = None
    signals_json: Optional[Dict[str, Any]] = None
    final_summary_json: Optional[Dict[str, Any]] = None
    rca_markdown: Optional[str] = None
    
    class Config:
        from_attributes = True


class IncidentDetailResponse(IncidentResponse):
    """Detailed incident response including agent runs."""
    agent_runs: List[AgentRunResponse] = Field(default_factory=list)


class IncidentListResponse(BaseModel):
    """Response for listing incidents."""
    items: List[IncidentResponse]
    total: int
    page: int
    size: int


# ============== Demo/Admin Schemas ==============

class FaultRequest(BaseModel):
    """Request to trigger a fault in demo service."""
    type: str = Field(
        ..., 
        description="Fault type: latency_spike, error_rate, db_slow, memory_leak, dependency_down"
    )
    duration_seconds: Optional[int] = Field(default=60, description="How long the fault should last")


class FaultResponse(BaseModel):
    """Response from fault trigger."""
    success: bool
    message: str
    fault_type: Optional[str] = None
    expires_at: Optional[datetime] = None


class DemoStatusResponse(BaseModel):
    """Current demo service status."""
    current_fault: Optional[str] = None
    fault_expires_at: Optional[datetime] = None
    healthy: bool
    uptime_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    database: str = "connected"
    redis: str = "connected"
    demo_service: str = "unknown"


# ============== Analysis/Rerun Schemas ==============

class RerunRequest(BaseModel):
    """Request to rerun incident analysis."""
    agents: Optional[List[str]] = Field(
        default=None,
        description="Specific agents to rerun. If None, runs all agents."
    )


class RerunResponse(BaseModel):
    """Response from rerun request."""
    success: bool
    message: str
    task_id: Optional[str] = None


# ============== RCA Evidence Schemas ==============

class TimelineEvent(BaseModel):
    """A single event in the evidence timeline."""
    timestamp: str
    event: str
    component: Optional[str] = None
    severity: str = "info"  # info, warning, critical
    event_type: str = "signal"  # signal, metric, log, agent


class EvidenceTimeline(BaseModel):
    """Complete timeline of events leading to the incident."""
    events: List[TimelineEvent] = Field(default_factory=list)
    first_anomaly: Optional[str] = None
    detection_time: Optional[str] = None
    resolution_time: Optional[str] = None


class CorrelatedMetricEvidence(BaseModel):
    """Evidence from a correlated metric."""
    metric_name: str
    value: float
    correlation_strength: float = Field(ge=0, le=1)
    direction: str = "positive"
    unit: Optional[str] = None


class LogPatternEvidence(BaseModel):
    """Evidence from log patterns."""
    pattern: str
    occurrences: int = 0
    component: Optional[str] = None
    sample_message: Optional[str] = None


class TemporalEvidence(BaseModel):
    """Temporal ordering evidence."""
    event: str
    timestamp: Optional[str] = None
    precedes: List[str] = Field(default_factory=list)
    lag_seconds: Optional[float] = None


class CausalChainStep(BaseModel):
    """A step in the causal chain."""
    component: str
    component_type: str
    anomaly_score: float = Field(ge=0, le=1)
    is_root: bool = False


class RootCauseCandidate(BaseModel):
    """A root cause candidate with confidence and evidence."""
    rank: int
    cause: str
    confidence: float = Field(ge=0, le=1)
    component: Optional[str] = None
    component_type: Optional[str] = None
    evidence_summary: List[str] = Field(default_factory=list)
    correlated_metrics: List[CorrelatedMetricEvidence] = Field(default_factory=list)
    log_patterns: List[LogPatternEvidence] = Field(default_factory=list)
    temporal_evidence: List[TemporalEvidence] = Field(default_factory=list)
    causal_chain: List[CausalChainStep] = Field(default_factory=list)


class CausalGraphSummary(BaseModel):
    """Summary of the causal graph analysis."""
    node_count: int = 0
    edge_count: int = 0
    root_candidates: List[str] = Field(default_factory=list)
    primary_causal_chain: Optional[str] = None
    graph_analysis_available: bool = False


class RCAEvidenceResponse(BaseModel):
    """Complete RCA evidence response."""
    incident_id: UUID
    timeline: EvidenceTimeline
    root_cause_candidates: List[RootCauseCandidate] = Field(default_factory=list)
    causal_graph_summary: CausalGraphSummary
    most_likely_cause: Optional[str] = None
    analysis_complete: bool = False


# ============== Chaos Engineering Schemas ==============

class FaultGroundTruthResponse(BaseModel):
    """Ground truth definition for a fault type."""
    fault_type: str
    display_name: str
    description: str
    expected_root_cause_keywords: List[str]
    expected_component: str
    expected_component_type: str
    expected_metrics: List[str] = Field(default_factory=list)
    expected_log_patterns: List[str] = Field(default_factory=list)
    expected_severity: str = "SEV2"
    max_detection_time_seconds: float = 60.0


class ChaosExperimentCreate(BaseModel):
    """Request to create a chaos experiment."""
    name: str = Field(description="Human-readable experiment name")
    description: Optional[str] = None
    fault_type: str = Field(description="Type of fault to inject")
    fault_duration_seconds: int = Field(default=60, ge=10, le=300)
    scheduled_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)


class ChaosExperimentResponse(BaseModel):
    """Response for a chaos experiment."""
    id: str
    name: str
    description: Optional[str] = None
    fault_type: str
    fault_duration_seconds: int
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str
    incident_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class DetectionResultResponse(BaseModel):
    """Detection evaluation result."""
    incident_detected: bool
    detection_time_seconds: Optional[float] = None
    within_threshold: bool
    incident_id: Optional[str] = None
    incident_title: Optional[str] = None
    incident_severity: Optional[str] = None


class RootCauseResultResponse(BaseModel):
    """Root cause evaluation result."""
    root_cause_identified: bool
    identified_cause: Optional[str] = None
    identified_component: Optional[str] = None
    identified_component_type: Optional[str] = None
    matches_ground_truth: bool
    component_match: bool
    confidence: float = 0.0


class EvaluationMetricsResponse(BaseModel):
    """Evaluation metrics response."""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_time_to_detection: Optional[float] = None
    min_time_to_detection: Optional[float] = None
    max_time_to_detection: Optional[float] = None
    root_cause_accuracy: float
    component_accuracy: float
    root_cause_matches: int
    component_matches: int
    total_evaluated: int


class ExperimentResultResponse(BaseModel):
    """Complete experiment result response."""
    experiment_id: str
    experiment_name: str
    fault_type: str
    started_at: datetime
    completed_at: datetime
    fault_duration_seconds: int
    success: bool
    ground_truth: FaultGroundTruthResponse
    detection: DetectionResultResponse
    root_cause: RootCauseResultResponse
    metrics: EvaluationMetricsResponse


class ExperimentScheduleCreate(BaseModel):
    """Request to create an experiment schedule."""
    name: str
    description: Optional[str] = None
    fault_types: List[str] = Field(
        default_factory=list,
        description="Fault types to test. Empty = all types."
    )
    fault_duration_seconds: int = Field(default=60, ge=10, le=300)
    interval_between_experiments_seconds: int = Field(default=120, ge=30)


class ExperimentScheduleResponse(BaseModel):
    """Response for an experiment schedule."""
    id: str
    name: str
    description: Optional[str] = None
    status: str
    total_experiments: int
    completed_experiments: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    aggregate_metrics: Optional[EvaluationMetricsResponse] = None
    results: List[ExperimentResultResponse] = Field(default_factory=list)


class RunExperimentRequest(BaseModel):
    """Request to run an experiment immediately."""
    fault_type: str
    fault_duration_seconds: int = Field(default=60, ge=10, le=300)
    wait_for_analysis: bool = True
    analysis_timeout_seconds: int = Field(default=120, ge=30, le=300)

