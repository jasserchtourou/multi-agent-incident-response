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

