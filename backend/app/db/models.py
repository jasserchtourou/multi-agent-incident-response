"""Database models for incident management."""
import enum
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column, String, Text, DateTime, Enum, ForeignKey, 
    Float, Integer, JSON, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class IncidentStatus(str, enum.Enum):
    """Incident status enumeration."""
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"


class Severity(str, enum.Enum):
    """Incident severity levels."""
    SEV1 = "SEV1"  # Critical - immediate action required
    SEV2 = "SEV2"  # High - major functionality impacted
    SEV3 = "SEV3"  # Medium - partial impact
    SEV4 = "SEV4"  # Low - minor issues


class Incident(Base):
    """Incident model representing a detected issue."""
    
    __tablename__ = "incidents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Status and classification
    status = Column(Enum(IncidentStatus), default=IncidentStatus.OPEN, nullable=False)
    title = Column(String(500), nullable=False)
    severity = Column(Enum(Severity), default=Severity.SEV3, nullable=False)
    
    # Timeline
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Data
    signals_json = Column(JSON, nullable=True, default=dict)  # Metrics/log pointers
    final_summary_json = Column(JSON, nullable=True, default=dict)  # Merged agent outputs
    rca_markdown = Column(Text, nullable=True)  # Final RCA report
    
    # Relationships
    agent_runs = relationship("AgentRun", back_populates="incident", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Incident(id={self.id}, title={self.title}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary."""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status.value if self.status else None,
            "title": self.title,
            "severity": self.severity.value if self.severity else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "signals_json": self.signals_json,
            "final_summary_json": self.final_summary_json,
            "rca_markdown": self.rca_markdown,
        }


class AgentRun(Base):
    """Record of an agent execution for an incident."""
    
    __tablename__ = "agent_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(UUID(as_uuid=True), ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False)
    
    # Agent identification
    agent_name = Column(String(100), nullable=False)
    
    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    
    # Input/Output
    input_json = Column(JSON, nullable=True, default=dict)
    output_json = Column(JSON, nullable=True, default=dict)
    
    # Metrics (optional)
    tokens_used = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    incident = relationship("Incident", back_populates="agent_runs")
    
    def __repr__(self) -> str:
        return f"<AgentRun(id={self.id}, agent={self.agent_name}, incident_id={self.incident_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent run to dictionary."""
        return {
            "id": str(self.id),
            "incident_id": str(self.incident_id),
            "agent_name": self.agent_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "input_json": self.input_json,
            "output_json": self.output_json,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
        }

