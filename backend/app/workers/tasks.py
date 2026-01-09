"""Celery task definitions."""
import asyncio
import structlog
from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID

from celery import shared_task
from sqlalchemy import select, and_
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from app.config import settings
from app.db.models import Incident, AgentRun, IncidentStatus, Severity
from app.workers.celery_app import celery_app

logger = structlog.get_logger()

# Create sync engine for Celery tasks
sync_engine = create_engine(settings.DATABASE_SYNC_URL, pool_pre_ping=True)


def get_sync_session() -> Session:
    """Get a synchronous database session for Celery tasks."""
    return Session(sync_engine)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def detect_incidents(self):
    """
    Periodic task to detect incidents from metrics and logs.
    Runs every 60 seconds via Celery beat.
    """
    logger.info("Running incident detection")
    
    try:
        # Import detection service
        from app.services.detection.detector import IncidentDetector
        
        detector = IncidentDetector()
        
        # Check for anomalies
        anomalies = detector.detect_anomalies()
        
        if not anomalies:
            logger.info("No anomalies detected")
            return {"detected": 0}
        
        logger.info("Anomalies detected", count=len(anomalies))
        
        with get_sync_session() as session:
            created_count = 0
            
            for anomaly in anomalies:
                # Check for duplicate incidents within the same time window (idempotency)
                window_start = anomaly["start_time"] - timedelta(minutes=5)
                window_end = anomaly["start_time"] + timedelta(minutes=5)
                
                existing = session.execute(
                    select(Incident).where(
                        and_(
                            Incident.title == anomaly["title"],
                            Incident.start_time >= window_start,
                            Incident.start_time <= window_end,
                            Incident.status != IncidentStatus.RESOLVED
                        )
                    )
                ).scalar_one_or_none()
                
                if existing:
                    logger.info("Duplicate incident skipped", title=anomaly["title"])
                    continue
                
                # Create new incident
                incident = Incident(
                    title=anomaly["title"],
                    severity=Severity(anomaly.get("severity", "SEV3")),
                    start_time=anomaly["start_time"],
                    signals_json=anomaly.get("signals", {}),
                    status=IncidentStatus.OPEN
                )
                session.add(incident)
                session.flush()
                
                logger.info("Incident created", incident_id=str(incident.id), title=incident.title)
                
                # Trigger analysis
                run_incident_analysis.delay(str(incident.id))
                created_count += 1
            
            session.commit()
        
        return {"detected": len(anomalies), "created": created_count}
    
    except Exception as e:
        logger.error("Incident detection failed", error=str(e))
        raise self.retry(exc=e)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def run_incident_analysis(self, incident_id: str, agents: Optional[List[str]] = None):
    """
    Run multi-agent analysis for an incident.
    
    Args:
        incident_id: UUID of the incident to analyze
        agents: Optional list of specific agents to run
    """
    logger.info("Starting incident analysis", incident_id=incident_id, agents=agents)
    
    try:
        with get_sync_session() as session:
            # Get incident
            incident = session.execute(
                select(Incident).where(Incident.id == UUID(incident_id))
            ).scalar_one_or_none()
            
            if not incident:
                logger.error("Incident not found", incident_id=incident_id)
                return {"error": "Incident not found"}
            
            # Update status to INVESTIGATING
            incident.status = IncidentStatus.INVESTIGATING
            session.commit()
        
        # Import and run orchestration
        from app.services.orchestration.supervisor import Supervisor
        
        supervisor = Supervisor()
        result = supervisor.analyze_incident(incident_id, agents)
        
        with get_sync_session() as session:
            incident = session.execute(
                select(Incident).where(Incident.id == UUID(incident_id))
            ).scalar_one_or_none()
            
            if incident:
                incident.status = IncidentStatus.RESOLVED
                incident.end_time = datetime.utcnow()
                incident.final_summary_json = result.get("summary", {})
                incident.rca_markdown = result.get("rca_markdown", "")
                session.commit()
        
        logger.info("Incident analysis complete", incident_id=incident_id)
        return {"success": True, "incident_id": incident_id}
    
    except Exception as e:
        logger.error("Incident analysis failed", incident_id=incident_id, error=str(e))
        raise self.retry(exc=e)


@celery_app.task
def cleanup_old_data():
    """Clean up old resolved incidents and agent runs."""
    logger.info("Running data cleanup")
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        with get_sync_session() as session:
            # Delete old resolved incidents (cascades to agent_runs)
            result = session.execute(
                select(Incident).where(
                    and_(
                        Incident.status == IncidentStatus.RESOLVED,
                        Incident.updated_at < cutoff_date
                    )
                )
            )
            old_incidents = result.scalars().all()
            
            for incident in old_incidents:
                session.delete(incident)
            
            session.commit()
            
            logger.info("Cleanup complete", deleted_count=len(old_incidents))
            return {"deleted": len(old_incidents)}
    
    except Exception as e:
        logger.error("Cleanup failed", error=str(e))
        return {"error": str(e)}

