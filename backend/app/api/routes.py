"""API routes for incident management."""
import structlog
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import get_db, Incident, AgentRun, IncidentStatus, Severity
from app.api.schemas import (
    IncidentCreate, IncidentUpdate, IncidentResponse, IncidentDetailResponse,
    IncidentListResponse, AgentRunResponse, FaultRequest, FaultResponse,
    DemoStatusResponse, HealthResponse, RerunRequest, RerunResponse
)
from app.config import settings

logger = structlog.get_logger()

router = APIRouter()


# ============== Health ==============

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint."""
    # Check database
    db_status = "connected"
    try:
        await db.execute(select(func.now()))
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        db_status = "disconnected"
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version=settings.APP_VERSION,
        database=db_status,
        redis="connected",  # Will be checked properly later
        demo_service="unknown"
    )


# ============== Incidents CRUD ==============

@router.get("/incidents", response_model=IncidentListResponse, tags=["Incidents"])
async def list_incidents(
    status: Optional[IncidentStatus] = Query(None, description="Filter by status"),
    severity: Optional[Severity] = Query(None, description="Filter by severity"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_db)
):
    """List incidents with optional filtering."""
    query = select(Incident)
    count_query = select(func.count(Incident.id))
    
    # Apply filters
    if status:
        query = query.where(Incident.status == status)
        count_query = count_query.where(Incident.status == status)
    if severity:
        query = query.where(Incident.severity == severity)
        count_query = count_query.where(Incident.severity == severity)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    query = query.order_by(desc(Incident.created_at))
    query = query.offset((page - 1) * size).limit(size)
    
    result = await db.execute(query)
    incidents = result.scalars().all()
    
    return IncidentListResponse(
        items=[IncidentResponse.model_validate(i) for i in incidents],
        total=total,
        page=page,
        size=size
    )


@router.get("/incidents/{incident_id}", response_model=IncidentDetailResponse, tags=["Incidents"])
async def get_incident(
    incident_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get incident details including agent runs and RCA."""
    query = (
        select(Incident)
        .where(Incident.id == incident_id)
        .options(selectinload(Incident.agent_runs))
    )
    result = await db.execute(query)
    incident = result.scalar_one_or_none()
    
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found"
        )
    
    return IncidentDetailResponse(
        **IncidentResponse.model_validate(incident).model_dump(),
        agent_runs=[AgentRunResponse.model_validate(ar) for ar in incident.agent_runs]
    )


@router.post("/incidents", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED, tags=["Incidents"])
async def create_incident(
    incident_data: IncidentCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new incident."""
    incident = Incident(
        title=incident_data.title,
        severity=incident_data.severity,
        start_time=incident_data.start_time,
        end_time=incident_data.end_time,
        signals_json=incident_data.signals_json,
        status=IncidentStatus.OPEN
    )
    
    db.add(incident)
    await db.flush()
    await db.refresh(incident)
    
    logger.info("Incident created", incident_id=str(incident.id), title=incident.title)
    
    return IncidentResponse.model_validate(incident)


@router.patch("/incidents/{incident_id}", response_model=IncidentResponse, tags=["Incidents"])
async def update_incident(
    incident_id: UUID,
    update_data: IncidentUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update an incident."""
    query = select(Incident).where(Incident.id == incident_id)
    result = await db.execute(query)
    incident = result.scalar_one_or_none()
    
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found"
        )
    
    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(incident, field, value)
    
    await db.flush()
    await db.refresh(incident)
    
    logger.info("Incident updated", incident_id=str(incident_id))
    
    return IncidentResponse.model_validate(incident)


@router.delete("/incidents/{incident_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Incidents"])
async def delete_incident(
    incident_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Delete an incident."""
    query = select(Incident).where(Incident.id == incident_id)
    result = await db.execute(query)
    incident = result.scalar_one_or_none()
    
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found"
        )
    
    await db.delete(incident)
    logger.info("Incident deleted", incident_id=str(incident_id))


@router.post("/incidents/{incident_id}/rerun", response_model=RerunResponse, tags=["Incidents"])
async def rerun_analysis(
    incident_id: UUID,
    request: RerunRequest = None,
    db: AsyncSession = Depends(get_db)
):
    """Rerun incident analysis with agents."""
    query = select(Incident).where(Incident.id == incident_id)
    result = await db.execute(query)
    incident = result.scalar_one_or_none()
    
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found"
        )
    
    # Import here to avoid circular imports
    from app.workers.tasks import run_incident_analysis
    
    # Trigger async analysis
    task = run_incident_analysis.delay(str(incident_id), request.agents if request else None)
    
    logger.info("Rerun analysis triggered", incident_id=str(incident_id), task_id=task.id)
    
    return RerunResponse(
        success=True,
        message="Analysis rerun triggered",
        task_id=task.id
    )


# ============== Admin/Demo ==============

@router.post("/admin/fault", response_model=FaultResponse, tags=["Admin"])
async def trigger_fault(request: FaultRequest):
    """Trigger a fault in the demo service."""
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.DEMO_SERVICE_URL}/admin/fault",
                params={"type": request.type, "duration": request.duration_seconds},
                timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            
            return FaultResponse(
                success=True,
                message=f"Fault '{request.type}' activated",
                fault_type=request.type,
                expires_at=datetime.fromisoformat(data.get("expires_at")) if data.get("expires_at") else None
            )
    except httpx.RequestError as e:
        logger.error("Failed to trigger fault", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Demo service unavailable: {str(e)}"
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Demo service error: {e.response.text}"
        )


@router.get("/demo/status", response_model=DemoStatusResponse, tags=["Admin"])
async def get_demo_status():
    """Get current demo service status and active faults."""
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.DEMO_SERVICE_URL}/admin/status",
                timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            
            return DemoStatusResponse(
                current_fault=data.get("current_fault"),
                fault_expires_at=datetime.fromisoformat(data["fault_expires_at"]) if data.get("fault_expires_at") else None,
                healthy=data.get("healthy", True),
                uptime_seconds=data.get("uptime_seconds", 0)
            )
    except httpx.RequestError as e:
        logger.warning("Demo service unavailable", error=str(e))
        return DemoStatusResponse(
            current_fault=None,
            fault_expires_at=None,
            healthy=False,
            uptime_seconds=0
        )

