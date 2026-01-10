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
    DemoStatusResponse, HealthResponse, RerunRequest, RerunResponse,
    RCAEvidenceResponse, EvidenceTimeline, TimelineEvent, RootCauseCandidate,
    CausalGraphSummary, CorrelatedMetricEvidence, LogPatternEvidence,
    TemporalEvidence, CausalChainStep,
    # Chaos Engineering
    FaultGroundTruthResponse, ChaosExperimentCreate, ChaosExperimentResponse,
    ExperimentResultResponse, ExperimentScheduleCreate, ExperimentScheduleResponse,
    RunExperimentRequest, EvaluationMetricsResponse, DetectionResultResponse,
    RootCauseResultResponse
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


@router.get("/incidents/{incident_id}/rca-evidence", response_model=RCAEvidenceResponse, tags=["Incidents"])
async def get_rca_evidence(
    incident_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    Get explainable RCA evidence for an incident.
    
    Returns:
    - Evidence timeline: chronological events leading to the incident
    - Root cause candidates: ranked causes with confidence and supporting evidence
    - Causal graph summary: overview of the dependency analysis
    """
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
    
    # Build evidence from agent runs and incident data
    evidence = _build_rca_evidence(incident)
    
    return evidence


def _build_rca_evidence(incident: Incident) -> RCAEvidenceResponse:
    """Build RCA evidence response from incident data."""
    
    # Extract data sources
    signals = incident.signals_json or {}
    summary = incident.final_summary_json or {}
    
    # Find root_cause agent run output
    root_cause_output = None
    monitoring_output = None
    log_analysis_output = None
    
    for agent_run in incident.agent_runs:
        if agent_run.agent_name == "root_cause" and agent_run.output_json:
            root_cause_output = agent_run.output_json
        elif agent_run.agent_name == "monitoring" and agent_run.output_json:
            monitoring_output = agent_run.output_json
        elif agent_run.agent_name == "log_analysis" and agent_run.output_json:
            log_analysis_output = agent_run.output_json
    
    # Build timeline
    timeline = _build_evidence_timeline(
        incident, signals, summary, monitoring_output, log_analysis_output
    )
    
    # Build root cause candidates
    candidates = _build_root_cause_candidates(root_cause_output, summary)
    
    # Build causal graph summary
    causal_summary = _build_causal_graph_summary(root_cause_output)
    
    # Determine most likely cause
    most_likely = None
    if root_cause_output:
        most_likely = root_cause_output.get("most_likely")
    elif summary.get("most_likely_cause"):
        most_likely = summary.get("most_likely_cause")
    
    return RCAEvidenceResponse(
        incident_id=incident.id,
        timeline=timeline,
        root_cause_candidates=candidates,
        causal_graph_summary=causal_summary,
        most_likely_cause=most_likely,
        analysis_complete=incident.rca_markdown is not None
    )


def _build_evidence_timeline(
    incident: Incident,
    signals: dict,
    summary: dict,
    monitoring_output: Optional[dict],
    log_analysis_output: Optional[dict]
) -> EvidenceTimeline:
    """Build the evidence timeline from incident data."""
    events = []
    
    # Add incident start
    if incident.start_time:
        events.append(TimelineEvent(
            timestamp=incident.start_time.isoformat(),
            event="Incident started - anomaly detected",
            severity="critical",
            event_type="signal"
        ))
    
    # Add timeline from monitoring agent
    if monitoring_output and monitoring_output.get("timeline"):
        for entry in monitoring_output["timeline"]:
            events.append(TimelineEvent(
                timestamp=entry.get("timestamp", ""),
                event=entry.get("event", ""),
                severity=entry.get("severity", "info"),
                event_type="metric"
            ))
    
    # Add timeline from summary (merged agent output)
    if summary.get("timeline"):
        for entry in summary["timeline"]:
            # Avoid duplicates
            existing_events = {e.event for e in events}
            if entry.get("event") not in existing_events:
                events.append(TimelineEvent(
                    timestamp=entry.get("timestamp", ""),
                    event=entry.get("event", ""),
                    severity=entry.get("severity", "info"),
                    event_type="metric"
                ))
    
    # Add log-based events
    if log_analysis_output and log_analysis_output.get("top_errors"):
        for error in log_analysis_output["top_errors"][:3]:
            events.append(TimelineEvent(
                timestamp=error.get("first_seen", ""),
                event=f"Log error: {error.get('message', 'Unknown')} ({error.get('count', 0)} occurrences)",
                component=log_analysis_output.get("probable_component"),
                severity="warning",
                event_type="log"
            ))
    
    # Add agent run events
    for agent_run in incident.agent_runs:
        if agent_run.finished_at:
            events.append(TimelineEvent(
                timestamp=agent_run.finished_at.isoformat(),
                event=f"{agent_run.agent_name.replace('_', ' ').title()} analysis complete",
                severity="info",
                event_type="agent"
            ))
    
    # Add resolution event
    if incident.end_time:
        events.append(TimelineEvent(
            timestamp=incident.end_time.isoformat(),
            event="Incident resolved",
            severity="info",
            event_type="signal"
        ))
    
    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp if e.timestamp else "")
    
    return EvidenceTimeline(
        events=events,
        first_anomaly=incident.start_time.isoformat() if incident.start_time else None,
        detection_time=incident.created_at.isoformat() if incident.created_at else None,
        resolution_time=incident.end_time.isoformat() if incident.end_time else None
    )


def _build_root_cause_candidates(
    root_cause_output: Optional[dict],
    summary: dict
) -> List[RootCauseCandidate]:
    """Build root cause candidates from agent output."""
    candidates = []
    
    hypotheses = []
    if root_cause_output and root_cause_output.get("hypotheses"):
        hypotheses = root_cause_output["hypotheses"]
    elif summary.get("hypotheses"):
        hypotheses = summary["hypotheses"]
    
    for i, hyp in enumerate(hypotheses):
        # Extract structured evidence if available
        correlated_metrics = []
        log_patterns = []
        temporal_evidence = []
        causal_chain = []
        
        structured = hyp.get("structured_evidence", {})
        
        if structured:
            # Correlated metrics
            for metric in structured.get("correlated_metrics", []):
                correlated_metrics.append(CorrelatedMetricEvidence(
                    metric_name=metric.get("metric_name", ""),
                    value=metric.get("value", 0),
                    correlation_strength=metric.get("correlation_strength", 0),
                    direction=metric.get("direction", "positive"),
                    unit=metric.get("unit")
                ))
            
            # Log patterns
            for pattern in structured.get("log_patterns", []):
                log_patterns.append(LogPatternEvidence(
                    pattern=pattern.get("pattern", ""),
                    occurrences=pattern.get("occurrences", 0),
                    component=pattern.get("component"),
                    sample_message=pattern.get("sample_message")
                ))
            
            # Temporal evidence
            for temp in structured.get("temporal_evidence", []):
                temporal_evidence.append(TemporalEvidence(
                    event=temp.get("event", ""),
                    timestamp=temp.get("timestamp"),
                    precedes=temp.get("precedes", []),
                    lag_seconds=temp.get("lag_seconds")
                ))
            
            # Causal chain
            causal_path = structured.get("causal_path", {})
            if causal_path and causal_path.get("nodes"):
                for node in causal_path["nodes"]:
                    causal_chain.append(CausalChainStep(
                        component=node.get("name", ""),
                        component_type=node.get("node_type", "service"),
                        anomaly_score=node.get("anomaly_score", 0),
                        is_root=node.get("is_root", False)
                    ))
        
        candidates.append(RootCauseCandidate(
            rank=i + 1,
            cause=hyp.get("cause", "Unknown"),
            confidence=hyp.get("confidence", 0),
            component=hyp.get("component"),
            component_type=hyp.get("component_type"),
            evidence_summary=hyp.get("evidence", []),
            correlated_metrics=correlated_metrics,
            log_patterns=log_patterns,
            temporal_evidence=temporal_evidence,
            causal_chain=causal_chain
        ))
    
    return candidates


def _build_causal_graph_summary(root_cause_output: Optional[dict]) -> CausalGraphSummary:
    """Build causal graph summary from root cause output."""
    if not root_cause_output:
        return CausalGraphSummary(graph_analysis_available=False)
    
    graph_summary = root_cause_output.get("causal_graph_summary", {})
    
    if not graph_summary:
        return CausalGraphSummary(graph_analysis_available=False)
    
    return CausalGraphSummary(
        node_count=graph_summary.get("node_count", 0),
        edge_count=graph_summary.get("edge_count", 0),
        root_candidates=graph_summary.get("root_candidates", []),
        primary_causal_chain=graph_summary.get("primary_causal_chain"),
        graph_analysis_available=True
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


# ============== Chaos Engineering ==============

# In-memory storage for experiments (would be database in production)
_experiments: dict = {}
_schedules: dict = {}


@router.get("/chaos/faults", response_model=List[FaultGroundTruthResponse], tags=["Chaos Engineering"])
async def list_fault_types():
    """
    List all available fault types with their ground-truth definitions.
    
    Returns the expected behavior for each fault type, useful for
    understanding what the system should detect.
    """
    from app.services.chaos.models import FAULT_GROUND_TRUTHS
    
    return [
        FaultGroundTruthResponse(
            fault_type=gt.fault_type,
            display_name=gt.display_name,
            description=gt.description,
            expected_root_cause_keywords=gt.expected_root_cause_keywords,
            expected_component=gt.expected_component,
            expected_component_type=gt.expected_component_type,
            expected_metrics=gt.expected_metrics,
            expected_log_patterns=gt.expected_log_patterns,
            expected_severity=gt.expected_severity,
            max_detection_time_seconds=gt.max_detection_time_seconds
        )
        for gt in FAULT_GROUND_TRUTHS.values()
    ]


@router.get("/chaos/faults/{fault_type}", response_model=FaultGroundTruthResponse, tags=["Chaos Engineering"])
async def get_fault_ground_truth(fault_type: str):
    """Get the ground-truth definition for a specific fault type."""
    from app.services.chaos.models import FAULT_GROUND_TRUTHS
    
    gt = FAULT_GROUND_TRUTHS.get(fault_type)
    if not gt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown fault type: {fault_type}"
        )
    
    return FaultGroundTruthResponse(
        fault_type=gt.fault_type,
        display_name=gt.display_name,
        description=gt.description,
        expected_root_cause_keywords=gt.expected_root_cause_keywords,
        expected_component=gt.expected_component,
        expected_component_type=gt.expected_component_type,
        expected_metrics=gt.expected_metrics,
        expected_log_patterns=gt.expected_log_patterns,
        expected_severity=gt.expected_severity,
        max_detection_time_seconds=gt.max_detection_time_seconds
    )


@router.post("/chaos/experiments", response_model=ChaosExperimentResponse, tags=["Chaos Engineering"])
async def create_experiment(request: ChaosExperimentCreate):
    """
    Create a new chaos experiment.
    
    The experiment is created but not started. Use the run endpoint to execute it.
    """
    from app.services.chaos.models import ChaosExperiment, FAULT_GROUND_TRUTHS
    
    if request.fault_type not in FAULT_GROUND_TRUTHS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown fault type: {request.fault_type}. Available: {list(FAULT_GROUND_TRUTHS.keys())}"
        )
    
    experiment = ChaosExperiment(
        name=request.name,
        description=request.description,
        fault_type=request.fault_type,
        fault_duration_seconds=request.fault_duration_seconds,
        scheduled_at=request.scheduled_at,
        tags=request.tags
    )
    
    _experiments[experiment.id] = experiment
    
    logger.info("Experiment created", experiment_id=experiment.id, fault_type=experiment.fault_type)
    
    return ChaosExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        fault_type=experiment.fault_type,
        fault_duration_seconds=experiment.fault_duration_seconds,
        scheduled_at=experiment.scheduled_at,
        started_at=experiment.started_at,
        completed_at=experiment.completed_at,
        status=experiment.status.value,
        incident_ids=experiment.incident_ids,
        tags=experiment.tags
    )


@router.get("/chaos/experiments", response_model=List[ChaosExperimentResponse], tags=["Chaos Engineering"])
async def list_experiments():
    """List all chaos experiments."""
    return [
        ChaosExperimentResponse(
            id=exp.id,
            name=exp.name,
            description=exp.description,
            fault_type=exp.fault_type,
            fault_duration_seconds=exp.fault_duration_seconds,
            scheduled_at=exp.scheduled_at,
            started_at=exp.started_at,
            completed_at=exp.completed_at,
            status=exp.status.value,
            incident_ids=exp.incident_ids,
            tags=exp.tags
        )
        for exp in _experiments.values()
    ]


@router.post("/chaos/experiments/run", response_model=ExperimentResultResponse, tags=["Chaos Engineering"])
async def run_experiment(
    request: RunExperimentRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Run a chaos experiment immediately.
    
    This will:
    1. Inject the specified fault into the demo service
    2. Wait for incident detection and analysis
    3. Evaluate the RCA against ground truth
    4. Return the evaluation results
    """
    from app.services.chaos.models import ChaosExperiment, FAULT_GROUND_TRUTHS
    from app.services.chaos.experiment_runner import ExperimentRunner
    
    if request.fault_type not in FAULT_GROUND_TRUTHS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown fault type: {request.fault_type}"
        )
    
    # Create experiment
    gt = FAULT_GROUND_TRUTHS[request.fault_type]
    experiment = ChaosExperiment(
        name=f"Quick Test: {gt.display_name}",
        description=f"Quick test of {request.fault_type}",
        fault_type=request.fault_type,
        fault_duration_seconds=request.fault_duration_seconds,
        tags=["quick-test", "api-triggered"]
    )
    
    _experiments[experiment.id] = experiment
    
    # Run experiment
    runner = ExperimentRunner()
    try:
        result = await runner.run_experiment(
            experiment=experiment,
            db=db,
            wait_for_analysis=request.wait_for_analysis,
            analysis_timeout_seconds=request.analysis_timeout_seconds
        )
        
        return _convert_result_to_response(result)
        
    except Exception as e:
        logger.error("Experiment failed", experiment_id=experiment.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment failed: {str(e)}"
        )


@router.post("/chaos/schedules", response_model=ExperimentScheduleResponse, tags=["Chaos Engineering"])
async def create_schedule(request: ExperimentScheduleCreate):
    """
    Create an experiment schedule.
    
    If fault_types is empty, all available fault types will be included.
    """
    from app.services.chaos.models import ChaosExperiment, ExperimentSchedule, FAULT_GROUND_TRUTHS
    
    # Determine which fault types to test
    fault_types = request.fault_types if request.fault_types else list(FAULT_GROUND_TRUTHS.keys())
    
    # Validate fault types
    invalid = [ft for ft in fault_types if ft not in FAULT_GROUND_TRUTHS]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown fault types: {invalid}"
        )
    
    # Create experiments
    experiments = []
    for fault_type in fault_types:
        gt = FAULT_GROUND_TRUTHS[fault_type]
        experiments.append(ChaosExperiment(
            name=f"Test: {gt.display_name}",
            description=gt.description,
            fault_type=fault_type,
            fault_duration_seconds=request.fault_duration_seconds,
            tags=["scheduled"]
        ))
    
    schedule = ExperimentSchedule(
        name=request.name,
        description=request.description,
        experiments=experiments,
        interval_between_experiments_seconds=request.interval_between_experiments_seconds
    )
    
    _schedules[schedule.id] = schedule
    
    logger.info(
        "Schedule created",
        schedule_id=schedule.id,
        total_experiments=len(experiments)
    )
    
    return ExperimentScheduleResponse(
        id=schedule.id,
        name=schedule.name,
        description=schedule.description,
        status=schedule.status.value,
        total_experiments=len(schedule.experiments),
        completed_experiments=len(schedule.results),
        started_at=schedule.started_at,
        completed_at=schedule.completed_at,
        aggregate_metrics=None,
        results=[]
    )


@router.post("/chaos/schedules/{schedule_id}/run", response_model=ExperimentScheduleResponse, tags=["Chaos Engineering"])
async def run_schedule(
    schedule_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Run a scheduled batch of experiments.
    
    This will execute all experiments in the schedule sequentially,
    waiting between each one, and return aggregate metrics.
    """
    from app.services.chaos.experiment_runner import ExperimentRunner
    
    schedule = _schedules.get(schedule_id)
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule not found: {schedule_id}"
        )
    
    runner = ExperimentRunner()
    
    try:
        updated_schedule = await runner.run_schedule(
            schedule=schedule,
            db=db
        )
        
        _schedules[schedule_id] = updated_schedule
        
        return _convert_schedule_to_response(updated_schedule)
        
    except Exception as e:
        logger.error("Schedule run failed", schedule_id=schedule_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schedule run failed: {str(e)}"
        )


@router.get("/chaos/schedules/{schedule_id}", response_model=ExperimentScheduleResponse, tags=["Chaos Engineering"])
async def get_schedule(schedule_id: str):
    """Get the status and results of an experiment schedule."""
    schedule = _schedules.get(schedule_id)
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule not found: {schedule_id}"
        )
    
    return _convert_schedule_to_response(schedule)


@router.get("/chaos/schedules/{schedule_id}/report", tags=["Chaos Engineering"])
async def get_schedule_report(
    schedule_id: str,
    format: str = Query("markdown", description="Report format: markdown or json")
):
    """
    Generate a comprehensive report for an experiment schedule.
    
    Returns a detailed report including:
    - Aggregate metrics
    - Individual experiment results
    - Failure analysis
    """
    from app.services.chaos.reporter import ExperimentReporter
    
    schedule = _schedules.get(schedule_id)
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule not found: {schedule_id}"
        )
    
    if not schedule.results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Schedule has no results yet. Run the schedule first."
        )
    
    reporter = ExperimentReporter()
    report = reporter.generate_schedule_report(schedule, format=format)
    
    if format == "json":
        return {"report": report, "format": "json"}
    
    return {"report": report, "format": "markdown"}


def _convert_result_to_response(result) -> ExperimentResultResponse:
    """Convert ExperimentResult to API response."""
    return ExperimentResultResponse(
        experiment_id=result.experiment_id,
        experiment_name=result.experiment_name,
        fault_type=result.fault_type,
        started_at=result.started_at,
        completed_at=result.completed_at,
        fault_duration_seconds=result.fault_duration_seconds,
        success=result.success,
        ground_truth=FaultGroundTruthResponse(
            fault_type=result.ground_truth.fault_type,
            display_name=result.ground_truth.display_name,
            description=result.ground_truth.description,
            expected_root_cause_keywords=result.ground_truth.expected_root_cause_keywords,
            expected_component=result.ground_truth.expected_component,
            expected_component_type=result.ground_truth.expected_component_type,
            expected_metrics=result.ground_truth.expected_metrics,
            expected_log_patterns=result.ground_truth.expected_log_patterns,
            expected_severity=result.ground_truth.expected_severity,
            max_detection_time_seconds=result.ground_truth.max_detection_time_seconds
        ),
        detection=DetectionResultResponse(
            incident_detected=result.detection.incident_detected,
            detection_time_seconds=result.detection.detection_time_seconds,
            within_threshold=result.detection.within_threshold,
            incident_id=result.detection.incident_id,
            incident_title=result.detection.incident_title,
            incident_severity=result.detection.incident_severity
        ),
        root_cause=RootCauseResultResponse(
            root_cause_identified=result.root_cause.root_cause_identified,
            identified_cause=result.root_cause.identified_cause,
            identified_component=result.root_cause.identified_component,
            identified_component_type=result.root_cause.identified_component_type,
            matches_ground_truth=result.root_cause.matches_ground_truth,
            component_match=result.root_cause.component_match,
            confidence=result.root_cause.confidence
        ),
        metrics=EvaluationMetricsResponse(
            precision=result.metrics.precision,
            recall=result.metrics.recall,
            f1_score=result.metrics.f1_score,
            true_positives=result.metrics.true_positives,
            false_positives=result.metrics.false_positives,
            false_negatives=result.metrics.false_negatives,
            mean_time_to_detection=result.metrics.mean_time_to_detection,
            min_time_to_detection=result.metrics.min_time_to_detection,
            max_time_to_detection=result.metrics.max_time_to_detection,
            root_cause_accuracy=result.metrics.root_cause_accuracy,
            component_accuracy=result.metrics.component_accuracy,
            root_cause_matches=result.metrics.root_cause_matches,
            component_matches=result.metrics.component_matches,
            total_evaluated=result.metrics.total_evaluated
        )
    )


def _convert_schedule_to_response(schedule) -> ExperimentScheduleResponse:
    """Convert ExperimentSchedule to API response."""
    metrics_response = None
    if schedule.aggregate_metrics:
        m = schedule.aggregate_metrics
        metrics_response = EvaluationMetricsResponse(
            precision=m.precision,
            recall=m.recall,
            f1_score=m.f1_score,
            true_positives=m.true_positives,
            false_positives=m.false_positives,
            false_negatives=m.false_negatives,
            mean_time_to_detection=m.mean_time_to_detection,
            min_time_to_detection=m.min_time_to_detection,
            max_time_to_detection=m.max_time_to_detection,
            root_cause_accuracy=m.root_cause_accuracy,
            component_accuracy=m.component_accuracy,
            root_cause_matches=m.root_cause_matches,
            component_matches=m.component_matches,
            total_evaluated=m.total_evaluated
        )
    
    results_response = [_convert_result_to_response(r) for r in schedule.results]
    
    return ExperimentScheduleResponse(
        id=schedule.id,
        name=schedule.name,
        description=schedule.description,
        status=schedule.status.value,
        total_experiments=len(schedule.experiments),
        completed_experiments=len(schedule.results),
        started_at=schedule.started_at,
        completed_at=schedule.completed_at,
        aggregate_metrics=metrics_response,
        results=results_response
    )

