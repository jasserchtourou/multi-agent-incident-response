"""
Chaos Experiment Runner - Executes and manages chaos experiments.

Provides:
- Single experiment execution
- Scheduled experiment batches
- Automatic validation after experiments
- Integration with demo service for fault injection
"""

import asyncio
import structlog
import httpx
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Incident, IncidentStatus
from app.services.chaos.models import (
    ChaosExperiment,
    ExperimentStatus,
    ExperimentResult,
    ExperimentSchedule,
    EvaluationMetrics,
    FAULT_GROUND_TRUTHS,
)
from app.services.chaos.evaluator import RCAEvaluator

logger = structlog.get_logger()


class ExperimentRunner:
    """
    Runs chaos experiments and validates results.
    
    This runner:
    1. Injects faults via the demo service
    2. Waits for detection and analysis
    3. Retrieves incidents created during the experiment
    4. Evaluates RCA correctness against ground truth
    """
    
    def __init__(
        self,
        demo_service_url: str = None,
        evaluator: RCAEvaluator = None
    ):
        self.demo_service_url = demo_service_url or settings.DEMO_SERVICE_URL
        self.evaluator = evaluator or RCAEvaluator()
        self._running_experiments: Dict[str, ChaosExperiment] = {}
    
    async def run_experiment(
        self,
        experiment: ChaosExperiment,
        db: AsyncSession,
        wait_for_analysis: bool = True,
        analysis_timeout_seconds: int = 120
    ) -> ExperimentResult:
        """
        Run a single chaos experiment.
        
        Args:
            experiment: The experiment to run
            db: Database session for querying incidents
            wait_for_analysis: Whether to wait for RCA to complete
            analysis_timeout_seconds: Max time to wait for analysis
        
        Returns:
            ExperimentResult with evaluation metrics
        """
        logger.info(
            "Starting chaos experiment",
            experiment_id=experiment.id,
            fault_type=experiment.fault_type,
            duration=experiment.fault_duration_seconds
        )
        
        # Update experiment status
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow()
        self._running_experiments[experiment.id] = experiment
        
        try:
            # Inject fault
            fault_response = await self._inject_fault(
                fault_type=experiment.fault_type,
                duration_seconds=experiment.fault_duration_seconds
            )
            
            if not fault_response.get("success"):
                logger.error(
                    "Failed to inject fault",
                    experiment_id=experiment.id,
                    response=fault_response
                )
                experiment.status = ExperimentStatus.FAILED
                raise RuntimeError(f"Fault injection failed: {fault_response}")
            
            logger.info(
                "Fault injected successfully",
                experiment_id=experiment.id,
                fault_type=experiment.fault_type
            )
            
            # Wait for fault duration + buffer for detection
            detection_wait = experiment.fault_duration_seconds + 30
            logger.info(
                "Waiting for detection",
                experiment_id=experiment.id,
                wait_seconds=detection_wait
            )
            await asyncio.sleep(detection_wait)
            
            # If waiting for analysis, poll for completion
            if wait_for_analysis:
                await self._wait_for_analysis(
                    experiment=experiment,
                    db=db,
                    timeout_seconds=analysis_timeout_seconds
                )
            
            # Get incidents created during experiment
            incidents = await self._get_experiment_incidents(
                experiment=experiment,
                db=db
            )
            
            # Update experiment with incident IDs
            experiment.incident_ids = [str(i.get("id")) for i in incidents]
            experiment.completed_at = datetime.utcnow()
            experiment.status = ExperimentStatus.COMPLETED
            
            # Evaluate results
            result = self.evaluator.evaluate_experiment(
                experiment=experiment,
                incidents=incidents
            )
            
            logger.info(
                "Experiment completed",
                experiment_id=experiment.id,
                incidents_detected=len(incidents),
                success=result.success
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Experiment failed",
                experiment_id=experiment.id,
                error=str(e)
            )
            experiment.status = ExperimentStatus.FAILED
            experiment.completed_at = datetime.utcnow()
            raise
        
        finally:
            self._running_experiments.pop(experiment.id, None)
    
    async def run_schedule(
        self,
        schedule: ExperimentSchedule,
        db: AsyncSession,
        on_experiment_complete: Optional[Callable[[ExperimentResult], None]] = None
    ) -> ExperimentSchedule:
        """
        Run a scheduled batch of experiments.
        
        Args:
            schedule: The experiment schedule to run
            db: Database session
            on_experiment_complete: Callback for each completed experiment
        
        Returns:
            Updated ExperimentSchedule with results
        """
        logger.info(
            "Starting experiment schedule",
            schedule_id=schedule.id,
            total_experiments=len(schedule.experiments)
        )
        
        schedule.status = ExperimentStatus.RUNNING
        schedule.started_at = datetime.utcnow()
        
        for i, experiment in enumerate(schedule.experiments):
            logger.info(
                "Running experiment",
                schedule_id=schedule.id,
                experiment_index=i + 1,
                total=len(schedule.experiments),
                fault_type=experiment.fault_type
            )
            
            try:
                result = await self.run_experiment(
                    experiment=experiment,
                    db=db
                )
                schedule.results.append(result)
                schedule.current_experiment_index = i + 1
                
                if on_experiment_complete:
                    on_experiment_complete(result)
                
                # Wait between experiments if not the last one
                if i < len(schedule.experiments) - 1:
                    logger.info(
                        "Waiting before next experiment",
                        wait_seconds=schedule.interval_between_experiments_seconds
                    )
                    await asyncio.sleep(schedule.interval_between_experiments_seconds)
                    
            except Exception as e:
                logger.error(
                    "Experiment in schedule failed",
                    schedule_id=schedule.id,
                    experiment_id=experiment.id,
                    error=str(e)
                )
                # Continue with next experiment
        
        # Aggregate metrics
        schedule.aggregate_metrics = self.evaluator.aggregate_metrics(schedule.results)
        schedule.status = ExperimentStatus.COMPLETED
        schedule.completed_at = datetime.utcnow()
        
        logger.info(
            "Schedule completed",
            schedule_id=schedule.id,
            total_results=len(schedule.results),
            precision=schedule.aggregate_metrics.precision,
            recall=schedule.aggregate_metrics.recall
        )
        
        return schedule
    
    async def _inject_fault(
        self,
        fault_type: str,
        duration_seconds: int
    ) -> Dict[str, Any]:
        """Inject a fault via the demo service."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.demo_service_url}/admin/fault",
                    params={
                        "type": fault_type,
                        "duration": duration_seconds
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                return {"success": False, "error": str(e)}
            except httpx.HTTPStatusError as e:
                return {"success": False, "error": e.response.text}
    
    async def _wait_for_analysis(
        self,
        experiment: ChaosExperiment,
        db: AsyncSession,
        timeout_seconds: int
    ) -> None:
        """Wait for RCA analysis to complete on detected incidents."""
        start_time = datetime.utcnow()
        poll_interval = 5  # seconds
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            # Check for resolved incidents
            query = select(Incident).where(
                and_(
                    Incident.status == IncidentStatus.RESOLVED,
                    Incident.created_at >= experiment.started_at
                )
            )
            result = await db.execute(query)
            resolved = result.scalars().all()
            
            if resolved:
                logger.info(
                    "Analysis completed",
                    experiment_id=experiment.id,
                    resolved_count=len(resolved)
                )
                return
            
            await asyncio.sleep(poll_interval)
        
        logger.warning(
            "Analysis wait timeout",
            experiment_id=experiment.id,
            timeout_seconds=timeout_seconds
        )
    
    async def _get_experiment_incidents(
        self,
        experiment: ChaosExperiment,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get all incidents created during the experiment window."""
        if not experiment.started_at:
            return []
        
        # Query incidents created after experiment start
        query = select(Incident).where(
            Incident.created_at >= experiment.started_at
        ).order_by(Incident.created_at)
        
        result = await db.execute(query)
        incidents = result.scalars().all()
        
        return [
            {
                "id": str(incident.id),
                "title": incident.title,
                "status": incident.status.value,
                "severity": incident.severity.value,
                "start_time": incident.start_time.isoformat() if incident.start_time else None,
                "created_at": incident.created_at.isoformat() if incident.created_at else None,
                "end_time": incident.end_time.isoformat() if incident.end_time else None,
                "signals_json": incident.signals_json,
                "final_summary_json": incident.final_summary_json,
                "rca_markdown": incident.rca_markdown,
            }
            for incident in incidents
        ]
    
    def create_quick_experiment(
        self,
        fault_type: str,
        duration_seconds: int = 60,
        name: Optional[str] = None
    ) -> ChaosExperiment:
        """Create a quick experiment for a fault type."""
        ground_truth = FAULT_GROUND_TRUTHS.get(fault_type)
        
        return ChaosExperiment(
            name=name or f"Quick Test: {ground_truth.display_name if ground_truth else fault_type}",
            description=f"Quick test of {fault_type} fault",
            fault_type=fault_type,
            fault_duration_seconds=duration_seconds,
            tags=["quick-test"]
        )
    
    def create_full_test_schedule(
        self,
        name: str = "Full Fault Coverage Test",
        duration_per_fault: int = 60,
        interval_seconds: int = 120
    ) -> ExperimentSchedule:
        """Create a schedule that tests all fault types."""
        experiments = []
        
        for fault_type, ground_truth in FAULT_GROUND_TRUTHS.items():
            experiments.append(ChaosExperiment(
                name=f"Test: {ground_truth.display_name}",
                description=ground_truth.description,
                fault_type=fault_type,
                fault_duration_seconds=duration_per_fault,
                tags=["full-coverage"]
            ))
        
        return ExperimentSchedule(
            name=name,
            description="Comprehensive test of all fault types",
            experiments=experiments,
            interval_between_experiments_seconds=interval_seconds
        )
    
    def get_running_experiments(self) -> List[ChaosExperiment]:
        """Get list of currently running experiments."""
        return list(self._running_experiments.values())
    
    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment."""
        experiment = self._running_experiments.get(experiment_id)
        if experiment:
            experiment.status = ExperimentStatus.CANCELLED
            experiment.completed_at = datetime.utcnow()
            self._running_experiments.pop(experiment_id, None)
            logger.info("Experiment cancelled", experiment_id=experiment_id)
            return True
        return False

