"""
RCA Evaluator - Computes evaluation metrics for RCA correctness.

Metrics computed:
- Precision: TP / (TP + FP) - How accurate are our detections?
- Recall: TP / (TP + FN) - How many real incidents do we catch?
- Time-to-Detection (TTD): How fast do we detect incidents?
- Root Cause Accuracy: How often is the identified root cause correct?
- Component Accuracy: How often is the affected component correct?
"""

import structlog
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.services.chaos.models import (
    FaultGroundTruth,
    ChaosExperiment,
    DetectionResult,
    RootCauseResult,
    EvaluationMetrics,
    ExperimentResult,
    FAULT_GROUND_TRUTHS,
)

logger = structlog.get_logger()


class RCAEvaluator:
    """
    Evaluates RCA correctness against ground-truth definitions.
    
    This evaluator:
    1. Compares detected incidents against expected outcomes
    2. Validates root cause identification accuracy
    3. Computes precision/recall metrics
    4. Measures time-to-detection performance
    """
    
    def __init__(self):
        pass  # Stateless evaluator - all data passed via method arguments
    
    def evaluate_experiment(
        self,
        experiment: ChaosExperiment,
        incidents: List[Dict[str, Any]]
    ) -> ExperimentResult:
        """
        Evaluate a single chaos experiment.
        
        Args:
            experiment: The chaos experiment that was run
            incidents: List of incidents detected during the experiment
        
        Returns:
            ExperimentResult with all evaluation metrics
        """
        ground_truth = experiment.ground_truth
        if not ground_truth:
            logger.warning(
                "No ground truth for fault type",
                fault_type=experiment.fault_type
            )
            ground_truth = FaultGroundTruth(
                fault_type=experiment.fault_type,
                display_name=experiment.fault_type,
                description="Unknown fault type",
                expected_root_cause_keywords=[experiment.fault_type],
                expected_component="Unknown",
                expected_component_type="unknown"
            )
        
        # Evaluate detection
        detection = self._evaluate_detection(
            experiment=experiment,
            incidents=incidents,
            ground_truth=ground_truth
        )
        
        # Evaluate root cause analysis
        root_cause = self._evaluate_root_cause(
            incidents=incidents,
            ground_truth=ground_truth
        )
        
        # Compute metrics
        metrics = self._compute_experiment_metrics(
            detection=detection,
            root_cause=root_cause,
            ground_truth=ground_truth
        )
        
        # Get incident data for debugging
        incident_data = None
        agent_outputs = None
        if incidents:
            incident_data = incidents[0]
            agent_outputs = incidents[0].get("final_summary_json")
        
        return ExperimentResult(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            fault_type=experiment.fault_type,
            started_at=experiment.started_at or datetime.utcnow(),
            completed_at=experiment.completed_at or datetime.utcnow(),
            fault_duration_seconds=experiment.fault_duration_seconds,
            ground_truth=ground_truth,
            detection=detection,
            root_cause=root_cause,
            metrics=metrics,
            incident_data=incident_data,
            agent_outputs=agent_outputs
        )
    
    def _evaluate_detection(
        self,
        experiment: ChaosExperiment,
        incidents: List[Dict[str, Any]],
        ground_truth: FaultGroundTruth
    ) -> DetectionResult:
        """Evaluate incident detection for an experiment."""
        
        if not incidents:
            logger.info(
                "No incidents detected for experiment",
                experiment_id=experiment.id
            )
            return DetectionResult(incident_detected=False)
        
        # Find the first relevant incident
        relevant_incident = None
        for incident in incidents:
            # Check if incident occurred during the experiment window
            incident_time = incident.get("start_time") or incident.get("created_at")
            if incident_time:
                if isinstance(incident_time, str):
                    incident_time = datetime.fromisoformat(incident_time.replace("Z", "+00:00"))
                
                if experiment.started_at and incident_time < experiment.started_at:
                    continue
            
            relevant_incident = incident
            break
        
        if not relevant_incident:
            return DetectionResult(incident_detected=False)
        
        # Calculate time to detection
        detection_time_seconds = None
        if experiment.started_at:
            incident_time = relevant_incident.get("start_time") or relevant_incident.get("created_at")
            if isinstance(incident_time, str):
                incident_time = datetime.fromisoformat(incident_time.replace("Z", "+00:00"))
            
            if incident_time:
                # Make both timezone-naive for comparison
                if hasattr(incident_time, 'tzinfo') and incident_time.tzinfo is not None:
                    incident_time = incident_time.replace(tzinfo=None)
                started = experiment.started_at
                if hasattr(started, 'tzinfo') and started.tzinfo is not None:
                    started = started.replace(tzinfo=None)
                
                detection_time_seconds = (incident_time - started).total_seconds()
        
        # Determine if detection was within acceptable threshold
        # Negative detection times indicate timing issues (clock skew or race conditions)
        within_threshold = False
        if detection_time_seconds is not None and detection_time_seconds >= 0:
            within_threshold = detection_time_seconds <= ground_truth.max_detection_time_seconds
        
        return DetectionResult(
            incident_detected=True,
            detection_time_seconds=detection_time_seconds,
            within_threshold=within_threshold,
            incident_id=str(relevant_incident.get("id", "")),
            incident_title=relevant_incident.get("title"),
            incident_severity=relevant_incident.get("severity")
        )
    
    def _evaluate_root_cause(
        self,
        incidents: List[Dict[str, Any]],
        ground_truth: FaultGroundTruth
    ) -> RootCauseResult:
        """Evaluate root cause analysis accuracy."""
        
        if not incidents:
            return RootCauseResult(root_cause_identified=False)
        
        # Get the first incident with RCA data
        incident = incidents[0]
        summary = incident.get("final_summary_json") or {}
        
        # Extract root cause from summary
        identified_cause = summary.get("most_likely_cause")
        identified_component = None
        identified_component_type = None
        confidence = 0.0
        
        # Try to get from hypotheses
        hypotheses = summary.get("hypotheses", [])
        if hypotheses:
            top_hypothesis = hypotheses[0]
            if not identified_cause:
                identified_cause = top_hypothesis.get("cause")
            identified_component = top_hypothesis.get("component")
            identified_component_type = top_hypothesis.get("component_type")
            confidence = top_hypothesis.get("confidence", 0.0)
        
        # Also check probable_component from log analysis
        if not identified_component:
            identified_component = summary.get("probable_component")
        
        if not identified_cause:
            return RootCauseResult(root_cause_identified=False)
        
        # Check matches
        matches_ground_truth = ground_truth.matches_root_cause(identified_cause)
        component_match = ground_truth.matches_component(identified_component)
        
        return RootCauseResult(
            root_cause_identified=True,
            identified_cause=identified_cause,
            identified_component=identified_component,
            identified_component_type=identified_component_type,
            matches_ground_truth=matches_ground_truth,
            component_match=component_match,
            confidence=confidence
        )
    
    def _compute_experiment_metrics(
        self,
        detection: DetectionResult,
        root_cause: RootCauseResult,
        ground_truth: FaultGroundTruth
    ) -> EvaluationMetrics:
        """Compute metrics for a single experiment."""
        metrics = EvaluationMetrics()
        
        # Detection metrics
        if detection.incident_detected:
            metrics.true_positives = 1
            if detection.detection_time_seconds is not None:
                metrics.detection_times = [detection.detection_time_seconds]
        else:
            metrics.false_negatives = 1
        
        # RCA metrics
        metrics.total_evaluated = 1
        if root_cause.matches_ground_truth:
            metrics.root_cause_matches = 1
        if root_cause.component_match:
            metrics.component_matches = 1
        
        metrics.compute_time_metrics()
        
        return metrics
    
    def aggregate_metrics(
        self,
        results: List[ExperimentResult]
    ) -> EvaluationMetrics:
        """
        Aggregate metrics across multiple experiment results.
        
        Args:
            results: List of experiment results to aggregate
        
        Returns:
            Aggregated EvaluationMetrics
        """
        aggregated = EvaluationMetrics()
        
        for result in results:
            m = result.metrics
            aggregated.true_positives += m.true_positives
            aggregated.false_positives += m.false_positives
            aggregated.false_negatives += m.false_negatives
            aggregated.detection_times.extend(m.detection_times)
            aggregated.root_cause_matches += m.root_cause_matches
            aggregated.component_matches += m.component_matches
            aggregated.total_evaluated += m.total_evaluated
        
        aggregated.compute_time_metrics()
        
        return aggregated
    
    def evaluate_false_positives(
        self,
        incidents: List[Dict[str, Any]],
        experiment_window_start: datetime,
        experiment_window_end: datetime
    ) -> int:
        """
        Count false positive incidents (detected when no fault was injected).
        
        Args:
            incidents: All incidents in a time window
            experiment_window_start: Start of the experiment window
            experiment_window_end: End of the experiment window
        
        Returns:
            Number of false positive incidents
        """
        false_positives = 0
        
        for incident in incidents:
            incident_time = incident.get("start_time") or incident.get("created_at")
            if isinstance(incident_time, str):
                incident_time = datetime.fromisoformat(incident_time.replace("Z", "+00:00"))
            
            if incident_time:
                # Make timezone-naive
                if hasattr(incident_time, 'tzinfo') and incident_time.tzinfo is not None:
                    incident_time = incident_time.replace(tzinfo=None)
                
                # Incidents outside the fault window are false positives
                if incident_time < experiment_window_start or incident_time > experiment_window_end:
                    false_positives += 1
        
        return false_positives
    
    def compute_precision_recall(
        self,
        total_faults_injected: int,
        incidents_detected: int,
        correct_detections: int,
        false_positives: int
    ) -> Dict[str, float]:
        """
        Compute precision and recall metrics.
        
        Args:
            total_faults_injected: Total number of faults that were injected
            incidents_detected: Total incidents that were detected
            correct_detections: Incidents that correctly matched injected faults
            false_positives: Incidents detected without a corresponding fault
        
        Returns:
            Dictionary with precision, recall, and F1 score
        """
        true_positives = correct_detections
        false_negatives = total_faults_injected - correct_detections
        
        precision = 0.0
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        
        recall = 0.0
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

