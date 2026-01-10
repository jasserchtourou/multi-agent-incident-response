"""
Tests for the Chaos Engineering and RCA Evaluation System.

Tests cover:
- Ground-truth definitions
- RCA evaluation metrics
- Experiment models
- Reporting
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.services.chaos.models import (
    FaultGroundTruth,
    ChaosExperiment,
    ExperimentStatus,
    EvaluationMetrics,
    DetectionResult,
    RootCauseResult,
    ExperimentResult,
    ExperimentSchedule,
    FAULT_GROUND_TRUTHS,
    get_ground_truth,
    list_available_faults,
)
from app.services.chaos.evaluator import RCAEvaluator
from app.services.chaos.reporter import ExperimentReporter


class TestFaultGroundTruth:
    """Tests for fault ground-truth definitions."""
    
    def test_all_fault_types_defined(self):
        """Test that expected fault types are defined."""
        expected_faults = [
            "latency_spike", "error_rate", "db_slow", 
            "memory_leak", "dependency_down"
        ]
        for fault in expected_faults:
            assert fault in FAULT_GROUND_TRUTHS
    
    def test_ground_truth_has_required_fields(self):
        """Test that each ground truth has required fields."""
        for fault_type, gt in FAULT_GROUND_TRUTHS.items():
            assert gt.fault_type == fault_type
            assert gt.display_name
            assert gt.description
            assert gt.expected_root_cause_keywords
            assert gt.expected_component
            assert gt.expected_component_type
    
    def test_matches_root_cause(self):
        """Test root cause matching."""
        gt = FAULT_GROUND_TRUTHS["error_rate"]
        
        assert gt.matches_root_cause("Database error rate spike")
        assert gt.matches_root_cause("Application failure detected")
        assert not gt.matches_root_cause("Network timeout")
    
    def test_matches_component(self):
        """Test component matching."""
        gt = FAULT_GROUND_TRUTHS["db_slow"]
        
        assert gt.matches_component("Database")
        assert gt.matches_component("database")
        assert not gt.matches_component("Application")
    
    def test_get_ground_truth(self):
        """Test getting ground truth by fault type."""
        gt = get_ground_truth("latency_spike")
        assert gt is not None
        assert gt.fault_type == "latency_spike"
        
        assert get_ground_truth("unknown") is None
    
    def test_list_available_faults(self):
        """Test listing available faults."""
        faults = list_available_faults()
        assert len(faults) >= 5
        
        for fault in faults:
            assert "fault_type" in fault
            assert "display_name" in fault
            assert "description" in fault


class TestChaosExperiment:
    """Tests for ChaosExperiment model."""
    
    def test_create_experiment(self):
        """Test creating an experiment."""
        exp = ChaosExperiment(
            name="Test Experiment",
            fault_type="error_rate",
            fault_duration_seconds=60
        )
        
        assert exp.id
        assert exp.name == "Test Experiment"
        assert exp.fault_type == "error_rate"
        assert exp.status == ExperimentStatus.PENDING
    
    def test_experiment_ground_truth(self):
        """Test getting ground truth from experiment."""
        exp = ChaosExperiment(
            name="Test",
            fault_type="latency_spike"
        )
        
        gt = exp.ground_truth
        assert gt is not None
        assert gt.fault_type == "latency_spike"
    
    def test_experiment_is_overdue(self):
        """Test overdue detection."""
        # Not overdue - no scheduled time
        exp1 = ChaosExperiment(name="Test", fault_type="error_rate")
        assert not exp1.is_overdue()
        
        # Overdue - scheduled in past
        exp2 = ChaosExperiment(
            name="Test",
            fault_type="error_rate",
            scheduled_at=datetime.utcnow() - timedelta(hours=1)
        )
        assert exp2.is_overdue()
        
        # Not overdue - already running
        exp3 = ChaosExperiment(
            name="Test",
            fault_type="error_rate",
            scheduled_at=datetime.utcnow() - timedelta(hours=1),
            status=ExperimentStatus.RUNNING
        )
        assert not exp3.is_overdue()


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics computation."""
    
    def test_precision_calculation(self):
        """Test precision calculation."""
        metrics = EvaluationMetrics(true_positives=8, false_positives=2)
        assert metrics.precision == 0.8
    
    def test_recall_calculation(self):
        """Test recall calculation."""
        metrics = EvaluationMetrics(true_positives=7, false_negatives=3)
        assert metrics.recall == 0.7
    
    def test_f1_score(self):
        """Test F1 score calculation."""
        metrics = EvaluationMetrics(
            true_positives=8,
            false_positives=2,
            false_negatives=2
        )
        # precision = 0.8, recall = 0.8
        assert abs(metrics.f1_score - 0.8) < 0.01
    
    def test_zero_division_handling(self):
        """Test handling of zero division."""
        metrics = EvaluationMetrics()
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
    
    def test_root_cause_accuracy(self):
        """Test root cause accuracy calculation."""
        metrics = EvaluationMetrics(
            root_cause_matches=7,
            total_evaluated=10
        )
        assert metrics.root_cause_accuracy == 0.7
    
    def test_time_metrics_computation(self):
        """Test time metrics computation."""
        metrics = EvaluationMetrics(
            detection_times=[10.0, 20.0, 30.0]
        )
        metrics.compute_time_metrics()
        
        assert metrics.mean_time_to_detection == 20.0
        assert metrics.min_time_to_detection == 10.0
        assert metrics.max_time_to_detection == 30.0


class TestRCAEvaluator:
    """Tests for RCAEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        return RCAEvaluator()
    
    @pytest.fixture
    def sample_experiment(self):
        return ChaosExperiment(
            name="Test Experiment",
            fault_type="error_rate",
            fault_duration_seconds=60,
            started_at=datetime.utcnow()
        )
    
    @pytest.fixture
    def sample_incidents(self):
        return [
            {
                "id": "test-incident-1",
                "title": "Error Rate Spike Detected",
                "status": "RESOLVED",
                "severity": "SEV1",
                "start_time": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat(),
                "final_summary_json": {
                    "most_likely_cause": "Application error rate increased due to database failures",
                    "hypotheses": [
                        {
                            "cause": "Database error causing application failures",
                            "confidence": 0.85,
                            "component": "Application",
                            "component_type": "service"
                        }
                    ]
                }
            }
        ]
    
    def test_evaluate_experiment(self, evaluator, sample_experiment, sample_incidents):
        """Test full experiment evaluation."""
        result = evaluator.evaluate_experiment(
            experiment=sample_experiment,
            incidents=sample_incidents
        )
        
        assert result.experiment_id == sample_experiment.id
        assert result.fault_type == "error_rate"
        assert result.detection.incident_detected
        assert result.root_cause.root_cause_identified
    
    def test_evaluate_no_incidents(self, evaluator, sample_experiment):
        """Test evaluation with no incidents."""
        result = evaluator.evaluate_experiment(
            experiment=sample_experiment,
            incidents=[]
        )
        
        assert not result.detection.incident_detected
        assert not result.root_cause.root_cause_identified
    
    def test_root_cause_matching(self, evaluator, sample_experiment, sample_incidents):
        """Test root cause matching against ground truth."""
        result = evaluator.evaluate_experiment(
            experiment=sample_experiment,
            incidents=sample_incidents
        )
        
        # The sample incident mentions "error" which should match error_rate ground truth
        assert result.root_cause.matches_ground_truth
    
    def test_aggregate_metrics(self, evaluator):
        """Test aggregating metrics from multiple results."""
        results = []
        for i in range(3):
            exp = ChaosExperiment(
                name=f"Test {i}",
                fault_type="error_rate",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            result = ExperimentResult(
                experiment_id=exp.id,
                experiment_name=exp.name,
                fault_type=exp.fault_type,
                started_at=exp.started_at,
                completed_at=exp.completed_at or datetime.utcnow(),
                fault_duration_seconds=60,
                ground_truth=FAULT_GROUND_TRUTHS["error_rate"],
                detection=DetectionResult(
                    incident_detected=True,
                    detection_time_seconds=15.0 + i * 5,
                    within_threshold=True
                ),
                root_cause=RootCauseResult(
                    root_cause_identified=True,
                    matches_ground_truth=i < 2,  # 2/3 match
                    component_match=i < 2
                ),
                metrics=EvaluationMetrics(
                    true_positives=1,
                    detection_times=[15.0 + i * 5],
                    root_cause_matches=1 if i < 2 else 0,
                    component_matches=1 if i < 2 else 0,
                    total_evaluated=1
                )
            )
            results.append(result)
        
        aggregated = evaluator.aggregate_metrics(results)
        
        assert aggregated.true_positives == 3
        assert aggregated.total_evaluated == 3
        assert len(aggregated.detection_times) == 3


class TestExperimentReporter:
    """Tests for ExperimentReporter."""
    
    @pytest.fixture
    def reporter(self):
        return ExperimentReporter()
    
    @pytest.fixture
    def sample_result(self):
        return ExperimentResult(
            experiment_id="test-123",
            experiment_name="Test Experiment",
            fault_type="error_rate",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            fault_duration_seconds=60,
            ground_truth=FAULT_GROUND_TRUTHS["error_rate"],
            detection=DetectionResult(
                incident_detected=True,
                detection_time_seconds=25.0,
                within_threshold=True,
                incident_id="inc-1",
                incident_title="Error Rate Spike"
            ),
            root_cause=RootCauseResult(
                root_cause_identified=True,
                identified_cause="Application error rate increased",
                identified_component="Application",
                matches_ground_truth=True,
                component_match=True,
                confidence=0.85
            ),
            metrics=EvaluationMetrics(
                true_positives=1,
                detection_times=[25.0],
                root_cause_matches=1,
                component_matches=1,
                total_evaluated=1
            )
        )
    
    def test_generate_markdown_report(self, reporter, sample_result):
        """Test generating a markdown report."""
        report = reporter.generate_experiment_report(sample_result, format="markdown")
        
        assert "# Chaos Experiment Report" in report
        assert "Test Experiment" in report
        assert "error_rate" in report
        assert "✅ SUCCESS" in report
    
    def test_generate_json_report(self, reporter, sample_result):
        """Test generating a JSON report."""
        import json
        
        report = reporter.generate_experiment_report(sample_result, format="json")
        data = json.loads(report)
        
        assert data["experiment_id"] == "test-123"
        assert data["success"] == True
        assert data["detection"]["incident_detected"] == True
    
    def test_generate_metrics_summary(self, reporter):
        """Test generating metrics summary."""
        metrics = EvaluationMetrics(
            true_positives=8,
            false_positives=2,
            false_negatives=1,
            detection_times=[10.0, 20.0, 30.0],
            root_cause_matches=7,
            component_matches=6,
            total_evaluated=10
        )
        metrics.compute_time_metrics()
        
        summary = reporter.generate_metrics_summary(metrics)
        
        assert summary["detection"]["precision"] == 0.8
        assert summary["detection"]["recall"] == pytest.approx(0.889, rel=0.01)
        assert summary["time_to_detection"]["mean_seconds"] == 20.0
        assert summary["rca_accuracy"]["root_cause_accuracy"] == 0.7


class TestExperimentSchedule:
    """Tests for ExperimentSchedule."""
    
    def test_create_schedule(self):
        """Test creating a schedule."""
        experiments = [
            ChaosExperiment(name=f"Test {i}", fault_type="error_rate")
            for i in range(3)
        ]
        
        schedule = ExperimentSchedule(
            name="Test Schedule",
            experiments=experiments
        )
        
        assert schedule.id
        assert len(schedule.experiments) == 3
        assert schedule.status == ExperimentStatus.PENDING
    
    def test_schedule_tracking(self):
        """Test schedule progress tracking."""
        schedule = ExperimentSchedule(
            name="Test Schedule",
            experiments=[
                ChaosExperiment(name="Test 1", fault_type="error_rate"),
                ChaosExperiment(name="Test 2", fault_type="latency_spike")
            ]
        )
        
        assert schedule.current_experiment_index == 0
        
        schedule.current_experiment_index = 1
        assert schedule.current_experiment_index == 1


class TestIntegration:
    """Integration tests for the chaos engineering system."""
    
    def test_full_evaluation_flow(self):
        """Test the full evaluation flow without actual fault injection."""
        from app.services.chaos.models import ChaosExperiment, FAULT_GROUND_TRUTHS
        from app.services.chaos.evaluator import RCAEvaluator
        from app.services.chaos.reporter import ExperimentReporter
        
        # Create experiment
        experiment = ChaosExperiment(
            name="Integration Test",
            fault_type="db_slow",
            fault_duration_seconds=60,
            started_at=datetime.utcnow()
        )
        
        # Simulate incident
        incidents = [
            {
                "id": "test-inc",
                "title": "Database Latency Spike",
                "status": "RESOLVED",
                "severity": "SEV2",
                "start_time": (datetime.utcnow() + timedelta(seconds=10)).isoformat(),
                "final_summary_json": {
                    "most_likely_cause": "Database query slowdown due to missing indexes",
                    "hypotheses": [
                        {
                            "cause": "Slow database queries",
                            "confidence": 0.90,
                            "component": "Database",
                            "component_type": "database"
                        }
                    ]
                }
            }
        ]
        
        # Evaluate
        evaluator = RCAEvaluator()
        result = evaluator.evaluate_experiment(experiment, incidents)
        
        # Generate report
        reporter = ExperimentReporter()
        markdown = reporter.generate_experiment_report(result, format="markdown")
        json_report = reporter.generate_experiment_report(result, format="json")
        
        # Assertions
        assert result.detection.incident_detected
        assert result.root_cause.matches_ground_truth
        assert result.root_cause.component_match
        assert result.success
        assert "Database" in markdown
        assert "db_slow" in json_report

