"""
Chaos Engineering Models and Ground-Truth Definitions.

Defines the expected outcomes for each fault type to enable
automated validation of RCA correctness.
"""

import enum
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ExperimentStatus(str, enum.Enum):
    """Status of a chaos experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FaultGroundTruth(BaseModel):
    """
    Ground-truth definition for a fault type.
    
    Defines the expected root cause, affected components, and
    signals that should be detected when this fault is injected.
    """
    fault_type: str = Field(description="The fault type identifier")
    display_name: str = Field(description="Human-readable fault name")
    description: str = Field(description="Description of what this fault simulates")
    
    # Expected root cause identification
    expected_root_cause_keywords: List[str] = Field(
        description="Keywords that should appear in the identified root cause"
    )
    expected_component: str = Field(
        description="The component that should be identified as the root cause"
    )
    expected_component_type: str = Field(
        description="The type of component (database, service, cache, etc.)"
    )
    
    # Expected signals
    expected_metrics: List[str] = Field(
        default_factory=list,
        description="Metrics that should show anomalies"
    )
    expected_log_patterns: List[str] = Field(
        default_factory=list,
        description="Log patterns that should be detected"
    )
    
    # Severity expectations
    expected_severity: str = Field(
        default="SEV2",
        description="Expected incident severity"
    )
    
    # Detection expectations
    max_detection_time_seconds: float = Field(
        default=60.0,
        description="Maximum acceptable time to detect the incident"
    )
    
    def matches_root_cause(self, identified_cause: str) -> bool:
        """Check if the identified root cause matches the ground truth."""
        if not identified_cause:
            return False
        cause_lower = identified_cause.lower()
        return any(
            keyword.lower() in cause_lower 
            for keyword in self.expected_root_cause_keywords
        )
    
    def matches_component(self, identified_component: Optional[str]) -> bool:
        """Check if the identified component matches the ground truth."""
        if not identified_component:
            return False
        return identified_component.lower() == self.expected_component.lower()


# Ground-truth definitions for each fault type
FAULT_GROUND_TRUTHS: Dict[str, FaultGroundTruth] = {
    "latency_spike": FaultGroundTruth(
        fault_type="latency_spike",
        display_name="Latency Spike",
        description="Simulates slow response times across the application",
        expected_root_cause_keywords=[
            "latency", "slow", "delay", "response time", "performance"
        ],
        expected_component="Application",
        expected_component_type="service",
        expected_metrics=["latency_p95_ms", "latency_p99_ms", "response_time"],
        expected_log_patterns=["timeout", "slow", "delayed"],
        expected_severity="SEV2",
        max_detection_time_seconds=60.0
    ),
    
    "error_rate": FaultGroundTruth(
        fault_type="error_rate",
        display_name="Error Rate Spike",
        description="Simulates elevated HTTP error rates (5xx errors)",
        expected_root_cause_keywords=[
            "error", "failure", "500", "exception", "crash"
        ],
        expected_component="Application",
        expected_component_type="service",
        expected_metrics=["error_rate", "http_5xx_count", "exception_count"],
        expected_log_patterns=["error", "exception", "failed", "500"],
        expected_severity="SEV1",
        max_detection_time_seconds=30.0
    ),
    
    "db_slow": FaultGroundTruth(
        fault_type="db_slow",
        display_name="Database Slowdown",
        description="Simulates slow database queries causing cascading latency",
        expected_root_cause_keywords=[
            "database", "db", "query", "sql", "connection pool", "slow query"
        ],
        expected_component="Database",
        expected_component_type="database",
        expected_metrics=["db_latency_ms", "query_time_ms", "connection_pool_usage"],
        expected_log_patterns=["database", "query", "timeout", "connection"],
        expected_severity="SEV2",
        max_detection_time_seconds=45.0
    ),
    
    "memory_leak": FaultGroundTruth(
        fault_type="memory_leak",
        display_name="Memory Leak",
        description="Simulates gradual memory consumption increase",
        expected_root_cause_keywords=[
            "memory", "leak", "oom", "heap", "gc", "garbage"
        ],
        expected_component="Application",
        expected_component_type="service",
        expected_metrics=["memory_usage_mb", "heap_usage", "gc_pause_ms"],
        expected_log_patterns=["memory", "heap", "oom", "gc"],
        expected_severity="SEV2",
        max_detection_time_seconds=90.0
    ),
    
    "dependency_down": FaultGroundTruth(
        fault_type="dependency_down",
        display_name="External Dependency Down",
        description="Simulates an external service being unavailable",
        expected_root_cause_keywords=[
            "dependency", "external", "downstream", "service", "unavailable",
            "connection refused", "unreachable"
        ],
        expected_component="External Service",
        expected_component_type="external_dependency",
        expected_metrics=["dependency_error_rate", "connection_failures"],
        expected_log_patterns=[
            "connection refused", "unreachable", "timeout", "external"
        ],
        expected_severity="SEV1",
        max_detection_time_seconds=30.0
    ),
    
    "cpu_spike": FaultGroundTruth(
        fault_type="cpu_spike",
        display_name="CPU Spike",
        description="Simulates high CPU utilization",
        expected_root_cause_keywords=[
            "cpu", "processor", "compute", "utilization", "load"
        ],
        expected_component="Application",
        expected_component_type="service",
        expected_metrics=["cpu_usage_percent", "system_load"],
        expected_log_patterns=["cpu", "high load", "throttling"],
        expected_severity="SEV3",
        max_detection_time_seconds=60.0
    ),
    
    "cache_miss": FaultGroundTruth(
        fault_type="cache_miss",
        display_name="Cache Miss Storm",
        description="Simulates cache failures causing increased database load",
        expected_root_cause_keywords=[
            "cache", "redis", "miss", "eviction", "hit rate"
        ],
        expected_component="Cache",
        expected_component_type="cache",
        expected_metrics=["cache_hit_rate", "cache_miss_count", "redis_latency_ms"],
        expected_log_patterns=["cache", "miss", "redis"],
        expected_severity="SEV3",
        max_detection_time_seconds=60.0
    ),
}


class ChaosExperiment(BaseModel):
    """
    A chaos experiment definition.
    
    Represents a scheduled or running chaos experiment with its
    configuration and expected outcomes.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Human-readable experiment name")
    description: Optional[str] = None
    
    # Fault configuration
    fault_type: str = Field(description="Type of fault to inject")
    fault_duration_seconds: int = Field(
        default=60,
        description="How long the fault should last"
    )
    
    # Scheduling
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Status
    status: ExperimentStatus = Field(default=ExperimentStatus.PENDING)
    
    # Results
    incident_ids: List[str] = Field(
        default_factory=list,
        description="IDs of incidents created during the experiment"
    )
    
    # Tags for organization
    tags: List[str] = Field(default_factory=list)
    
    @property
    def ground_truth(self) -> Optional[FaultGroundTruth]:
        """Get the ground-truth definition for this experiment's fault type."""
        return FAULT_GROUND_TRUTHS.get(self.fault_type)
    
    def is_overdue(self) -> bool:
        """Check if a scheduled experiment is overdue."""
        if self.status != ExperimentStatus.PENDING:
            return False
        if not self.scheduled_at:
            return False
        return datetime.utcnow() > self.scheduled_at


class DetectionResult(BaseModel):
    """Result of incident detection evaluation."""
    incident_detected: bool = False
    detection_time_seconds: Optional[float] = None
    within_threshold: bool = False
    incident_id: Optional[str] = None
    incident_title: Optional[str] = None
    incident_severity: Optional[str] = None


class RootCauseResult(BaseModel):
    """Result of root cause analysis evaluation."""
    root_cause_identified: bool = False
    identified_cause: Optional[str] = None
    identified_component: Optional[str] = None
    identified_component_type: Optional[str] = None
    matches_ground_truth: bool = False
    component_match: bool = False
    confidence: float = 0.0


class EvaluationMetrics(BaseModel):
    """
    RCA evaluation metrics for an experiment or set of experiments.
    """
    # Detection metrics
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Time metrics
    detection_times: List[float] = Field(default_factory=list)
    mean_time_to_detection: Optional[float] = None
    max_time_to_detection: Optional[float] = None
    min_time_to_detection: Optional[float] = None
    
    # RCA accuracy metrics
    root_cause_matches: int = 0
    component_matches: int = 0
    total_evaluated: int = 0
    
    # Computed metrics
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def root_cause_accuracy(self) -> float:
        """Percentage of root causes correctly identified."""
        if self.total_evaluated == 0:
            return 0.0
        return self.root_cause_matches / self.total_evaluated
    
    @property
    def component_accuracy(self) -> float:
        """Percentage of components correctly identified."""
        if self.total_evaluated == 0:
            return 0.0
        return self.component_matches / self.total_evaluated
    
    def compute_time_metrics(self) -> None:
        """Compute time-based metrics from detection times."""
        if self.detection_times:
            self.mean_time_to_detection = sum(self.detection_times) / len(self.detection_times)
            self.max_time_to_detection = max(self.detection_times)
            self.min_time_to_detection = min(self.detection_times)


class ExperimentResult(BaseModel):
    """
    Complete result of a chaos experiment including all evaluations.
    """
    experiment_id: str
    experiment_name: str
    fault_type: str
    
    # Timing
    started_at: datetime
    completed_at: datetime
    fault_duration_seconds: int
    
    # Ground truth
    ground_truth: FaultGroundTruth
    
    # Detection results
    detection: DetectionResult
    
    # RCA results
    root_cause: RootCauseResult
    
    # Overall metrics
    metrics: EvaluationMetrics
    
    # Raw data for debugging
    incident_data: Optional[Dict[str, Any]] = None
    agent_outputs: Optional[Dict[str, Any]] = None
    
    @property
    def success(self) -> bool:
        """Was the experiment successful (detection + correct RCA)?"""
        return (
            self.detection.incident_detected and
            self.detection.within_threshold and
            self.root_cause.matches_ground_truth
        )


class ExperimentSchedule(BaseModel):
    """
    A schedule for running multiple chaos experiments.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    
    # Schedule configuration
    experiments: List[ChaosExperiment] = Field(default_factory=list)
    interval_between_experiments_seconds: int = Field(
        default=120,
        description="Time to wait between experiments"
    )
    
    # Status
    status: ExperimentStatus = Field(default=ExperimentStatus.PENDING)
    current_experiment_index: int = 0
    
    # Results
    results: List[ExperimentResult] = Field(default_factory=list)
    aggregate_metrics: Optional[EvaluationMetrics] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


def get_ground_truth(fault_type: str) -> Optional[FaultGroundTruth]:
    """Get the ground-truth definition for a fault type."""
    return FAULT_GROUND_TRUTHS.get(fault_type)


def list_available_faults() -> List[Dict[str, str]]:
    """List all available fault types with their descriptions."""
    return [
        {
            "fault_type": gt.fault_type,
            "display_name": gt.display_name,
            "description": gt.description,
            "expected_component": gt.expected_component,
            "expected_severity": gt.expected_severity,
        }
        for gt in FAULT_GROUND_TRUTHS.values()
    ]

