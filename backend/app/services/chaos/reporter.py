"""
Experiment Reporter - Generates reports for chaos experiments.

Produces:
- Individual experiment reports
- Schedule summary reports
- Markdown and JSON formats
- Metrics dashboards
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.services.chaos.models import (
    ExperimentResult,
    ExperimentSchedule,
    EvaluationMetrics,
    FaultGroundTruth,
)


class ExperimentReporter:
    """
    Generates comprehensive reports for chaos experiments.
    
    Supports:
    - Individual experiment reports
    - Batch schedule reports
    - Multiple output formats (Markdown, JSON)
    - Metrics summaries
    """
    
    def generate_experiment_report(
        self,
        result: ExperimentResult,
        format: str = "markdown"
    ) -> str:
        """
        Generate a report for a single experiment.
        
        Args:
            result: The experiment result to report on
            format: Output format ("markdown" or "json")
        
        Returns:
            Formatted report string
        """
        if format == "json":
            return self._generate_json_report(result)
        return self._generate_markdown_report(result)
    
    def generate_schedule_report(
        self,
        schedule: ExperimentSchedule,
        format: str = "markdown"
    ) -> str:
        """
        Generate a comprehensive report for a schedule of experiments.
        
        Args:
            schedule: The experiment schedule with results
            format: Output format ("markdown" or "json")
        
        Returns:
            Formatted report string
        """
        if format == "json":
            return self._generate_schedule_json(schedule)
        return self._generate_schedule_markdown(schedule)
    
    def _generate_markdown_report(self, result: ExperimentResult) -> str:
        """Generate a Markdown report for a single experiment."""
        detection = result.detection
        root_cause = result.root_cause
        metrics = result.metrics
        
        # Status indicators
        detection_icon = "✅" if detection.incident_detected else "❌"
        ttd_icon = "✅" if detection.within_threshold else "⚠️"
        rca_icon = "✅" if root_cause.matches_ground_truth else "❌"
        component_icon = "✅" if root_cause.component_match else "❌"
        
        report = f"""# Chaos Experiment Report

## Experiment Details

| Field | Value |
|-------|-------|
| **ID** | `{result.experiment_id}` |
| **Name** | {result.experiment_name} |
| **Fault Type** | `{result.fault_type}` |
| **Duration** | {result.fault_duration_seconds}s |
| **Started** | {result.started_at.isoformat()} |
| **Completed** | {result.completed_at.isoformat()} |

---

## Ground Truth

| Field | Expected |
|-------|----------|
| **Root Cause Keywords** | {', '.join(result.ground_truth.expected_root_cause_keywords[:5])} |
| **Component** | {result.ground_truth.expected_component} |
| **Component Type** | {result.ground_truth.expected_component_type} |
| **Max Detection Time** | {result.ground_truth.max_detection_time_seconds}s |
| **Expected Severity** | {result.ground_truth.expected_severity} |

---

## Detection Results

| Metric | Result | Status |
|--------|--------|--------|
| **Incident Detected** | {detection.incident_detected} | {detection_icon} |
| **Detection Time** | {f"{detection.detection_time_seconds:.1f}s" if detection.detection_time_seconds else "N/A"} | {ttd_icon} |
| **Within Threshold** | {detection.within_threshold} | {ttd_icon} |
| **Incident ID** | `{detection.incident_id or "N/A"}` | |
| **Incident Title** | {detection.incident_title or "N/A"} | |
| **Severity** | {detection.incident_severity or "N/A"} | |

---

## Root Cause Analysis Results

| Metric | Result | Status |
|--------|--------|--------|
| **RCA Identified** | {root_cause.root_cause_identified} | |
| **Identified Cause** | {root_cause.identified_cause or "N/A"} | |
| **Matches Ground Truth** | {root_cause.matches_ground_truth} | {rca_icon} |
| **Identified Component** | {root_cause.identified_component or "N/A"} | |
| **Component Match** | {root_cause.component_match} | {component_icon} |
| **Confidence** | {f"{root_cause.confidence:.1%}" if root_cause.confidence else "N/A"} | |

---

## Metrics Summary

| Metric | Value |
|--------|-------|
| **Precision** | {metrics.precision:.1%} |
| **Recall** | {metrics.recall:.1%} |
| **F1 Score** | {metrics.f1_score:.1%} |
| **Root Cause Accuracy** | {metrics.root_cause_accuracy:.1%} |
| **Component Accuracy** | {metrics.component_accuracy:.1%} |
| **Mean TTD** | {f"{metrics.mean_time_to_detection:.1f}s" if metrics.mean_time_to_detection else "N/A"} |

---

## Overall Result

**{"✅ SUCCESS" if result.success else "❌ FAILURE"}**

"""
        
        if not result.success:
            report += "\n### Failure Reasons\n\n"
            if not detection.incident_detected:
                report += "- ❌ No incident was detected\n"
            elif not detection.within_threshold:
                report += f"- ⚠️ Detection time ({detection.detection_time_seconds:.1f}s) exceeded threshold ({result.ground_truth.max_detection_time_seconds}s)\n"
            if not root_cause.matches_ground_truth:
                report += f"- ❌ Root cause mismatch: got '{root_cause.identified_cause}', expected keywords: {result.ground_truth.expected_root_cause_keywords}\n"
            if not root_cause.component_match:
                report += f"- ❌ Component mismatch: got '{root_cause.identified_component}', expected '{result.ground_truth.expected_component}'\n"
        
        return report
    
    def _generate_json_report(self, result: ExperimentResult) -> str:
        """Generate a JSON report for a single experiment."""
        return json.dumps({
            "experiment_id": result.experiment_id,
            "experiment_name": result.experiment_name,
            "fault_type": result.fault_type,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat(),
            "fault_duration_seconds": result.fault_duration_seconds,
            "success": result.success,
            "ground_truth": {
                "expected_root_cause_keywords": result.ground_truth.expected_root_cause_keywords,
                "expected_component": result.ground_truth.expected_component,
                "expected_component_type": result.ground_truth.expected_component_type,
                "max_detection_time_seconds": result.ground_truth.max_detection_time_seconds,
            },
            "detection": {
                "incident_detected": result.detection.incident_detected,
                "detection_time_seconds": result.detection.detection_time_seconds,
                "within_threshold": result.detection.within_threshold,
                "incident_id": result.detection.incident_id,
                "incident_title": result.detection.incident_title,
                "incident_severity": result.detection.incident_severity,
            },
            "root_cause": {
                "root_cause_identified": result.root_cause.root_cause_identified,
                "identified_cause": result.root_cause.identified_cause,
                "identified_component": result.root_cause.identified_component,
                "matches_ground_truth": result.root_cause.matches_ground_truth,
                "component_match": result.root_cause.component_match,
                "confidence": result.root_cause.confidence,
            },
            "metrics": {
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1_score": result.metrics.f1_score,
                "root_cause_accuracy": result.metrics.root_cause_accuracy,
                "component_accuracy": result.metrics.component_accuracy,
                "mean_time_to_detection": result.metrics.mean_time_to_detection,
            }
        }, indent=2)
    
    def _generate_schedule_markdown(self, schedule: ExperimentSchedule) -> str:
        """Generate a Markdown report for a schedule of experiments."""
        metrics = schedule.aggregate_metrics or EvaluationMetrics()
        
        # Count successes and failures
        successes = sum(1 for r in schedule.results if r.success)
        failures = len(schedule.results) - successes
        
        report = f"""# Chaos Experiment Schedule Report

## Schedule Overview

| Field | Value |
|-------|-------|
| **ID** | `{schedule.id}` |
| **Name** | {schedule.name} |
| **Description** | {schedule.description or "N/A"} |
| **Total Experiments** | {len(schedule.experiments)} |
| **Completed** | {len(schedule.results)} |
| **Status** | {schedule.status.value} |
| **Started** | {schedule.started_at.isoformat() if schedule.started_at else "N/A"} |
| **Completed** | {schedule.completed_at.isoformat() if schedule.completed_at else "N/A"} |

---

## Aggregate Metrics

### Detection Performance

| Metric | Value |
|--------|-------|
| **Precision** | {metrics.precision:.1%} |
| **Recall** | {metrics.recall:.1%} |
| **F1 Score** | {metrics.f1_score:.1%} |
| **True Positives** | {metrics.true_positives} |
| **False Positives** | {metrics.false_positives} |
| **False Negatives** | {metrics.false_negatives} |

### Time-to-Detection

| Metric | Value |
|--------|-------|
| **Mean TTD** | {f"{metrics.mean_time_to_detection:.1f}s" if metrics.mean_time_to_detection else "N/A"} |
| **Min TTD** | {f"{metrics.min_time_to_detection:.1f}s" if metrics.min_time_to_detection else "N/A"} |
| **Max TTD** | {f"{metrics.max_time_to_detection:.1f}s" if metrics.max_time_to_detection else "N/A"} |

### RCA Accuracy

| Metric | Value |
|--------|-------|
| **Root Cause Accuracy** | {metrics.root_cause_accuracy:.1%} |
| **Component Accuracy** | {metrics.component_accuracy:.1%} |
| **Root Cause Matches** | {metrics.root_cause_matches}/{metrics.total_evaluated} |
| **Component Matches** | {metrics.component_matches}/{metrics.total_evaluated} |

---

## Results Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ **Success** | {successes} | {successes/len(schedule.results)*100:.1f}% |
| ❌ **Failure** | {failures} | {failures/len(schedule.results)*100:.1f}% |

---

## Individual Experiment Results

| # | Fault Type | Detected | TTD | RCA Match | Component Match | Success |
|---|------------|----------|-----|-----------|-----------------|---------|
"""
        
        for i, result in enumerate(schedule.results, 1):
            detected = "✅" if result.detection.incident_detected else "❌"
            ttd = f"{result.detection.detection_time_seconds:.1f}s" if result.detection.detection_time_seconds else "N/A"
            rca = "✅" if result.root_cause.matches_ground_truth else "❌"
            component = "✅" if result.root_cause.component_match else "❌"
            success = "✅" if result.success else "❌"
            
            report += f"| {i} | `{result.fault_type}` | {detected} | {ttd} | {rca} | {component} | {success} |\n"
        
        report += "\n---\n\n"
        
        # Add failure analysis if there are failures
        if failures > 0:
            report += "## Failure Analysis\n\n"
            
            for result in schedule.results:
                if not result.success:
                    report += f"### {result.fault_type}\n\n"
                    
                    if not result.detection.incident_detected:
                        report += "- ❌ **No incident detected**\n"
                    elif not result.detection.within_threshold:
                        report += f"- ⚠️ **Slow detection**: {result.detection.detection_time_seconds:.1f}s (threshold: {result.ground_truth.max_detection_time_seconds}s)\n"
                    
                    if not result.root_cause.matches_ground_truth:
                        report += f"- ❌ **RCA mismatch**:\n"
                        report += f"  - Got: `{result.root_cause.identified_cause}`\n"
                        report += f"  - Expected keywords: {result.ground_truth.expected_root_cause_keywords}\n"
                    
                    if not result.root_cause.component_match:
                        report += f"- ❌ **Component mismatch**:\n"
                        report += f"  - Got: `{result.root_cause.identified_component}`\n"
                        report += f"  - Expected: `{result.ground_truth.expected_component}`\n"
                    
                    report += "\n"
        
        report += """---

*Report generated by Multi-Agent Incident Response Chaos Engineering System*
"""
        
        return report
    
    def _generate_schedule_json(self, schedule: ExperimentSchedule) -> str:
        """Generate a JSON report for a schedule of experiments."""
        metrics = schedule.aggregate_metrics or EvaluationMetrics()
        
        return json.dumps({
            "schedule_id": schedule.id,
            "name": schedule.name,
            "description": schedule.description,
            "status": schedule.status.value,
            "started_at": schedule.started_at.isoformat() if schedule.started_at else None,
            "completed_at": schedule.completed_at.isoformat() if schedule.completed_at else None,
            "total_experiments": len(schedule.experiments),
            "completed_experiments": len(schedule.results),
            "aggregate_metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "mean_time_to_detection": metrics.mean_time_to_detection,
                "min_time_to_detection": metrics.min_time_to_detection,
                "max_time_to_detection": metrics.max_time_to_detection,
                "root_cause_accuracy": metrics.root_cause_accuracy,
                "component_accuracy": metrics.component_accuracy,
                "root_cause_matches": metrics.root_cause_matches,
                "component_matches": metrics.component_matches,
                "total_evaluated": metrics.total_evaluated,
            },
            "results": [
                {
                    "experiment_id": r.experiment_id,
                    "fault_type": r.fault_type,
                    "success": r.success,
                    "detection": {
                        "incident_detected": r.detection.incident_detected,
                        "detection_time_seconds": r.detection.detection_time_seconds,
                        "within_threshold": r.detection.within_threshold,
                    },
                    "root_cause": {
                        "matches_ground_truth": r.root_cause.matches_ground_truth,
                        "component_match": r.root_cause.component_match,
                        "confidence": r.root_cause.confidence,
                    }
                }
                for r in schedule.results
            ]
        }, indent=2)
    
    def generate_metrics_summary(
        self,
        metrics: EvaluationMetrics
    ) -> Dict[str, Any]:
        """
        Generate a summary dictionary of metrics.
        
        Useful for dashboard displays or API responses.
        """
        return {
            "detection": {
                "precision": round(metrics.precision, 3),
                "recall": round(metrics.recall, 3),
                "f1_score": round(metrics.f1_score, 3),
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
            },
            "time_to_detection": {
                "mean_seconds": round(metrics.mean_time_to_detection, 1) if metrics.mean_time_to_detection else None,
                "min_seconds": round(metrics.min_time_to_detection, 1) if metrics.min_time_to_detection else None,
                "max_seconds": round(metrics.max_time_to_detection, 1) if metrics.max_time_to_detection else None,
            },
            "rca_accuracy": {
                "root_cause_accuracy": round(metrics.root_cause_accuracy, 3),
                "component_accuracy": round(metrics.component_accuracy, 3),
                "root_cause_matches": metrics.root_cause_matches,
                "component_matches": metrics.component_matches,
                "total_evaluated": metrics.total_evaluated,
            }
        }

