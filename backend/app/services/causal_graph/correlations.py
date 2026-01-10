"""
Correlation calculations for causal edge weighting.

Implements:
- Pearson correlation for linear metric relationships
- Spearman correlation for monotonic relationships
- Log co-occurrence scoring for error pattern matching
- Temporal alignment scoring for causality direction

All functions are deterministic given the same inputs.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from app.services.causal_graph.models import (
    CausalNode,
    MetricTimeSeries,
    EdgeWeight,
)


def pearson_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """
    Compute Pearson correlation coefficient between two series.
    
    Pearson measures linear correlation: how well the relationship
    between two variables can be described by a linear equation.
    
    Args:
        x: First time series values
        y: Second time series values (must be same length as x)
        
    Returns:
        Correlation coefficient between -1 and 1, or None if insufficient data
        
    Edge Cases:
        - Returns None if len(x) != len(y) (mismatched series)
        - Returns None if n < 3 (insufficient data for meaningful correlation)
        - Returns None if either series is constant (std = 0)
        
    Example:
        >>> pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        1.0  # Perfect positive correlation
    """
    n = len(x)
    if n != len(y) or n < 3:
        return None
    
    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate standard deviations and covariance
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    sum_coproduct = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    # Avoid division by zero (constant series have no correlation)
    std_x = math.sqrt(sum_sq_x / n)
    std_y = math.sqrt(sum_sq_y / n)
    
    if std_x == 0 or std_y == 0:
        return None
    
    correlation = sum_coproduct / (n * std_x * std_y)
    
    # Clamp to [-1, 1] to handle floating point rounding errors
    return max(-1.0, min(1.0, correlation))


def spearman_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """
    Compute Spearman rank correlation coefficient.
    
    Spearman measures monotonic correlation: whether the relationship
    between variables is consistently increasing or decreasing,
    even if not linear. More robust to outliers than Pearson.
    
    Args:
        x: First time series values
        y: Second time series values (must be same length as x)
        
    Returns:
        Correlation coefficient between -1 and 1, or None if insufficient data
        
    Example:
        >>> spearman_correlation([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
        1.0  # Perfect monotonic correlation (quadratic relationship)
    """
    n = len(x)
    if n != len(y) or n < 3:
        return None
    
    # Convert to ranks
    rank_x = _compute_ranks(x)
    rank_y = _compute_ranks(y)
    
    # Compute Pearson on ranks
    return pearson_correlation(rank_x, rank_y)


def _compute_ranks(values: List[float]) -> List[float]:
    """
    Compute ranks for a list of values, handling ties with average rank.
    
    Args:
        values: List of numeric values
        
    Returns:
        List of ranks (1-based, with ties averaged)
    """
    n = len(values)
    # Create list of (value, original_index)
    indexed = [(v, i) for i, v in enumerate(values)]
    # Sort by value
    sorted_indexed = sorted(indexed, key=lambda x: x[0])
    
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find all ties
        while j < n and sorted_indexed[j][0] == sorted_indexed[i][0]:
            j += 1
        # Average rank for ties
        avg_rank = (i + j + 1) / 2  # +1 for 1-based ranking
        for k in range(i, j):
            ranks[sorted_indexed[k][1]] = avg_rank
        i = j
    
    return ranks


def log_cooccurrence_score(
    logs_a: List[Dict[str, Any]],
    logs_b: List[Dict[str, Any]],
    time_window_seconds: float = 60.0
) -> float:
    """
    Compute log co-occurrence score between two nodes.
    
    Measures how often error logs from two components appear within
    the same time window. High co-occurrence suggests correlation.
    
    Args:
        logs_a: Log entries from first node (must have 'timestamp' and 'level')
        logs_b: Log entries from second node
        time_window_seconds: Time window to consider as "co-occurring"
        
    Returns:
        Score between 0 and 1 indicating co-occurrence frequency
    """
    if not logs_a or not logs_b:
        return 0.0
    
    # Filter to error logs only
    errors_a = _filter_error_logs(logs_a)
    errors_b = _filter_error_logs(logs_b)
    
    if not errors_a or not errors_b:
        return 0.0
    
    # Parse timestamps
    times_a = _parse_timestamps(errors_a)
    times_b = _parse_timestamps(errors_b)
    
    if not times_a or not times_b:
        return 0.0
    
    # Count co-occurrences
    cooccurrences = 0
    window = timedelta(seconds=time_window_seconds)
    
    for ta in times_a:
        for tb in times_b:
            if abs((ta - tb).total_seconds()) <= time_window_seconds:
                cooccurrences += 1
                break  # Count each error from A only once
    
    # Normalize by the smaller set size
    min_size = min(len(times_a), len(times_b))
    score = cooccurrences / min_size if min_size > 0 else 0.0
    
    return min(1.0, score)


def _filter_error_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter log entries to only errors and warnings."""
    return [
        log for log in logs
        if log.get("level", "").lower() in ("error", "warning", "critical", "fatal")
    ]


def _parse_timestamps(logs: List[Dict[str, Any]]) -> List[datetime]:
    """Parse timestamps from log entries."""
    timestamps = []
    for log in logs:
        ts = log.get("timestamp")
        if ts:
            if isinstance(ts, datetime):
                timestamps.append(ts)
            elif isinstance(ts, str):
                try:
                    # Try ISO format
                    timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
                except ValueError:
                    pass
    return timestamps


def temporal_alignment_score(
    first_anomaly_a: Optional[datetime],
    first_anomaly_b: Optional[datetime],
    max_lag_seconds: float = 300.0
) -> Tuple[float, bool]:
    """
    Compute temporal alignment score and causality direction.
    
    Earlier anomalies are weighted higher as potential causes.
    Returns both a score and the direction of causality.
    
    Args:
        first_anomaly_a: When component A first showed anomaly
        first_anomaly_b: When component B first showed anomaly
        max_lag_seconds: Maximum time lag to consider as related
        
    Returns:
        Tuple of (score, a_precedes_b):
        - score: 0 to 1, higher if times are close
        - a_precedes_b: True if A's anomaly came first
    """
    if first_anomaly_a is None or first_anomaly_b is None:
        return 0.0, True  # Unknown, assume no temporal signal
    
    lag_seconds = (first_anomaly_b - first_anomaly_a).total_seconds()
    a_precedes_b = lag_seconds >= 0
    
    abs_lag = abs(lag_seconds)
    
    # If lag exceeds maximum, no temporal relationship
    if abs_lag > max_lag_seconds:
        return 0.0, a_precedes_b
    
    # Score decreases linearly with lag
    # Closest in time = highest score
    score = 1.0 - (abs_lag / max_lag_seconds)
    
    # Boost score if there's a clear temporal ordering (not simultaneous)
    if 5.0 < abs_lag < max_lag_seconds * 0.8:
        score *= 1.2  # Small boost for clear causality
        score = min(1.0, score)
    
    return score, a_precedes_b


def keyword_overlap_score(
    keywords_a: List[str],
    keywords_b: List[str]
) -> float:
    """
    Compute keyword overlap between two sets of log keywords.
    
    Args:
        keywords_a: Keywords extracted from node A's logs
        keywords_b: Keywords extracted from node B's logs
        
    Returns:
        Jaccard similarity between 0 and 1
    """
    if not keywords_a or not keywords_b:
        return 0.0
    
    set_a = set(k.lower() for k in keywords_a)
    set_b = set(k.lower() for k in keywords_b)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


class CorrelationCalculator:
    """
    Calculates correlations between causal graph nodes.
    
    Combines multiple correlation methods to produce edge weights
    that capture different aspects of the relationship:
    - Metric correlation (Pearson/Spearman)
    - Log co-occurrence
    - Temporal alignment
    """
    
    def __init__(
        self,
        metric_weight: float = 0.4,
        log_weight: float = 0.3,
        temporal_weight: float = 0.3,
        time_window_seconds: float = 60.0,
        max_lag_seconds: float = 300.0
    ):
        """
        Initialize the correlation calculator.
        
        Args:
            metric_weight: Weight for metric correlation in final score
            log_weight: Weight for log co-occurrence in final score
            temporal_weight: Weight for temporal alignment in final score
            time_window_seconds: Window for log co-occurrence
            max_lag_seconds: Maximum lag for temporal correlation
        """
        self.metric_weight = metric_weight
        self.log_weight = log_weight
        self.temporal_weight = temporal_weight
        self.time_window_seconds = time_window_seconds
        self.max_lag_seconds = max_lag_seconds
    
    def compute_edge_weight(
        self,
        node_a: CausalNode,
        node_b: CausalNode,
        logs_a: Optional[List[Dict[str, Any]]] = None,
        logs_b: Optional[List[Dict[str, Any]]] = None
    ) -> EdgeWeight:
        """
        Compute the edge weight between two nodes.
        
        Args:
            node_a: Source node
            node_b: Target node
            logs_a: Optional log entries for node A
            logs_b: Optional log entries for node B
            
        Returns:
            EdgeWeight with all component scores
        """
        weight = EdgeWeight()
        
        # 1. Metric correlation
        metric_corr = self._compute_metric_correlation(node_a, node_b)
        if metric_corr is not None:
            weight.metric_correlation = metric_corr["combined"]
            weight.pearson = metric_corr.get("pearson")
            weight.spearman = metric_corr.get("spearman")
        
        # 2. Log co-occurrence
        if logs_a and logs_b:
            weight.log_cooccurrence = log_cooccurrence_score(
                logs_a, logs_b, self.time_window_seconds
            )
        elif node_a.logs.error_messages and node_b.logs.error_messages:
            # Fall back to stored error messages
            weight.log_cooccurrence = keyword_overlap_score(
                node_a.logs.keywords, node_b.logs.keywords
            )
        
        # 3. Temporal alignment
        temporal_score, _ = temporal_alignment_score(
            node_a.first_anomaly_time,
            node_b.first_anomaly_time,
            self.max_lag_seconds
        )
        weight.temporal_weight = temporal_score
        
        return weight
    
    def _compute_metric_correlation(
        self,
        node_a: CausalNode,
        node_b: CausalNode
    ) -> Optional[Dict[str, float]]:
        """
        Compute metric correlation between two nodes.
        
        Uses error rate and latency time series if available.
        
        Returns:
            Dict with pearson, spearman, and combined scores
        """
        correlations = []
        pearson_scores = []
        spearman_scores = []
        
        # Try error rate series
        if (node_a.metrics.error_rate_series and 
            node_b.metrics.error_rate_series):
            series_a = node_a.metrics.error_rate_series.to_list()
            series_b = node_b.metrics.error_rate_series.to_list()
            
            if len(series_a) == len(series_b) and len(series_a) >= 3:
                p = pearson_correlation(series_a, series_b)
                s = spearman_correlation(series_a, series_b)
                if p is not None:
                    pearson_scores.append(p)
                    correlations.append(abs(p))
                if s is not None:
                    spearman_scores.append(s)
                    correlations.append(abs(s))
        
        # Try latency series
        if (node_a.metrics.latency_series and 
            node_b.metrics.latency_series):
            series_a = node_a.metrics.latency_series.to_list()
            series_b = node_b.metrics.latency_series.to_list()
            
            if len(series_a) == len(series_b) and len(series_a) >= 3:
                p = pearson_correlation(series_a, series_b)
                s = spearman_correlation(series_a, series_b)
                if p is not None:
                    pearson_scores.append(p)
                    correlations.append(abs(p))
                if s is not None:
                    spearman_scores.append(s)
                    correlations.append(abs(s))
        
        if not correlations:
            return None
        
        return {
            "combined": sum(correlations) / len(correlations),
            "pearson": sum(pearson_scores) / len(pearson_scores) if pearson_scores else None,
            "spearman": sum(spearman_scores) / len(spearman_scores) if spearman_scores else None,
        }
    
    def compute_causality_direction(
        self,
        node_a: CausalNode,
        node_b: CausalNode
    ) -> Tuple[bool, float]:
        """
        Determine the direction of causality between two nodes.
        
        Args:
            node_a: First node
            node_b: Second node
            
        Returns:
            Tuple of (a_causes_b, confidence):
            - a_causes_b: True if A likely caused B's issues
            - confidence: 0 to 1, how confident we are in direction
        """
        _, a_precedes_b = temporal_alignment_score(
            node_a.first_anomaly_time,
            node_b.first_anomaly_time,
            self.max_lag_seconds
        )
        
        # Base confidence on time difference
        if node_a.first_anomaly_time and node_b.first_anomaly_time:
            lag = abs((node_b.first_anomaly_time - node_a.first_anomaly_time).total_seconds())
            if lag > 10:  # Clear temporal ordering
                confidence = min(1.0, 0.5 + lag / self.max_lag_seconds)
            else:  # Nearly simultaneous, low confidence
                confidence = 0.3
        else:
            confidence = 0.5  # Unknown
        
        return a_precedes_b, confidence

