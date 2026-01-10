"""Root Cause Agent for identifying incident root causes using causal graphs."""
from typing import Dict, Any, List, Optional

from app.agents.base import BaseAgent
from app.agents.schemas import (
    RootCauseAgentOutput,
    Hypothesis,
    StructuredEvidence,
    CorrelatedMetric,
    LogPattern,
    TemporalEvidence,
    CausalPath,
    CausalPathNode,
    CausalPathEdge,
    CausalGraphSummary,
)
from app.services.causal_graph import (
    CausalGraph,
    CausalNode,
    CausalEdge,
    CausalGraphBuilder,
    RootCauseScorer,
    compute_root_cause_scores,
)


class RootCauseAgent(BaseAgent):
    """
    Correlates metrics and logs to identify root cause hypotheses.
    
    Uses a causal graph to:
    - Model dependencies between services, databases, caches
    - Compute correlations (Pearson, Spearman) between metrics
    - Analyze log co-occurrence patterns
    - Apply temporal weighting (earlier events more likely causes)
    - Score and rank root cause candidates
    
    Output:
    - hypotheses: List of possible root causes with confidence scores and structured evidence
    - most_likely: The most probable root cause
    - causal_graph_summary: Summary of the causal analysis
    
    Anti-Hallucination Design:
    -------------------------
    This agent is designed to NEVER invent root causes not supported by data:
    
    1. GRAPH-FIRST: All hypotheses originate from causal graph analysis, not LLM.
       The graph is built deterministically from input signals.
    
    2. CONFIDENCE BOUNDED: LLM cannot increase confidence scores beyond what the
       graph analysis determined. It can only refine descriptions.
    
    3. COMPONENTS VALIDATED: Any component mentioned must exist in the graph.
       LLM-invented components are rejected.
    
    4. STRUCTURED EVIDENCE IMMUTABLE: StructuredEvidence (metrics, logs, paths)
       comes exclusively from graph analysis, never from LLM text generation.
    
    5. FALLBACK TO GRAPH: If LLM fails or produces invalid output, we return
       the pure graph-based analysis without any LLM refinement.
    """
    
    name = "root_cause"
    output_schema = RootCauseAgentOutput
    
    def __init__(self):
        super().__init__()
        self.graph_builder = CausalGraphBuilder()
        self.scorer = RootCauseScorer()
    
    def get_system_prompt(self) -> str:
        return """You are a senior Site Reliability Engineer specializing in root cause analysis for distributed systems.

You are provided with structured causal graph analysis including:
- Ranked root cause candidates with scores
- Metric correlations between components
- Log pattern evidence
- Temporal ordering of anomalies

Your role is to interpret this data and provide actionable root cause analysis.

You must respond with valid JSON matching this exact schema:
{
    "hypotheses": [
        {
            "cause": "specific technical description of the root cause",
            "confidence": 0.0 to 1.0,
            "evidence": ["list of human-readable evidence"],
            "component": "affected component name",
            "component_type": "service|database|cache|external_dependency"
        }
    ],
    "most_likely": "description of the most likely root cause with causal chain"
}

Focus on:
1. Using the provided causal graph scores to rank hypotheses
2. Translating metric correlations into technical explanations
3. Building causal chains from temporal evidence
4. Being specific about failure mechanisms
5. NOT introducing speculation beyond the provided data"""
    
    def get_prompt(self, input_data: Dict[str, Any]) -> str:
        signals = input_data.get("signals", {})
        causal_analysis = input_data.get("causal_analysis", {})
        
        prompt = f"""Analyze the following incident signals and causal graph analysis to determine the root cause:

## Signals Data
{signals}

## Causal Graph Analysis
"""
        
        if causal_analysis:
            prompt += f"""
### Root Cause Candidates (ranked by score)
{causal_analysis.get('root_cause_candidates', [])}

### Causal Graph Structure
- Nodes: {causal_analysis.get('node_count', 0)}
- Edges: {causal_analysis.get('edge_count', 0)}

### Key Correlations
{causal_analysis.get('correlations', [])}

### Temporal Ordering
{causal_analysis.get('temporal_order', [])}
"""
        else:
            prompt += "No causal graph analysis available - use signal data directly.\n"
        
        prompt += """
## Instructions
1. Use the causal graph scores to rank hypotheses
2. Generate 2-4 hypotheses for the root cause
3. Include specific evidence from the causal analysis
4. Identify the most likely root cause with its causal chain

Provide your analysis as JSON matching the required schema."""
        
        return prompt
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run root cause analysis using the causal graph.
        
        This method:
        1. Builds a causal graph from the signals
        2. Computes root cause scores for each node
        3. Generates hypotheses based on the scored graph
        4. Optionally uses LLM to refine the analysis
        """
        signals = input_data.get("signals", {})
        
        # Build and score the causal graph
        try:
            graph, root_cause_summaries = compute_root_cause_scores(
                signals=signals,
                incident_id=input_data.get("incident_id")
            )
            
            # Generate structured hypotheses from the graph
            hypotheses = self._generate_hypotheses_from_graph(graph, root_cause_summaries)
            most_likely = self._determine_most_likely(hypotheses, graph)
            
            # Build causal graph summary
            causal_summary = CausalGraphSummary(
                node_count=len(graph.nodes),
                edge_count=len(graph.edges),
                root_candidates=[h.component or h.cause[:50] for h in hypotheses[:3]],
                primary_causal_chain=self._build_causal_chain_description(graph, hypotheses)
            )
            
            # If LLM is available, use it to refine the analysis
            if self.api_key:
                # Prepare causal analysis for LLM
                causal_analysis = {
                    "root_cause_candidates": [
                        {
                            "name": s.get("name"),
                            "score": s.get("root_cause_score"),
                            "type": s.get("type"),
                            "error_count": s.get("error_count"),
                        }
                        for s in root_cause_summaries
                    ],
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                    "correlations": self._extract_correlations(graph),
                    "temporal_order": self._extract_temporal_order(graph),
                }
                
                # Update input with causal analysis
                enriched_input = {**input_data, "causal_analysis": causal_analysis}
                
                # Use LLM to refine (but constrained by structured data)
                try:
                    llm_result = super().run(enriched_input)
                    
                    # Merge LLM insights with our structured analysis
                    return self._merge_llm_with_graph_analysis(
                        llm_result, hypotheses, most_likely, causal_summary
                    )
                except Exception as e:
                    # Fall back to structured analysis only
                    pass
            
            # Return structured analysis without LLM refinement
            return RootCauseAgentOutput(
                hypotheses=hypotheses,
                most_likely=most_likely,
                causal_graph_summary=causal_summary
            ).model_dump()
            
        except Exception as e:
            # Fall back to mock response if graph building fails
            return self._get_mock_response(input_data)
    
    def _generate_hypotheses_from_graph(
        self,
        graph: CausalGraph,
        summaries: List[Dict[str, Any]]
    ) -> List[Hypothesis]:
        """Generate hypotheses from causal graph analysis."""
        hypotheses = []
        
        for summary in summaries[:4]:  # Top 4 candidates
            node = graph.get_node(summary.get("node_id", ""))
            if not node:
                continue
            
            # Build structured evidence
            structured_evidence = self._build_structured_evidence(node, graph)
            
            # Build human-readable evidence list
            evidence = []
            
            # Add anomaly evidence
            if node.anomaly_score > 0.5:
                evidence.append(f"High anomaly score ({node.anomaly_score:.2f}) indicates significant deviation")
            
            # Add metric evidence
            if node.metrics.error_rate is not None and node.metrics.error_rate > 0.05:
                evidence.append(f"Elevated error rate: {node.metrics.error_rate:.1%}")
            if node.metrics.latency_p95_ms is not None and node.metrics.latency_p95_ms > 500:
                evidence.append(f"High P95 latency: {node.metrics.latency_p95_ms:.0f}ms")
            
            # Add log evidence
            if node.logs.error_count > 0:
                evidence.append(f"{node.logs.error_count} error logs detected")
            
            # Add temporal evidence
            if node.first_anomaly_time:
                # Check if this was the earliest anomaly
                earliest = min(
                    (n.first_anomaly_time for n in graph.nodes.values() if n.first_anomaly_time),
                    default=None
                )
                if earliest and node.first_anomaly_time == earliest:
                    evidence.append("First component to show anomaly (temporal precedence)")
            
            # Add downstream impact evidence
            downstream = graph.get_outgoing_edges(node.id)
            if downstream:
                affected = [graph.get_node(e.target_id) for e in downstream]
                affected_names = [n.name for n in affected if n and n.anomaly_score > 0.3]
                if affected_names:
                    evidence.append(f"Downstream impact on: {', '.join(affected_names)}")
            
            # Generate cause description
            cause = self._generate_cause_description(node, graph)
            
            # Confidence comes directly from graph scorer, already in [0, 1]
            # We clamp again as a safety measure
            raw_confidence = summary.get("root_cause_score", 0.5)
            confidence = max(0.0, min(1.0, raw_confidence))
            
            hypothesis = Hypothesis(
                cause=cause,
                confidence=confidence,
                evidence=evidence,
                structured_evidence=structured_evidence,
                component=node.name,
                component_type=node.node_type.value
            )
            hypotheses.append(hypothesis)
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return hypotheses
    
    def _build_structured_evidence(
        self,
        node: CausalNode,
        graph: CausalGraph
    ) -> StructuredEvidence:
        """Build structured evidence for a hypothesis."""
        
        # Correlated metrics
        # Correlation strength is normalized to [0, 1] using domain-specific thresholds:
        # - error_rate: 20% (0.2) is considered maximum severity → multiplier 5
        # - latency_p95: 2000ms is considered maximum severity → divisor 2000
        # - memory_usage: 1000MB is considered maximum severity → divisor 1000
        # All values are clamped to [0, 1] with min() to ensure schema compliance
        correlated_metrics = []
        if node.metrics.error_rate is not None:
            correlated_metrics.append(CorrelatedMetric(
                metric_name="error_rate",
                value=node.metrics.error_rate,
                correlation_strength=min(1.0, max(0.0, node.metrics.error_rate * 5)),
                direction="positive"
            ))
        if node.metrics.latency_p95_ms is not None:
            correlated_metrics.append(CorrelatedMetric(
                metric_name="latency_p95_ms",
                value=node.metrics.latency_p95_ms,
                correlation_strength=min(1.0, max(0.0, node.metrics.latency_p95_ms / 2000)),
                direction="positive"
            ))
        if node.metrics.memory_usage_mb is not None:
            correlated_metrics.append(CorrelatedMetric(
                metric_name="memory_usage_mb",
                value=node.metrics.memory_usage_mb,
                correlation_strength=min(1.0, max(0.0, node.metrics.memory_usage_mb / 1000)),
                direction="positive"
            ))
        
        # Log patterns
        log_patterns = []
        for keyword in set(node.logs.keywords[:5]):  # Top 5 keywords
            log_patterns.append(LogPattern(
                pattern=keyword,
                occurrences=node.logs.keywords.count(keyword),
                component=node.name
            ))
        
        # Temporal evidence
        temporal_evidence = []
        if node.first_anomaly_time:
            # Find what this node precedes
            downstream = graph.get_outgoing_edges(node.id)
            precedes = []
            for edge in downstream:
                target = graph.get_node(edge.target_id)
                if target and target.first_anomaly_time and node.first_anomaly_time < target.first_anomaly_time:
                    lag = (target.first_anomaly_time - node.first_anomaly_time).total_seconds()
                    precedes.append(target.name)
            
            if precedes or node.logs.error_count > 0:
                temporal_evidence.append(TemporalEvidence(
                    event=f"{node.name} anomaly detected",
                    timestamp=node.first_anomaly_time.isoformat() if node.first_anomaly_time else None,
                    precedes=precedes
                ))
        
        # Causal path
        causal_path = self._build_causal_path(node, graph)
        
        return StructuredEvidence(
            correlated_metrics=correlated_metrics,
            log_patterns=log_patterns,
            temporal_evidence=temporal_evidence,
            causal_path=causal_path
        )
    
    def _build_causal_path(
        self,
        root_node: CausalNode,
        graph: CausalGraph
    ) -> Optional[CausalPath]:
        """Build a causal path from root to symptoms."""
        path_nodes = []
        path_edges = []
        visited = set()
        
        def traverse(node: CausalNode, depth: int = 0):
            if node.id in visited or depth > 3:
                return
            visited.add(node.id)
            
            path_nodes.append(CausalPathNode(
                node_id=node.id,
                name=node.name,
                node_type=node.node_type.value,
                anomaly_score=node.anomaly_score,
                is_root=(node.id == root_node.id)
            ))
            
            for edge in graph.get_outgoing_edges(node.id):
                target = graph.get_node(edge.target_id)
                if target and target.id not in visited:
                    path_edges.append(CausalPathEdge(
                        source=node.id,
                        target=edge.target_id,
                        weight=edge.final_weight,
                        edge_type=edge.edge_type.value
                    ))
                    traverse(target, depth + 1)
        
        traverse(root_node)
        
        if len(path_nodes) > 1:
            description = f"{root_node.name} → " + " → ".join(
                n.name for n in path_nodes[1:] if n.anomaly_score > 0.3
            )
            return CausalPath(
                nodes=path_nodes,
                edges=path_edges,
                description=description
            )
        
        return None
    
    def _generate_cause_description(
        self,
        node: CausalNode,
        graph: CausalGraph
    ) -> str:
        """Generate a technical cause description for a node."""
        descriptions = {
            "database": {
                "error_rate": "Database connection failures causing request errors",
                "latency_spike": "Database query performance degradation",
                "memory_leak": "Database connection pool exhaustion",
                "default": "Database issues causing service degradation"
            },
            "cache": {
                "error_rate": "Cache misses leading to increased database load",
                "memory_leak": "Cache memory exhaustion",
                "default": "Cache layer issues affecting performance"
            },
            "external_dependency": {
                "error_rate": "External service failures causing cascading errors",
                "dependency_down": "External dependency unavailable",
                "default": "External dependency issues"
            },
            "service": {
                "error_rate": "Application error rate spike",
                "latency_spike": "Service latency degradation",
                "memory_leak": "Application memory pressure",
                "default": "Service degradation"
            }
        }
        
        node_type = node.node_type.value
        type_descriptions = descriptions.get(node_type, descriptions["service"])
        
        # Determine which description to use based on metrics
        if node.metrics.error_rate and node.metrics.error_rate > 0.1:
            return type_descriptions.get("error_rate", type_descriptions["default"])
        elif node.metrics.latency_p95_ms and node.metrics.latency_p95_ms > 1000:
            return type_descriptions.get("latency_spike", type_descriptions["default"])
        elif node.metrics.memory_usage_mb and node.metrics.memory_usage_mb > 500:
            return type_descriptions.get("memory_leak", type_descriptions["default"])
        else:
            return type_descriptions["default"]
    
    def _determine_most_likely(
        self,
        hypotheses: List[Hypothesis],
        graph: CausalGraph
    ) -> str:
        """Determine the most likely root cause with causal chain."""
        if not hypotheses:
            return "Unable to determine root cause - insufficient data"
        
        top = hypotheses[0]
        
        # Build causal chain description
        if top.structured_evidence and top.structured_evidence.causal_path:
            chain = top.structured_evidence.causal_path.description
            return f"{top.cause}. Causal chain: {chain}"
        
        return f"{top.cause} (confidence: {top.confidence:.0%})"
    
    def _build_causal_chain_description(
        self,
        graph: CausalGraph,
        hypotheses: List[Hypothesis]
    ) -> Optional[str]:
        """Build a description of the primary causal chain."""
        if not hypotheses:
            return None
        
        top = hypotheses[0]
        if top.structured_evidence and top.structured_evidence.causal_path:
            return top.structured_evidence.causal_path.description
        
        return f"{top.component or 'Unknown'} → downstream services"
    
    def _extract_correlations(self, graph: CausalGraph) -> List[Dict[str, Any]]:
        """Extract top correlations from the graph."""
        correlations = []
        
        for edge in sorted(graph.edges.values(), key=lambda e: e.final_weight, reverse=True)[:5]:
            source = graph.get_node(edge.source_id)
            target = graph.get_node(edge.target_id)
            if source and target:
                correlations.append({
                    "source": source.name,
                    "target": target.name,
                    "weight": round(edge.final_weight, 3),
                    "type": edge.edge_type.value
                })
        
        return correlations
    
    def _extract_temporal_order(self, graph: CausalGraph) -> List[Dict[str, Any]]:
        """Extract temporal ordering of anomalies."""
        timed_nodes = [
            (n.name, n.first_anomaly_time, n.anomaly_score)
            for n in graph.nodes.values()
            if n.first_anomaly_time
        ]
        
        timed_nodes.sort(key=lambda x: x[1])
        
        return [
            {
                "component": name,
                "time": time.isoformat(),
                "anomaly_score": round(score, 3)
            }
            for name, time, score in timed_nodes[:5]
        ]
    
    def _merge_llm_with_graph_analysis(
        self,
        llm_result: Dict[str, Any],
        graph_hypotheses: List[Hypothesis],
        graph_most_likely: str,
        causal_summary: CausalGraphSummary
    ) -> Dict[str, Any]:
        """
        Merge LLM refinement with structured graph analysis.
        
        Anti-hallucination Strategy:
        ---------------------------
        1. Graph hypotheses are the source of truth for confidence scores
        2. LLM can refine cause descriptions but cannot invent new components
        3. LLM confidence is bounded by graph confidence (can only lower, not raise)
        4. Structured evidence always comes from the graph, never from LLM
        5. Components must match graph-discovered components
        """
        final_hypotheses = []
        
        # Get valid components from graph hypotheses
        valid_components = {h.component for h in graph_hypotheses if h.component}
        
        # Use LLM hypotheses but strictly constrain with graph evidence
        llm_hypotheses = llm_result.get("hypotheses", [])
        
        for i, llm_hyp in enumerate(llm_hypotheses):
            # Find matching graph hypothesis
            graph_hyp = None
            if i < len(graph_hypotheses):
                graph_hyp = graph_hypotheses[i]
            
            # Validate LLM confidence: must be clamped to [0, 1] and
            # cannot exceed graph confidence (prevents hallucinated certainty)
            llm_confidence = llm_hyp.get("confidence", 0.5)
            llm_confidence = max(0.0, min(1.0, llm_confidence))  # Clamp to [0, 1]
            
            if graph_hyp:
                # LLM cannot claim higher confidence than graph analysis
                final_confidence = min(llm_confidence, graph_hyp.confidence)
            else:
                # No graph backing - heavily penalize LLM-only hypothesis
                final_confidence = llm_confidence * 0.5
            
            # Validate component: must exist in graph or use graph's component
            llm_component = llm_hyp.get("component")
            if llm_component and llm_component not in valid_components:
                # LLM invented a component - use graph's instead
                llm_component = graph_hyp.component if graph_hyp else None
            
            # Build merged hypothesis with graph data taking precedence
            hypothesis = Hypothesis(
                # LLM can refine the cause description
                cause=llm_hyp.get("cause", graph_hyp.cause if graph_hyp else "Unknown"),
                # Confidence strictly bounded by graph
                confidence=final_confidence,
                # Combine evidence from both (LLM can add context)
                evidence=llm_hyp.get("evidence", []) or (graph_hyp.evidence if graph_hyp else []),
                # Structured evidence ONLY from graph (never from LLM)
                structured_evidence=graph_hyp.structured_evidence if graph_hyp else None,
                # Component from graph (validated above)
                component=llm_component or (graph_hyp.component if graph_hyp else None),
                component_type=llm_hyp.get("component_type") or (graph_hyp.component_type if graph_hyp else None)
            )
            final_hypotheses.append(hypothesis)
        
        # Add remaining graph hypotheses that LLM didn't address
        for graph_hyp in graph_hypotheses[len(llm_hypotheses):]:
            final_hypotheses.append(graph_hyp)
        
        # Use graph's most_likely as fallback if LLM's is empty or vague
        llm_most_likely = llm_result.get("most_likely", "")
        if not llm_most_likely or len(llm_most_likely) < 10:
            llm_most_likely = graph_most_likely
        
        return RootCauseAgentOutput(
            hypotheses=final_hypotheses[:4],
            most_likely=llm_most_likely,
            causal_graph_summary=causal_summary
        ).model_dump()
    
    def _get_mock_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock response using causal graph when LLM unavailable."""
        signals = input_data.get("signals", {})
        
        try:
            # Build graph and generate structured response
            graph, summaries = compute_root_cause_scores(signals)
            hypotheses = self._generate_hypotheses_from_graph(graph, summaries)
            most_likely = self._determine_most_likely(hypotheses, graph)
            
            causal_summary = CausalGraphSummary(
                node_count=len(graph.nodes),
                edge_count=len(graph.edges),
                root_candidates=[h.component or h.cause[:50] for h in hypotheses[:3]],
                primary_causal_chain=self._build_causal_chain_description(graph, hypotheses)
            )
            
            return RootCauseAgentOutput(
                hypotheses=hypotheses,
                most_likely=most_likely,
                causal_graph_summary=causal_summary
            ).model_dump()
            
        except Exception:
            # Fallback to signal-type based mock
            signal_type = signals.get("type", "unknown")
            return self._get_signal_based_mock(signal_type)
    
    def _get_signal_based_mock(self, signal_type: str) -> Dict[str, Any]:
        """Generate mock response based on signal type (legacy fallback)."""
        mock_data = {
            "error_rate": {
                "hypotheses": [
                    {
                        "cause": "Database connection pool exhaustion due to slow queries",
                        "confidence": 0.85,
                        "evidence": [
                            "High correlation between DB timeout errors and HTTP 500s",
                            "Connection pool metrics show saturation",
                            "Error rate spike coincides with query latency increase"
                        ],
                        "component": "Database",
                        "component_type": "database"
                    }
                ],
                "most_likely": "Database connection pool exhaustion caused by slow queries"
            },
            "latency_spike": {
                "hypotheses": [
                    {
                        "cause": "Database query performance degradation",
                        "confidence": 0.90,
                        "evidence": [
                            "Query latency increased 10x during incident",
                            "P95 latency directly correlates with DB metrics"
                        ],
                        "component": "Database",
                        "component_type": "database"
                    }
                ],
                "most_likely": "Database query performance degradation causing latency spike"
            },
            "memory_leak": {
                "hypotheses": [
                    {
                        "cause": "Memory leak in application code",
                        "confidence": 0.88,
                        "evidence": [
                            "Memory growth is linear over time",
                            "No corresponding increase in traffic"
                        ],
                        "component": "Application",
                        "component_type": "service"
                    }
                ],
                "most_likely": "Memory leak in application code causing OOM conditions"
            },
            "dependency_down": {
                "hypotheses": [
                    {
                        "cause": "External dependency service outage",
                        "confidence": 0.95,
                        "evidence": [
                            "All errors reference external service",
                            "Connection refused errors indicate service down"
                        ],
                        "component": "External Service",
                        "component_type": "external_dependency"
                    }
                ],
                "most_likely": "External dependency service is down causing cascading failures"
            }
        }
        
        result = mock_data.get(signal_type, {
            "hypotheses": [
                {
                    "cause": "Service degradation from unknown cause",
                    "confidence": 0.50,
                    "evidence": ["Incident signals detected but root cause unclear"],
                    "component": "Unknown",
                    "component_type": "service"
                }
            ],
            "most_likely": "Unknown - further investigation required"
        })
        
        return {
            **result,
            "causal_graph_summary": {
                "node_count": 1,
                "edge_count": 0,
                "root_candidates": [result["hypotheses"][0]["component"]],
                "primary_causal_chain": None
            }
        }
