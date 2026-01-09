"""Supervisor for orchestrating multi-agent incident analysis."""
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from app.config import settings
from app.db.models import Incident, AgentRun
from app.agents.base import BaseAgent
from app.agents.monitoring import MonitoringAgent
from app.agents.log_analysis import LogAnalysisAgent
from app.agents.root_cause import RootCauseAgent
from app.agents.mitigation import MitigationAgent
from app.agents.reporter import ReporterAgent

logger = structlog.get_logger()

# Create sync engine for supervisor
sync_engine = create_engine(settings.DATABASE_SYNC_URL, pool_pre_ping=True)


class Supervisor:
    """
    Orchestrates multiple AI agents to analyze incidents in parallel.
    
    The supervisor:
    1. Loads incident data and signals
    2. Runs agents in parallel (MonitoringAgent, LogAnalysisAgent, RootCauseAgent, MitigationAgent)
    3. Merges agent outputs
    4. Runs ReporterAgent to generate final RCA
    5. Updates incident with results
    """
    
    # Agent registry
    AGENTS = {
        "monitoring": MonitoringAgent,
        "log_analysis": LogAnalysisAgent,
        "root_cause": RootCauseAgent,
        "mitigation": MitigationAgent,
    }
    
    def __init__(self):
        self.max_workers = 4  # Run up to 4 agents in parallel
    
    def analyze_incident(
        self,
        incident_id: str,
        agents_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run multi-agent analysis for an incident.
        
        Args:
            incident_id: UUID of the incident
            agents_to_run: Optional list of specific agent names to run
        
        Returns:
            Dictionary with merged agent outputs and RCA
        """
        logger.info("Starting incident analysis", incident_id=incident_id)
        
        # Load incident data
        with Session(sync_engine) as session:
            incident = session.execute(
                select(Incident).where(Incident.id == UUID(incident_id))
            ).scalar_one_or_none()
            
            if not incident:
                raise ValueError(f"Incident {incident_id} not found")
            
            signals = incident.signals_json or {}
            incident_title = incident.title
        
        # Determine which agents to run
        if agents_to_run:
            agent_names = [a for a in agents_to_run if a in self.AGENTS]
        else:
            agent_names = list(self.AGENTS.keys())
        
        # Run agents in parallel
        agent_outputs = self._run_agents_parallel(
            incident_id=incident_id,
            agent_names=agent_names,
            signals=signals
        )
        
        # Merge outputs
        merged_analysis = self._merge_outputs(agent_outputs)
        
        # Generate RCA report
        reporter = ReporterAgent()
        rca_result = self._run_agent(
            agent=reporter,
            incident_id=incident_id,
            agent_name="reporter",
            input_data={
                "incident_title": incident_title,
                "signals": signals,
                "agent_outputs": merged_analysis
            }
        )
        
        # Build final result
        final_result = {
            "summary": merged_analysis,
            "rca_markdown": rca_result.get("markdown_report", ""),
            "executive_summary": rca_result.get("executive_summary", ""),
            "action_items": rca_result.get("action_items", []),
            "agent_outputs": agent_outputs
        }
        
        logger.info("Incident analysis complete", incident_id=incident_id)
        return final_result
    
    def _run_agents_parallel(
        self,
        incident_id: str,
        agent_names: List[str],
        signals: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Run multiple agents in parallel using ThreadPoolExecutor."""
        outputs = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all agent tasks
            futures = {}
            for agent_name in agent_names:
                agent_class = self.AGENTS[agent_name]
                agent = agent_class()
                
                future = executor.submit(
                    self._run_agent,
                    agent=agent,
                    incident_id=incident_id,
                    agent_name=agent_name,
                    input_data={"signals": signals}
                )
                futures[future] = agent_name
            
            # Collect results
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result()
                    outputs[agent_name] = result
                    logger.info("Agent completed", agent=agent_name, incident_id=incident_id)
                except Exception as e:
                    logger.error("Agent failed", agent=agent_name, error=str(e))
                    outputs[agent_name] = {"error": str(e)}
        
        return outputs
    
    def _run_agent(
        self,
        agent: BaseAgent,
        incident_id: str,
        agent_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single agent and record the execution."""
        started_at = datetime.utcnow()
        
        # Create agent run record
        with Session(sync_engine) as session:
            agent_run = AgentRun(
                incident_id=UUID(incident_id),
                agent_name=agent_name,
                started_at=started_at,
                input_json=input_data
            )
            session.add(agent_run)
            session.commit()
            agent_run_id = agent_run.id
        
        try:
            # Run the agent
            result = agent.run(input_data)
            finished_at = datetime.utcnow()
            latency_ms = (finished_at - started_at).total_seconds() * 1000
            
            # Update agent run record
            with Session(sync_engine) as session:
                agent_run = session.get(AgentRun, agent_run_id)
                if agent_run:
                    agent_run.finished_at = finished_at
                    agent_run.output_json = result
                    agent_run.latency_ms = latency_ms
                    session.commit()
            
            return result
        
        except Exception as e:
            # Record error
            with Session(sync_engine) as session:
                agent_run = session.get(AgentRun, agent_run_id)
                if agent_run:
                    agent_run.finished_at = datetime.utcnow()
                    agent_run.error_message = str(e)
                    session.commit()
            raise
    
    def _merge_outputs(self, agent_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Merge outputs from multiple agents into a unified analysis."""
        merged = {
            "anomaly_summary": "",
            "key_metrics": [],
            "timeline": [],
            "top_errors": [],
            "error_patterns": [],
            "probable_component": "",
            "hypotheses": [],
            "most_likely_cause": "",
            "immediate_actions": [],
            "longer_term_fixes": [],
            "risk_notes": []
        }
        
        # Merge monitoring agent output
        if "monitoring" in agent_outputs and "error" not in agent_outputs["monitoring"]:
            monitoring = agent_outputs["monitoring"]
            merged["anomaly_summary"] = monitoring.get("anomaly_summary", "")
            merged["key_metrics"] = monitoring.get("key_metrics", [])
            merged["timeline"].extend(monitoring.get("timeline", []))
        
        # Merge log analysis output
        if "log_analysis" in agent_outputs and "error" not in agent_outputs["log_analysis"]:
            log_analysis = agent_outputs["log_analysis"]
            merged["top_errors"] = log_analysis.get("top_errors", [])
            merged["error_patterns"] = log_analysis.get("patterns", [])
            merged["probable_component"] = log_analysis.get("probable_component", "")
        
        # Merge root cause output
        if "root_cause" in agent_outputs and "error" not in agent_outputs["root_cause"]:
            root_cause = agent_outputs["root_cause"]
            merged["hypotheses"] = root_cause.get("hypotheses", [])
            merged["most_likely_cause"] = root_cause.get("most_likely", "")
        
        # Merge mitigation output
        if "mitigation" in agent_outputs and "error" not in agent_outputs["mitigation"]:
            mitigation = agent_outputs["mitigation"]
            merged["immediate_actions"] = mitigation.get("immediate_actions", [])
            merged["longer_term_fixes"] = mitigation.get("longer_term_fixes", [])
            merged["risk_notes"] = mitigation.get("risk_notes", [])
        
        return merged

