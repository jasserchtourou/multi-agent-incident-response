# Multi-Agent Incident Response System - Detailed Explanation

This document provides a comprehensive explanation of the entire system architecture, every component, and how they work together.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Directory Structure Explained](#3-directory-structure-explained)
4. [Core Components](#4-core-components)
5. [Data Flow](#5-data-flow)
6. [Database Schema](#6-database-schema)
7. [AI Agents Explained](#7-ai-agents-explained)
8. [Demo Service & Fault Injection](#8-demo-service--fault-injection)
9. [Detection System](#9-detection-system)
10. [API Endpoints](#10-api-endpoints)
11. [UI Dashboard](#11-ui-dashboard)
12. [Docker Setup](#12-docker-setup)
13. [Configuration Options](#13-configuration-options)
14. [Testing](#14-testing)
15. [How to Use](#15-how-to-use)

---

## 1. Project Overview

### What Is This System?

This is an **automated incident response system** that combines Site Reliability Engineering (SRE) practices with AI. It:

1. **Monitors** a backend application for problems
2. **Detects** incidents automatically using rules and thresholds
3. **Analyzes** incidents using 5 specialized AI agents running in parallel
4. **Generates** comprehensive Root Cause Analysis (RCA) reports
5. **Displays** everything in a web dashboard

### Why Is This Useful?

In production environments, when something goes wrong:
- Engineers spend hours manually analyzing logs and metrics
- Root cause analysis is slow and error-prone
- Knowledge isn't captured systematically

This system automates that entire process, providing instant analysis and documentation.

### Key Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Main programming language |
| **FastAPI** | Web framework for the API |
| **SQLAlchemy** | Database ORM (Object-Relational Mapping) |
| **PostgreSQL** | Relational database for storing incidents |
| **Celery** | Distributed task queue for background jobs |
| **Redis** | Message broker for Celery |
| **OpenAI/LLM** | AI for agent analysis (optional, has mock mode) |
| **Docker Compose** | Container orchestration |
| **Prometheus** | Metrics collection (optional) |
| **Loki** | Log aggregation (optional) |

---

## 2. Architecture Deep Dive

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. DEMO SERVICE generates metrics & logs (simulates real app)     │
│                              │                                      │
│                              ▼                                      │
│  2. DETECTION SERVICE polls every 60 seconds                       │
│                              │                                      │
│                              ▼                                      │
│  3. When anomaly found → Create INCIDENT in database               │
│                              │                                      │
│                              ▼                                      │
│  4. Trigger 5 AI AGENTS to analyze in parallel                     │
│     ┌─────────────────────────────────────────────┐                │
│     │  MonitoringAgent    │  LogAnalysisAgent     │                │
│     │  RootCauseAgent     │  MitigationAgent      │                │
│     └─────────────────────────────────────────────┘                │
│                              │                                      │
│                              ▼                                      │
│  5. SUPERVISOR merges all outputs                                   │
│                              │                                      │
│                              ▼                                      │
│  6. REPORTER AGENT generates final RCA report                      │
│                              │                                      │
│                              ▼                                      │
│  7. Update incident with RCA → Display in DASHBOARD                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Communication

```
┌──────────────┐     HTTP      ┌──────────────┐
│   Browser    │◄─────────────►│   FastAPI    │
│  (Dashboard) │               │   Backend    │
└──────────────┘               └──────┬───────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
             ┌──────────┐      ┌──────────┐      ┌──────────┐
             │ PostgreSQL│      │  Redis   │      │  Demo    │
             │ (Storage) │      │ (Queue)  │      │ Service  │
             └──────────┘      └────┬─────┘      └──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                   ┌──────────┐          ┌──────────┐
                   │  Celery  │          │  Celery  │
                   │  Worker  │          │   Beat   │
                   │ (Tasks)  │          │(Scheduler)│
                   └──────────┘          └──────────┘
```

---

## 3. Directory Structure Explained

```
multi-agent-incident-response/
│
├── backend/                      # Main application
│   ├── app/                      # Application code
│   │   ├── __init__.py          # Package marker
│   │   ├── config.py            # Configuration & environment variables
│   │   ├── main.py              # FastAPI app entry point
│   │   │
│   │   ├── api/                 # REST API layer
│   │   │   ├── __init__.py
│   │   │   ├── routes.py        # API endpoint definitions
│   │   │   └── schemas.py       # Request/Response Pydantic models
│   │   │
│   │   ├── db/                  # Database layer
│   │   │   ├── __init__.py
│   │   │   ├── models.py        # SQLAlchemy table definitions
│   │   │   └── session.py       # Database connection management
│   │   │
│   │   ├── services/            # Business logic layer
│   │   │   ├── detection/       # Incident detection
│   │   │   │   └── detector.py  # Anomaly detection rules
│   │   │   ├── metrics/         # Metrics data access
│   │   │   │   └── client.py    # Prometheus/Simulated metrics client
│   │   │   ├── logs/            # Logs data access
│   │   │   │   └── client.py    # Loki/Simulated logs client
│   │   │   └── orchestration/   # Agent coordination
│   │   │       └── supervisor.py # Runs agents, merges results
│   │   │
│   │   ├── agents/              # AI Agents
│   │   │   ├── base.py          # Base agent class (shared logic)
│   │   │   ├── schemas.py       # Agent output Pydantic schemas
│   │   │   ├── monitoring.py    # MonitoringAgent
│   │   │   ├── log_analysis.py  # LogAnalysisAgent
│   │   │   ├── root_cause.py    # RootCauseAgent
│   │   │   ├── mitigation.py    # MitigationAgent
│   │   │   ├── reporter.py      # ReporterAgent
│   │   │   └── prompts/         # LLM prompt templates
│   │   │
│   │   ├── workers/             # Background tasks
│   │   │   ├── celery_app.py    # Celery configuration
│   │   │   └── tasks.py         # Task definitions
│   │   │
│   │   └── ui/                  # Web UI
│   │       ├── routes.py        # Page routes
│   │       └── templates/       # HTML templates (Jinja2)
│   │           ├── base.html    # Base layout
│   │           ├── dashboard.html
│   │           ├── incidents.html
│   │           ├── incident_detail.html
│   │           └── demo.html
│   │
│   ├── tests/                   # Test suite
│   │   ├── conftest.py          # Pytest fixtures
│   │   ├── test_detection.py    # Detection tests
│   │   ├── test_schemas.py      # Schema validation tests
│   │   └── test_merge_logic.py  # Supervisor merge tests
│   │
│   ├── Dockerfile               # Container image for backend
│   ├── requirements.txt         # Python dependencies
│   └── pytest.ini               # Pytest configuration
│
├── demo_service/                # Simulated application
│   ├── app.py                   # FastAPI demo service
│   ├── Dockerfile               # Container image
│   └── requirements.txt         # Dependencies
│
├── infra/                       # Infrastructure configs
│   ├── prometheus.yml           # Prometheus scrape config
│   ├── loki-config.yml          # Loki storage config
│   └── promtail-config.yml      # Log shipping config
│
├── scripts/                     # Helper scripts
│   ├── demo.sh                  # Demo script (Linux/Mac)
│   └── demo.ps1                 # Demo script (Windows)
│
├── docker-compose.yml           # Service orchestration
├── README.md                    # Project documentation
├── EXPLANATION.md               # This file
└── .gitignore                   # Git ignore rules
```

---

## 4. Core Components

### 4.1 FastAPI Backend (`backend/app/main.py`)

The main web application that:
- Serves the REST API
- Serves the web dashboard
- Handles database connections
- Configures logging

```python
# Key parts:
app = FastAPI(title="Multi-Agent Incident Response")

# On startup: create database tables
async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)

# Routes
app.include_router(router, prefix="/api")  # API endpoints
app.include_router(ui_router)               # UI pages
```

### 4.2 Database Models (`backend/app/db/models.py`)

Two main tables:

**Incidents Table:**
```python
class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(UUID)                    # Unique identifier
    status = Column(Enum)                # OPEN, INVESTIGATING, RESOLVED
    severity = Column(Enum)              # SEV1, SEV2, SEV3, SEV4
    title = Column(String)               # "High Error Rate Detected (15%)"
    start_time = Column(DateTime)        # When incident started
    end_time = Column(DateTime)          # When resolved
    signals_json = Column(JSON)          # Raw metrics/logs data
    final_summary_json = Column(JSON)    # Merged agent outputs
    rca_markdown = Column(Text)          # Final RCA report
```

**Agent Runs Table:**
```python
class AgentRun(Base):
    __tablename__ = "agent_runs"
    
    id = Column(UUID)
    incident_id = Column(UUID, ForeignKey)  # Links to incident
    agent_name = Column(String)              # "monitoring", "root_cause", etc.
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    input_json = Column(JSON)                # What was sent to agent
    output_json = Column(JSON)               # What agent returned
    latency_ms = Column(Float)               # Execution time
    error_message = Column(Text)             # If agent failed
```

### 4.3 Celery Workers (`backend/app/workers/`)

**Celery App (`celery_app.py`):**
- Configures Celery to use Redis as broker
- Sets up beat schedule for periodic tasks

**Tasks (`tasks.py`):**

```python
# Runs every 60 seconds
@celery_app.task
def detect_incidents():
    detector = IncidentDetector()
    anomalies = detector.detect_anomalies()
    
    for anomaly in anomalies:
        # Check for duplicates (idempotency)
        # Create incident in database
        # Trigger analysis
        run_incident_analysis.delay(incident_id)

# Runs when incident is created or manually triggered
@celery_app.task
def run_incident_analysis(incident_id, agents=None):
    supervisor = Supervisor()
    result = supervisor.analyze_incident(incident_id)
    # Updates incident with RCA
```

### 4.4 Configuration (`backend/app/config.py`)

Uses Pydantic Settings for type-safe configuration:

```python
class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://..."
    
    # Detection thresholds
    ERROR_RATE_THRESHOLD: float = 0.05      # 5%
    LATENCY_P95_THRESHOLD_MS: float = 1000  # 1 second
    MEMORY_THRESHOLD_MB: float = 500
    
    # OpenAI (optional)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Data mode
    DATA_MODE: str = "simulated"  # or "prometheus_loki"
```

---

## 5. Data Flow

### Step-by-Step Flow

```
1. USER triggers fault via UI
   └─► POST /api/admin/fault {"type": "error_rate"}
       └─► Backend proxies to Demo Service
           └─► Demo Service activates fault mode

2. DEMO SERVICE starts generating errors
   └─► 30% of requests return HTTP 500
   └─► Error logs written to buffer
   └─► Prometheus metrics updated

3. CELERY BEAT (every 60s) triggers detection task
   └─► detect_incidents.delay()

4. DETECTION SERVICE runs
   └─► MetricsClient.get_recent_metrics()
       └─► Returns: {"error_rate": 0.15, "latency_p95_ms": 100, ...}
   └─► LogsClient.get_recent_logs()
       └─► Returns: [{"level": "error", "message": "HTTP 500"}, ...]
   └─► IncidentDetector checks all rules:
       └─► error_rate 0.15 > threshold 0.05? YES → anomaly!

5. INCIDENT CREATED in PostgreSQL
   └─► id: "abc-123"
   └─► status: OPEN
   └─► title: "High Error Rate Detected (15%)"
   └─► signals_json: {metrics, logs captured}

6. ANALYSIS TRIGGERED
   └─► run_incident_analysis.delay("abc-123")

7. SUPERVISOR orchestrates agents
   └─► ThreadPoolExecutor(max_workers=4)
   └─► Parallel execution:
       ├─► MonitoringAgent.run(signals)  → {anomaly_summary, timeline}
       ├─► LogAnalysisAgent.run(signals) → {top_errors, patterns}
       ├─► RootCauseAgent.run(signals)   → {hypotheses, most_likely}
       └─► MitigationAgent.run(signals)  → {immediate_actions, fixes}

8. OUTPUTS MERGED
   └─► Supervisor._merge_outputs()
   └─► Combined analysis from all agents

9. REPORTER AGENT generates final RCA
   └─► ReporterAgent.run({merged_outputs})
   └─► Returns Markdown report

10. INCIDENT UPDATED
    └─► status: RESOLVED
    └─► final_summary_json: {merged analysis}
    └─► rca_markdown: "# Root Cause Analysis..."

11. USER views in dashboard
    └─► GET /incidents/abc-123
    └─► Beautiful RCA report displayed
```

---

## 6. Database Schema

### Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         INCIDENTS                            │
├─────────────────────────────────────────────────────────────┤
│ PK  id              UUID                                     │
│     created_at      TIMESTAMP WITH TIME ZONE                 │
│     updated_at      TIMESTAMP WITH TIME ZONE                 │
│     status          ENUM (OPEN, INVESTIGATING, RESOLVED)     │
│     title           VARCHAR(500)                             │
│     severity        ENUM (SEV1, SEV2, SEV3, SEV4)           │
│     start_time      TIMESTAMP WITH TIME ZONE                 │
│     end_time        TIMESTAMP WITH TIME ZONE (nullable)      │
│     signals_json    JSONB                                    │
│     final_summary_json JSONB                                 │
│     rca_markdown    TEXT                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        AGENT_RUNS                            │
├─────────────────────────────────────────────────────────────┤
│ PK  id              UUID                                     │
│ FK  incident_id     UUID → incidents.id                      │
│     agent_name      VARCHAR(100)                             │
│     started_at      TIMESTAMP WITH TIME ZONE                 │
│     finished_at     TIMESTAMP WITH TIME ZONE (nullable)      │
│     input_json      JSONB                                    │
│     output_json     JSONB                                    │
│     tokens_used     INTEGER (nullable)                       │
│     latency_ms      FLOAT (nullable)                         │
│     error_message   TEXT (nullable)                          │
└─────────────────────────────────────────────────────────────┘
```

### Example Data

**Incident Record:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "RESOLVED",
  "title": "High Error Rate Detected (15%)",
  "severity": "SEV2",
  "start_time": "2024-01-15T10:30:00Z",
  "end_time": "2024-01-15T10:35:00Z",
  "signals_json": {
    "type": "error_rate",
    "error_rate": 0.15,
    "error_count": 45,
    "sample_errors": [
      {"level": "error", "message": "HTTP 500 Internal Server Error"}
    ]
  },
  "final_summary_json": {
    "anomaly_summary": "High error rate detected at 15%",
    "most_likely_cause": "Database connection pool exhaustion",
    "immediate_actions": [
      {"action": "Increase connection pool", "priority": "high"}
    ]
  },
  "rca_markdown": "# Root Cause Analysis Report\n\n## Executive Summary..."
}
```

---

## 7. AI Agents Explained

### Agent Architecture

All agents inherit from `BaseAgent`:

```python
class BaseAgent(ABC):
    name: str                    # Agent identifier
    output_schema: Type[BaseModel]  # Pydantic schema for validation
    
    def run(self, input_data):
        # 1. Generate prompt from template
        prompt = self.get_prompt(input_data)
        
        # 2. Call LLM (or mock if no API key)
        response = self._call_llm(prompt)
        
        # 3. Validate response against schema
        validated = self.output_schema.model_validate(response)
        
        return validated.model_dump()
```

### The 5 Agents

#### 1. MonitoringAgent
**Purpose:** Analyze metrics data and create timeline

**Input:**
```json
{"signals": {"error_rate": 0.15, "latency_p95_ms": 2500, ...}}
```

**Output:**
```json
{
  "anomaly_summary": "High error rate detected at 15%. This exceeds normal thresholds.",
  "key_metrics": [
    {"name": "error_rate", "value": 15, "unit": "%", "status": "critical"},
    {"name": "latency_p95", "value": 2500, "unit": "ms", "status": "warning"}
  ],
  "timeline": [
    {"timestamp": "T-5m", "event": "Error rate began increasing", "severity": "warning"},
    {"timestamp": "T-3m", "event": "Error rate crossed threshold", "severity": "critical"}
  ]
}
```

#### 2. LogAnalysisAgent
**Purpose:** Find error patterns in logs

**Input:**
```json
{"signals": {"sample_errors": [{"level": "error", "message": "DB timeout"}]}}
```

**Output:**
```json
{
  "top_errors": [
    {"message": "HTTP 500 Internal Server Error", "count": 45},
    {"message": "Connection timeout to database", "count": 23}
  ],
  "patterns": [
    {"pattern": "Database connection failures correlate with HTTP 500s", "frequency": 68, "significance": "high"}
  ],
  "probable_component": "Database connection pool",
  "evidence": ["68% of errors mention database", "Error spike correlates with latency"]
}
```

#### 3. RootCauseAgent
**Purpose:** Identify why the incident happened

**Input:** Signals from incident

**Output:**
```json
{
  "hypotheses": [
    {
      "cause": "Database connection pool exhaustion",
      "confidence": 0.85,
      "evidence": ["High DB timeout errors", "Connection pool at capacity"]
    },
    {
      "cause": "Memory pressure causing GC pauses",
      "confidence": 0.45,
      "evidence": ["Elevated memory during incident"]
    }
  ],
  "most_likely": "Database connection pool exhaustion caused by slow queries"
}
```

#### 4. MitigationAgent
**Purpose:** Recommend what to do

**Input:** Signals from incident

**Output:**
```json
{
  "immediate_actions": [
    {
      "action": "Increase database connection pool size",
      "priority": "high",
      "estimated_impact": "Should restore connectivity in 2-3 minutes"
    },
    {
      "action": "Enable circuit breaker for database calls",
      "priority": "high",
      "estimated_impact": "Prevents cascade failures"
    }
  ],
  "longer_term_fixes": [
    {
      "action": "Implement connection pool monitoring",
      "priority": "high",
      "estimated_impact": "Prevents future incidents"
    }
  ],
  "risk_notes": [
    "Increasing pool may impact database performance"
  ]
}
```

#### 5. ReporterAgent
**Purpose:** Generate final human-readable RCA

**Input:** All merged agent outputs

**Output:**
```json
{
  "markdown_report": "# Root Cause Analysis Report\n\n## Executive Summary\n...",
  "executive_summary": "Service disruption due to database connection exhaustion.",
  "action_items": [
    "Increase connection pool size",
    "Add monitoring alerts",
    "Schedule post-incident review"
  ]
}
```

### Mock Mode vs Real LLM

If `OPENAI_API_KEY` is not set, agents use **mock responses**:

```python
def _get_mock_response(self, input_data):
    # Returns pre-defined response based on signal type
    signal_type = input_data.get("signals", {}).get("type")
    
    if signal_type == "error_rate":
        return {
            "anomaly_summary": "High error rate detected...",
            # ... structured mock data
        }
```

This allows the system to work without any AI API costs for demos.

---

## 8. Demo Service & Fault Injection

### Demo Service (`demo_service/app.py`)

A simulated production application that:
- Exposes `/health`, `/api/work` endpoints
- Generates Prometheus metrics at `/metrics`
- Stores logs in memory buffer
- Supports fault injection

### Fault Types Explained

| Fault | What It Does | Detection Trigger |
|-------|--------------|-------------------|
| `latency_spike` | Adds 1-3s delay to requests | P95 latency > 1000ms |
| `error_rate` | 30% of requests return 500 | Error rate > 5% |
| `db_slow` | Simulates 0.5-2s DB queries | Slow query logs |
| `memory_leak` | Allocates 100KB per request | Memory > 500MB |
| `dependency_down` | 50% external API failures | Connection error logs |

### How Faults Work

```python
@app.get("/api/work")
async def do_work():
    # Check if fault is active
    if state.is_fault_active("latency_spike"):
        await asyncio.sleep(random.uniform(1.0, 3.0))  # Add delay
        
    if state.is_fault_active("error_rate"):
        if random.random() < 0.3:  # 30% chance
            raise HTTPException(status_code=500)
    
    # Normal processing
    return {"status": "ok"}
```

### Activating Faults

```bash
# Via API
curl -X POST "http://localhost:8000/api/admin/fault" \
  -H "Content-Type: application/json" \
  -d '{"type": "error_rate", "duration_seconds": 120}'

# The demo service will:
# 1. Set current_fault = "error_rate"
# 2. Set fault_expires_at = now + 120 seconds
# 3. Start generating errors
```

---

## 9. Detection System

### Detection Service (`backend/app/services/detection/detector.py`)

Runs every 60 seconds to find anomalies:

```python
class IncidentDetector:
    def detect_anomalies(self):
        anomalies = []
        
        # Get data
        metrics = self.metrics_client.get_recent_metrics(minutes=5)
        logs = self.logs_client.get_recent_logs(minutes=5)
        
        # Check each rule
        if metrics["error_rate"] > self.error_rate_threshold:
            anomalies.append({
                "title": f"High Error Rate ({metrics['error_rate']:.1%})",
                "severity": "SEV2" if metrics["error_rate"] > 0.1 else "SEV3",
                "signals": {"type": "error_rate", ...}
            })
        
        if metrics["latency_p95_ms"] > self.latency_threshold:
            anomalies.append({...})
        
        # ... more rules
        
        return anomalies
```

### Detection Rules

| Rule | Threshold | Severity |
|------|-----------|----------|
| Error Rate | > 5% | SEV3 (SEV2 if >10%) |
| P95 Latency | > 1000ms | SEV3 (SEV2 if >2000ms) |
| Memory Usage | > 500MB | SEV3 (SEV2 if >800MB) |
| Dependency Errors | ≥ 3 errors | SEV2 |

### Idempotency

To prevent duplicate incidents:

```python
# Check for existing incident with same title in 5-minute window
existing = session.query(Incident).filter(
    Incident.title == anomaly["title"],
    Incident.start_time >= window_start,
    Incident.start_time <= window_end,
    Incident.status != "RESOLVED"
).first()

if existing:
    logger.info("Duplicate incident skipped")
    continue  # Don't create new incident
```

---

## 10. API Endpoints

### Health & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System health check |
| `GET` | `/api/demo/status` | Demo service status |

### Incidents CRUD

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/incidents` | List incidents (supports `?status=OPEN&severity=SEV1&page=1&size=20`) |
| `GET` | `/api/incidents/{id}` | Get incident with RCA |
| `POST` | `/api/incidents` | Create incident manually |
| `PATCH` | `/api/incidents/{id}` | Update incident |
| `DELETE` | `/api/incidents/{id}` | Delete incident |
| `POST` | `/api/incidents/{id}/rerun` | Re-run agent analysis |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/admin/fault` | Trigger fault in demo service |

### Example Requests

```bash
# List open incidents
curl "http://localhost:8000/api/incidents?status=OPEN"

# Get incident detail with RCA
curl "http://localhost:8000/api/incidents/abc-123"

# Trigger fault
curl -X POST "http://localhost:8000/api/admin/fault" \
  -H "Content-Type: application/json" \
  -d '{"type": "latency_spike", "duration_seconds": 60}'

# Re-run analysis
curl -X POST "http://localhost:8000/api/incidents/abc-123/rerun"
```

---

## 11. UI Dashboard

### Pages

| Page | URL | Purpose |
|------|-----|---------|
| Dashboard | `/` | Overview, stats, recent incidents |
| Incidents | `/incidents` | Full list with filtering |
| Incident Detail | `/incidents/{id}` | Full RCA report |
| Demo Control | `/demo` | Trigger faults |

### Technology

- **Jinja2 Templates** - Server-side rendering
- **Vanilla JavaScript** - No framework, fetches from API
- **Custom CSS** - Dark theme, modern design

### Key Features

1. **Dashboard**
   - Stat cards (Open, Investigating, Resolved counts)
   - Recent incidents table
   - Demo service status

2. **Incidents List**
   - Filter by status and severity
   - Pagination
   - Quick actions (View, Analyze)

3. **Incident Detail**
   - Timeline info (start, end, duration)
   - Full RCA report (Markdown rendered)
   - Signals data (JSON)
   - Agent runs list

4. **Demo Control**
   - Fault type cards with descriptions
   - Duration selector
   - Real-time status display

---

## 12. Docker Setup

### Services in `docker-compose.yml`

```yaml
services:
  postgres:      # Database
  redis:         # Celery broker
  backend:       # FastAPI app
  worker:        # Celery worker
  beat:          # Celery scheduler
  demo_service:  # Simulated app
```

### Service Dependencies

```
postgres ───┐
            ├──► backend ───► worker ───► beat
redis ──────┘                    │
                                 │
demo_service ◄───────────────────┘
```

### Volumes

```yaml
volumes:
  postgres_data:   # Persistent database storage
  redis_data:      # Persistent queue storage
```

### Networks

All services on `incident_network` for internal communication.

### Starting Everything

```bash
cd multi-agent-incident-response
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend worker
```

---

## 13. Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | (docker default) | PostgreSQL connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |
| `DATA_MODE` | `simulated` | `simulated` or `prometheus_loki` |
| `OPENAI_API_KEY` | (empty) | For real LLM responses |
| `OPENAI_MODEL` | `gpt-4-turbo-preview` | Which model to use |
| `ERROR_RATE_THRESHOLD` | `0.05` | 5% error rate triggers |
| `LATENCY_P95_THRESHOLD_MS` | `1000` | 1s latency triggers |
| `MEMORY_THRESHOLD_MB` | `500` | Memory alert threshold |

### Using Real AI

To use actual GPT-4 responses:

```bash
# Set in docker-compose.yml or .env
OPENAI_API_KEY=sk-your-api-key-here

# Restart services
docker-compose down
docker-compose up -d
```

### Enabling Prometheus/Loki

1. Uncomment observability services in `docker-compose.yml`
2. Set `DATA_MODE=prometheus_loki`
3. Restart

---

## 14. Testing

### Test Files

| File | What It Tests |
|------|---------------|
| `test_detection.py` | Detection rules & thresholds |
| `test_schemas.py` | Pydantic schema validation |
| `test_merge_logic.py` | Supervisor output merging |
| `conftest.py` | Shared fixtures |

### Running Tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

### Example Test

```python
def test_detect_error_rate_spike(self):
    """Test detection of high error rates."""
    metrics = {"error_rate": 0.15}  # 15%, above 5% threshold
    logs = [{"level": "error", "message": "HTTP 500"}]
    
    anomaly = detector._detect_error_rate_spike(metrics, logs, now)
    
    assert anomaly is not None
    assert "Error Rate" in anomaly["title"]
    assert anomaly["severity"] == "SEV2"  # > 10% = SEV2
```

---

## 15. How to Use

### Quick Start

```bash
# 1. Start services
cd multi-agent-incident-response
docker-compose up -d

# 2. Wait 30 seconds for startup

# 3. Open dashboard
# http://localhost:8000

# 4. Trigger a fault
# Go to http://localhost:8000/demo
# Click "Error Rate Spike"

# 5. Wait 60-90 seconds for detection

# 6. View incident
# http://localhost:8000/incidents
# Click on the detected incident
# See the full RCA report!
```

### Demo Script (Windows)

```powershell
cd multi-agent-incident-response
.\scripts\demo.ps1 demo
```

### Demo Script (Linux/Mac)

```bash
cd multi-agent-incident-response
chmod +x scripts/demo.sh
./scripts/demo.sh demo
```

### Manual API Demo

```bash
# 1. Check health
curl http://localhost:8000/api/health

# 2. Check demo service
curl http://localhost:8000/api/demo/status

# 3. Trigger fault
curl -X POST http://localhost:8000/api/admin/fault \
  -H "Content-Type: application/json" \
  -d '{"type": "error_rate", "duration_seconds": 120}'

# 4. Wait 60-90 seconds, then list incidents
curl http://localhost:8000/api/incidents

# 5. Get incident details (replace ID)
curl http://localhost:8000/api/incidents/{incident-id}
```

---

## Summary

This system provides:

1. ✅ **Automated incident detection** from metrics and logs
2. ✅ **Parallel AI agent analysis** for fast diagnosis
3. ✅ **Comprehensive RCA reports** generated automatically
4. ✅ **Beautiful web dashboard** for visualization
5. ✅ **Demo mode** with fault injection for testing
6. ✅ **Production-ready architecture** with Docker, Celery, PostgreSQL

The entire flow from fault to RCA report takes about 60-90 seconds, demonstrating how AI can accelerate incident response!

---

**Questions?** Check the README.md or API docs at http://localhost:8000/docs

