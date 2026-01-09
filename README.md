# Multi-Agent Incident Response System

> **SRE × AI** - An intelligent incident response system that uses multiple AI agents to automatically detect, diagnose, and generate Root Cause Analysis (RCA) reports.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![Celery](https://img.shields.io/badge/Celery-5.3-37814A.svg)](https://docs.celeryq.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)

## 🎯 Overview

This system monitors backend applications, detects incidents from metrics and logs, and orchestrates multiple AI agents in parallel to:

- **Analyze metrics** and identify anomalies
- **Parse logs** to find error patterns
- **Determine root causes** with confidence scores
- **Recommend mitigations** (immediate and long-term)
- **Generate comprehensive RCA reports** automatically

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MULTI-AGENT INCIDENT RESPONSE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐  │
│   │             │     │             │     │      AI AGENTS (Parallel)    │  │
│   │   Demo      │────▶│  Detection  │────▶│  ┌─────────────────────────┐ │  │
│   │   Service   │     │   Service   │     │  │   MonitoringAgent       │ │  │
│   │             │     │             │     │  │   LogAnalysisAgent      │ │  │
│   │  /metrics   │     │  (Celery)   │     │  │   RootCauseAgent        │ │  │
│   │  /logs      │     │             │     │  │   MitigationAgent       │ │  │
│   │  /health    │     │  60s poll   │     │  └─────────────────────────┘ │  │
│   └─────────────┘     └──────┬──────┘     │             │                │  │
│         │                    │            │             ▼                │  │
│         │                    │            │  ┌─────────────────────────┐ │  │
│         ▼                    │            │  │   ReporterAgent         │ │  │
│   ┌─────────────┐            │            │  │   (generates RCA)       │ │  │
│   │  Prometheus │            │            │  └─────────────────────────┘ │  │
│   │  (optional) │            │            └──────────────┬──────────────┘  │
│   └─────────────┘            │                           │                 │
│         │                    │                           │                 │
│         ▼                    ▼                           ▼                 │
│   ┌─────────────┐     ┌─────────────┐           ┌───────────────┐         │
│   │    Loki     │     │  PostgreSQL │◀──────────│   Supervisor  │         │
│   │  (optional) │     │   Database  │           │   (Merges     │         │
│   └─────────────┘     └─────────────┘           │    Results)   │         │
│                              │                  └───────────────┘         │
│                              │                                            │
│                              ▼                                            │
│                       ┌─────────────┐     ┌─────────────┐                │
│                       │   FastAPI   │◀───▶│    Redis    │                │
│                       │   Backend   │     │   (Celery)  │                │
│                       └──────┬──────┘     └─────────────┘                │
│                              │                                            │
│                              ▼                                            │
│                       ┌─────────────┐                                    │
│                       │  Dashboard  │                                    │
│                       │     UI      │                                    │
│                       └─────────────┘                                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Demo Service** generates metrics and logs (simulated production app)
2. **Detection Service** (Celery beat) polls every 60 seconds for anomalies
3. When an incident is detected, **5 AI Agents** run in parallel:
   - `MonitoringAgent` - Analyzes metrics, creates timeline
   - `LogAnalysisAgent` - Clusters errors, finds patterns
   - `RootCauseAgent` - Correlates signals, generates hypotheses
   - `MitigationAgent` - Recommends immediate and long-term fixes
   - `ReporterAgent` - Generates final RCA Markdown report
4. **Supervisor** merges all outputs and updates the incident
5. **Dashboard** displays incidents and RCA reports

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- (Optional) OpenAI API key for real LLM responses

### 1. Clone and Start

```bash
cd multi-agent-incident-response

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend worker
```

### 2. Access the System

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |
| **Demo Service** | http://localhost:8001/health |

### 3. Trigger a Demo Incident

**Option A: Via UI**
1. Go to http://localhost:8000/demo
2. Click "Error Rate Spike" or another fault type
3. Wait 60-90 seconds for detection
4. View the generated incident at http://localhost:8000/incidents

**Option B: Via API**
```bash
# Trigger a latency spike fault
curl -X POST "http://localhost:8000/api/admin/fault" \
  -H "Content-Type: application/json" \
  -d '{"type": "latency_spike", "duration_seconds": 120}'

# Check demo service status
curl http://localhost:8000/api/demo/status

# List incidents (after ~60s)
curl http://localhost:8000/api/incidents
```

### 4. View the RCA Report

```bash
# Get incident details with RCA
curl http://localhost:8000/api/incidents/{incident_id}
```

## 📁 Project Structure

```
multi-agent-incident-response/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes & schemas
│   │   ├── db/               # SQLAlchemy models & session
│   │   ├── services/
│   │   │   ├── detection/    # Anomaly detection rules
│   │   │   ├── metrics/      # Metrics client (Prometheus/simulated)
│   │   │   ├── logs/         # Logs client (Loki/simulated)
│   │   │   └── orchestration/# Supervisor agent coordination
│   │   ├── agents/           # AI agents
│   │   │   ├── prompts/      # Agent prompt templates
│   │   │   ├── schemas.py    # Pydantic output schemas
│   │   │   ├── base.py       # Base agent class
│   │   │   ├── monitoring.py
│   │   │   ├── log_analysis.py
│   │   │   ├── root_cause.py
│   │   │   ├── mitigation.py
│   │   │   └── reporter.py
│   │   ├── workers/          # Celery tasks & scheduler
│   │   ├── ui/               # HTML templates
│   │   └── main.py           # FastAPI app
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── demo_service/             # Simulated backend with fault injection
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── infra/
│   ├── prometheus.yml
│   ├── loki-config.yml
│   └── promtail-config.yml
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_MODE` | `simulated` | `simulated` or `prometheus_loki` |
| `OPENAI_API_KEY` | (none) | OpenAI API key for real LLM |
| `OPENAI_MODEL` | `gpt-4-turbo-preview` | Model to use |
| `ERROR_RATE_THRESHOLD` | `0.05` | Error rate detection threshold |
| `LATENCY_P95_THRESHOLD_MS` | `1000` | Latency threshold in ms |
| `MEMORY_THRESHOLD_MB` | `500` | Memory threshold in MB |

### Using OpenAI (Optional)

To use real LLM responses instead of mock responses:

```bash
# Set your API key in docker-compose or .env
export OPENAI_API_KEY=sk-your-key-here
docker-compose up -d
```

### Enabling Prometheus + Loki

Uncomment the observability services in `docker-compose.yml` and set `DATA_MODE=prometheus_loki`.

## 📊 Data Model

### Incidents Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `status` | Enum | OPEN, INVESTIGATING, RESOLVED |
| `severity` | Enum | SEV1, SEV2, SEV3, SEV4 |
| `title` | String | Incident title |
| `start_time` | DateTime | When incident started |
| `end_time` | DateTime | When resolved |
| `signals_json` | JSON | Metrics/log data |
| `final_summary_json` | JSON | Merged agent outputs |
| `rca_markdown` | Text | Final RCA report |

### Agent Runs Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `incident_id` | UUID | Foreign key |
| `agent_name` | String | Agent identifier |
| `started_at` | DateTime | Execution start |
| `finished_at` | DateTime | Execution end |
| `input_json` | JSON | Agent input |
| `output_json` | JSON | Agent output |
| `latency_ms` | Float | Execution time |

## 🧪 Testing

```bash
# Run tests
cd backend
pip install -r requirements.txt
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 🔌 API Endpoints

### Health & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/demo/status` | Demo service status |

### Incidents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/incidents` | List incidents (filterable) |
| GET | `/api/incidents/{id}` | Get incident details + RCA |
| POST | `/api/incidents/{id}/rerun` | Rerun agent analysis |

### Demo/Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/fault` | Trigger fault in demo service |

## 🎭 Fault Types

| Type | Effect | Detection |
|------|--------|-----------|
| `latency_spike` | 1-3s delay | P95 latency > 1000ms |
| `error_rate` | 30% HTTP 500s | Error rate > 5% |
| `db_slow` | 0.5-2s query delay | Slow query logs |
| `memory_leak` | +100KB/request | Memory > 500MB |
| `dependency_down` | 50% external failures | Connection errors |

## 📸 Screenshots

### Dashboard
```
┌─────────────────────────────────────────────────────┐
│  ⚡ IncidentAI                    Dashboard │ Demo  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐              │
│  │  2   │ │  1   │ │  5   │ │  ✓   │              │
│  │ Open │ │ Inv  │ │ Res  │ │ Demo │              │
│  └──────┘ └──────┘ └──────┘ └──────┘              │
│                                                     │
│  Recent Incidents                                   │
│  ┌─────────────────────────────────────────────┐   │
│  │ High Error Rate (15%)  │ SEV2 │ OPEN │ View │   │
│  │ Latency Spike (2.5s)   │ SEV3 │ RES  │ View │   │
│  │ Memory Usage (600MB)   │ SEV3 │ RES  │ View │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### RCA Report
```
┌─────────────────────────────────────────────────────┐
│  # Root Cause Analysis Report                       │
│                                                     │
│  ## Executive Summary                               │
│  High error rate detected at 15%. Root cause:      │
│  Database connection pool exhaustion.               │
│                                                     │
│  ## Timeline                                        │
│  - T-5m: Error rate began increasing               │
│  - T-3m: Error rate crossed threshold              │
│  - T-0m: Incident detected                         │
│                                                     │
│  ## Root Cause                                      │
│  Database connection pool exhaustion (85% conf)    │
│                                                     │
│  ## Immediate Actions                               │
│  ✓ Increase connection pool size                   │
│  ✓ Enable circuit breaker                          │
│                                                     │
│  ## Prevention                                      │
│  - Add connection pool monitoring                  │
│  - Implement auto-scaling                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## 🛣️ Roadmap

- [ ] **Milestone B**: Enhanced detection with ML-based anomaly detection
- [ ] **Milestone C**: LangGraph integration for agent orchestration
- [ ] **Milestone D**: Grafana dashboard integration
- [ ] **Milestone E**: Slack/PagerDuty notifications
- [ ] **Milestone F**: Historical pattern matching across incidents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for the SRE community**

