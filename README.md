# Multi-Agent Incident Response System

> **SRE × AI** - An intelligent incident response system that uses multiple AI agents powered by **Groq** to automatically detect, diagnose, and generate Root Cause Analysis (RCA) reports.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![Celery](https://img.shields.io/badge/Celery-5.3-37814A.svg)](https://docs.celeryq.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docs.docker.com/compose/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3-orange.svg)](https://groq.com/)

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
- Groq API key (get one free at [console.groq.com](https://console.groq.com))

### 1. Clone and Configure

```bash
cd multi-agent-incident-response

# Create .env file with your Groq API key
echo "GROQ_API_KEY=your-groq-api-key-here" > .env
echo "GROQ_MODEL=llama-3.3-70b-versatile" >> .env
```

### 2. Start All Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f backend worker
```

### 3. Access the System

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:3000 |
| **API Docs** | http://localhost:3000/docs |
| **Demo Service** | http://localhost:3001/health |

### 4. Trigger a Demo Incident

**Option A: Via UI**
1. Go to http://localhost:3000/demo
2. Click "Error Rate Spike" or another fault type
3. Wait 60-90 seconds for detection
4. View the generated incident at http://localhost:3000/incidents

**Option B: Via API**
```bash
# Trigger a latency spike fault
curl -X POST "http://localhost:3000/api/admin/fault" \
  -H "Content-Type: application/json" \
  -d '{"type": "latency_spike", "duration_seconds": 120}'

# Check demo service status
curl http://localhost:3000/api/demo/status

# List incidents (after ~60s)
curl http://localhost:3000/api/incidents
```

### 5. View the RCA Report

```bash
# Get incident details with RCA
curl http://localhost:3000/api/incidents/{incident_id}
```

## 🏃 Running Locally (Without Docker)

### 1. Set Up the Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Create .env file
cp env.example .env

# Edit .env and add your Groq API key
GROQ_API_KEY=your-groq-api-key-here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Start Services

You'll need PostgreSQL and Redis running locally, or use SQLite for development:

```bash
# Start the FastAPI backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# In another terminal, start Celery worker
celery -A app.workers.celery_app worker --loglevel=info

# In another terminal, start Celery beat (scheduler)
celery -A app.workers.celery_app beat --loglevel=info
```

### 4. Run Tests

```bash
cd backend
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
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
│   │   │   ├── base.py       # Base agent class (Groq integration)
│   │   │   ├── monitoring.py
│   │   │   ├── log_analysis.py
│   │   │   ├── root_cause.py
│   │   │   ├── mitigation.py
│   │   │   └── reporter.py
│   │   ├── workers/          # Celery tasks & scheduler
│   │   ├── ui/               # HTML templates (DaisyUI)
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
| `GROQ_API_KEY` | (required) | Groq API key for LLM |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `ERROR_RATE_THRESHOLD` | `0.05` | Error rate detection threshold |
| `LATENCY_P95_THRESHOLD_MS` | `1000` | Latency threshold in ms |
| `MEMORY_THRESHOLD_MB` | `500` | Memory threshold in MB |
| `DATABASE_URL` | PostgreSQL | Database connection string |

### Available Groq Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| `llama-3.3-70b-versatile` | Fast | High | Recommended default |
| `llama-3.1-70b-versatile` | Fast | High | Alternative |
| `llama-3.1-8b-instant` | Ultra fast | Good | Quick analysis |
| `mixtral-8x7b-32768` | Fast | High | Long context |

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
┌─────────────────────────────────────────────────────────────┐
│  ⚡ IncidentAI                     Dashboard │ Incidents │ Demo │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │    2     │ │    1     │ │    5     │ │    ✓     │      │
│  │   Open   │ │  Invest  │ │ Resolved │ │ Groq LLM │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                             │
│  Recent Incidents                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ High Error Rate (15%)     │ SEV2 │ OPEN │ → View   │   │
│  │ Latency Spike (2.5s)      │ SEV3 │ RES  │ → View   │   │
│  │ Memory Usage (600MB)      │ SEV3 │ RES  │ → View   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### RCA Report (AI Generated)
```
┌─────────────────────────────────────────────────────────────┐
│  # Root Cause Analysis Report                               │
│                                                             │
│  ## Executive Summary                                       │
│  High error rate detected at 15%. Root cause:              │
│  Database connection pool exhaustion.                       │
│                                                             │
│  ## Timeline                                                │
│  - T-5m: Error rate began increasing                       │
│  - T-3m: Error rate crossed threshold                      │
│  - T-0m: Incident detected                                 │
│                                                             │
│  ## Root Cause                                              │
│  Database connection pool exhaustion (85% confidence)      │
│                                                             │
│  ## Immediate Actions                                       │
│  ✓ Increase connection pool size                           │
│  ✓ Enable circuit breaker                                  │
│                                                             │
│  ## Prevention                                              │
│  - Add connection pool monitoring                          │
│  - Implement auto-scaling                                  │
│                                                             │
│  Generated by Groq LLaMA 3.3 70B                           │
└─────────────────────────────────────────────────────────────┘
```

## 🛣️ Roadmap

- [x] **Groq Integration** - Fast LLM inference with LLaMA 3.3
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

*Powered by [Groq](https://groq.com) - Ultra-fast LLM inference*
