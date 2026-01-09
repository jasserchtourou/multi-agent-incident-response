"""
Demo Service - A simulated backend service that generates metrics and logs.

This service simulates a production application with controllable fault injection
for testing the incident response system.

Fault types:
- latency_spike: Adds artificial delay to requests
- error_rate: Randomly returns HTTP 500 errors
- db_slow: Simulates slow database queries
- memory_leak: Simulates growing memory usage
- dependency_down: Simulates external API failures
"""
import asyncio
import random
import time
import json
import logging
import resource
import sys
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

# Configure structured JSON logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        return json.dumps(log_data)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("demo_service")


# ============== Prometheus Metrics ==============

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

MEMORY_USAGE = Gauge(
    'process_memory_usage_bytes',
    'Current memory usage in bytes'
)

ERROR_RATE_GAUGE = Gauge(
    'service_error_rate',
    'Current error rate'
)

ACTIVE_FAULT = Gauge(
    'service_active_fault',
    'Currently active fault (0=none, 1=active)',
    ['fault_type']
)


# ============== State Management ==============

class ServiceState:
    """Manages service state and fault injection."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.current_fault: Optional[str] = None
        self.fault_expires_at: Optional[datetime] = None
        self.fault_params: Dict[str, Any] = {}
        
        # Metrics tracking
        self.request_count = 0
        self.error_count = 0
        self.latencies: deque = deque(maxlen=1000)
        
        # Memory leak simulation
        self.leaked_memory: List[bytes] = []
        
        # Log buffer for querying
        self.log_buffer: deque = deque(maxlen=10000)
    
    def activate_fault(self, fault_type: str, duration_seconds: int = 60, **params):
        """Activate a fault mode."""
        self.current_fault = fault_type
        self.fault_expires_at = datetime.utcnow() + timedelta(seconds=duration_seconds)
        self.fault_params = params
        
        # Update Prometheus gauge
        for ft in ["latency_spike", "error_rate", "db_slow", "memory_leak", "dependency_down"]:
            ACTIVE_FAULT.labels(fault_type=ft).set(1 if ft == fault_type else 0)
        
        logger.info(f"Fault activated: {fault_type}", extra={'extra_data': {
            'fault_type': fault_type,
            'duration': duration_seconds,
            'expires_at': self.fault_expires_at.isoformat()
        }})
    
    def clear_fault(self):
        """Clear the current fault."""
        if self.current_fault:
            logger.info(f"Fault cleared: {self.current_fault}")
        self.current_fault = None
        self.fault_expires_at = None
        self.fault_params = {}
        
        # Clear memory leak
        self.leaked_memory = []
        
        # Update Prometheus gauge
        for ft in ["latency_spike", "error_rate", "db_slow", "memory_leak", "dependency_down"]:
            ACTIVE_FAULT.labels(fault_type=ft).set(0)
    
    def check_fault_expiry(self):
        """Check if fault has expired and clear if so."""
        if self.fault_expires_at and datetime.utcnow() > self.fault_expires_at:
            self.clear_fault()
    
    def is_fault_active(self, fault_type: str) -> bool:
        """Check if a specific fault is active."""
        self.check_fault_expiry()
        return self.current_fault == fault_type
    
    def add_log(self, level: str, message: str, **extra):
        """Add a log entry to the buffer."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **extra
        }
        self.log_buffer.append(entry)
    
    def get_error_rate(self) -> float:
        """Calculate current error rate."""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    def get_latency_p95(self) -> float:
        """Calculate p95 latency in milliseconds."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)] * 1000


state = ServiceState()


# ============== FastAPI App ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Demo service starting")
    yield
    logger.info("Demo service shutting down")


app = FastAPI(
    title="Demo Service",
    description="Simulated backend service for incident response testing",
    version="1.0.0",
    lifespan=lifespan
)


# ============== Middleware ==============

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Calculate latency
    latency = time.time() - start_time
    state.latencies.append(latency)
    state.request_count += 1
    
    if response.status_code >= 500:
        state.error_count += 1
    
    # Update Prometheus metrics
    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=endpoint
    ).observe(latency)
    
    ERROR_RATE_GAUGE.set(state.get_error_rate())
    
    return response


# ============== Health & Monitoring ==============

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    state.check_fault_expiry()
    return {
        "status": "healthy" if not state.current_fault else "degraded",
        "fault": state.current_fault,
        "uptime_seconds": (datetime.utcnow() - state.start_time).total_seconds()
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    # Update memory gauge
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        MEMORY_USAGE.set(usage.ru_maxrss * 1024)  # Convert to bytes
    except:
        pass
    
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/metrics/summary")
async def metrics_summary():
    """Get metrics summary for the incident system."""
    state.check_fault_expiry()
    
    # Calculate memory (simulate if memory leak is active)
    base_memory = 256
    if state.is_fault_active("memory_leak"):
        # Memory grows over time during fault
        elapsed = (datetime.utcnow() - (state.fault_expires_at - timedelta(seconds=60))).total_seconds()
        memory = base_memory + (elapsed * 5)  # Grow 5MB per second
    else:
        memory = base_memory
    
    # Simulate CPU
    cpu_usage = 25
    if state.current_fault:
        cpu_usage = 65  # Higher CPU during faults
    
    return {
        "error_rate": state.get_error_rate(),
        "latency_p95_ms": state.get_latency_p95(),
        "latency_avg_ms": sum(state.latencies) / len(state.latencies) * 1000 if state.latencies else 0,
        "request_count": state.request_count,
        "memory_usage_mb": memory,
        "cpu_usage_percent": cpu_usage,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics/history")
async def metrics_history(
    metric: str = Query(..., description="Metric name"),
    start: str = Query(..., description="Start time ISO"),
    end: str = Query(..., description="End time ISO")
):
    """Get metric history (simulated)."""
    # Return simulated history data
    return {"data": [], "metric": metric}


# ============== Application Endpoints ==============

@app.get("/api/work")
async def do_work(delay: float = Query(0, description="Additional delay in seconds")):
    """
    Simulated work endpoint that can be affected by faults.
    """
    state.check_fault_expiry()
    
    # Apply faults
    if state.is_fault_active("latency_spike"):
        # Add significant delay
        await asyncio.sleep(random.uniform(1.0, 3.0))
        state.add_log("warning", "High latency detected in request processing")
    
    if state.is_fault_active("error_rate"):
        # Randomly fail
        if random.random() < 0.3:  # 30% error rate
            state.add_log("error", "Internal server error", error="RandomFailure")
            raise HTTPException(status_code=500, detail="Internal server error (simulated)")
    
    if state.is_fault_active("db_slow"):
        # Simulate slow DB query
        await asyncio.sleep(random.uniform(0.5, 2.0))
        state.add_log("warning", "Slow database query detected", query_time_ms=random.uniform(500, 2000))
    
    if state.is_fault_active("dependency_down"):
        # Simulate dependency failure
        if random.random() < 0.5:  # 50% of requests fail
            state.add_log("error", "Connection refused to external dependency", 
                         dependency="external-api", error="ConnectionRefused")
            raise HTTPException(status_code=503, detail="Dependency unavailable")
    
    if state.is_fault_active("memory_leak"):
        # Allocate some memory (simulate leak)
        state.leaked_memory.append(bytes(1024 * 100))  # 100KB per request
        state.add_log("debug", "Memory allocated", size_kb=100)
    
    # Normal processing
    await asyncio.sleep(delay + random.uniform(0.01, 0.05))
    
    state.add_log("info", "Request processed successfully", endpoint="/api/work")
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "processing_time_ms": random.uniform(10, 50)
    }


# ============== Logs Endpoints ==============

@app.get("/logs/recent")
async def get_recent_logs(minutes: int = Query(5, description="Minutes of logs to retrieve")):
    """Get recent log entries."""
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    
    logs = [
        log for log in state.log_buffer
        if datetime.fromisoformat(log["timestamp"]) > cutoff
    ]
    
    return {"logs": logs, "count": len(logs)}


@app.get("/logs/search")
async def search_logs(
    query: str = Query("", description="Search query"),
    start: str = Query(..., description="Start time ISO"),
    end: str = Query(..., description="End time ISO"),
    limit: int = Query(100, description="Max logs to return")
):
    """Search log entries."""
    start_time = datetime.fromisoformat(start)
    end_time = datetime.fromisoformat(end)
    
    logs = []
    for log in state.log_buffer:
        log_time = datetime.fromisoformat(log["timestamp"])
        if start_time <= log_time <= end_time:
            if not query or query.lower() in log.get("message", "").lower():
                logs.append(log)
                if len(logs) >= limit:
                    break
    
    return {"logs": logs, "count": len(logs)}


# ============== Admin Endpoints ==============

@app.post("/admin/fault")
async def trigger_fault(
    type: str = Query(..., description="Fault type"),
    duration: int = Query(60, description="Duration in seconds")
):
    """
    Trigger a fault in the demo service.
    
    Fault types:
    - latency_spike: Adds 1-3 second delay to requests
    - error_rate: Random 500 errors (30% rate)
    - db_slow: Simulates slow DB queries (0.5-2s)
    - memory_leak: Memory grows by 100KB per request
    - dependency_down: External API failures (50% rate)
    """
    valid_faults = ["latency_spike", "error_rate", "db_slow", "memory_leak", "dependency_down"]
    
    if type not in valid_faults:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fault type. Must be one of: {valid_faults}"
        )
    
    state.activate_fault(type, duration)
    
    # Generate some error logs to be detected
    for _ in range(5):
        if type == "error_rate":
            state.add_log("error", "HTTP 500 Internal Server Error", status=500)
        elif type == "latency_spike":
            state.add_log("warning", "Request latency exceeded threshold", latency_ms=2500)
        elif type == "db_slow":
            state.add_log("warning", "Slow database query", query_time_ms=1500)
        elif type == "memory_leak":
            state.add_log("warning", "Memory usage increasing", memory_mb=550)
        elif type == "dependency_down":
            state.add_log("error", "Connection timeout to external dependency", 
                         dependency="external-api", error="Timeout")
    
    return {
        "success": True,
        "fault_type": type,
        "duration": duration,
        "expires_at": state.fault_expires_at.isoformat() if state.fault_expires_at else None
    }


@app.post("/admin/clear")
async def clear_fault():
    """Clear any active fault."""
    state.clear_fault()
    return {"success": True, "message": "Fault cleared"}


@app.get("/admin/status")
async def get_status():
    """Get current service status."""
    state.check_fault_expiry()
    
    return {
        "current_fault": state.current_fault,
        "fault_expires_at": state.fault_expires_at.isoformat() if state.fault_expires_at else None,
        "healthy": state.current_fault is None,
        "uptime_seconds": (datetime.utcnow() - state.start_time).total_seconds(),
        "request_count": state.request_count,
        "error_count": state.error_count,
        "error_rate": state.get_error_rate(),
        "latency_p95_ms": state.get_latency_p95()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

