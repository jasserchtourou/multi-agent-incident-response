"""Pytest configuration and fixtures."""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base


# Create test database engine (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def mock_metrics_client():
    """Mock metrics client for testing."""
    with patch('app.services.metrics.client.MetricsClient') as mock:
        instance = mock.return_value
        instance.get_recent_metrics.return_value = {
            "error_rate": 0.01,
            "latency_p95_ms": 100,
            "memory_usage_mb": 256,
            "request_count": 1000
        }
        yield instance


@pytest.fixture
def mock_logs_client():
    """Mock logs client for testing."""
    with patch('app.services.logs.client.LogsClient') as mock:
        instance = mock.return_value
        instance.get_recent_logs.return_value = []
        yield instance


@pytest.fixture
def sample_incident_data():
    """Sample incident data for testing."""
    from datetime import datetime
    return {
        "title": "Test Incident",
        "severity": "SEV3",
        "start_time": datetime.utcnow(),
        "signals_json": {
            "type": "error_rate",
            "error_rate": 0.15,
            "error_count": 50
        }
    }


@pytest.fixture
def sample_agent_outputs():
    """Sample agent outputs for testing merge logic."""
    return {
        "monitoring": {
            "anomaly_summary": "Test anomaly",
            "key_metrics": [],
            "timeline": []
        },
        "log_analysis": {
            "top_errors": [],
            "patterns": [],
            "probable_component": "Test",
            "evidence": []
        },
        "root_cause": {
            "hypotheses": [],
            "most_likely": "Unknown"
        },
        "mitigation": {
            "immediate_actions": [],
            "longer_term_fixes": [],
            "risk_notes": []
        }
    }

