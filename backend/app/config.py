"""Application configuration."""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "Multi-Agent Incident Response"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/incidents"
    DATABASE_SYNC_URL: str = "postgresql://postgres:postgres@postgres:5432/incidents"
    
    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    
    # Demo Service
    DEMO_SERVICE_URL: str = "http://demo_service:8001"
    
    # Prometheus/Loki (Mode 2)
    PROMETHEUS_URL: str = "http://prometheus:9090"
    LOKI_URL: str = "http://loki:3100"
    
    # Data Mode: "simulated" or "prometheus_loki"
    DATA_MODE: str = "simulated"
    
    # Groq LLM
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    # Detection thresholds
    ERROR_RATE_THRESHOLD: float = 0.05  # 5%
    LATENCY_P95_THRESHOLD_MS: float = 1000  # 1 second
    MEMORY_THRESHOLD_MB: float = 500
    
    # Agent settings
    AGENT_MAX_RETRIES: int = 2
    AGENT_TIMEOUT_SECONDS: int = 60
    AGENT_RATE_LIMIT_RPM: int = 20
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

