"""Celery application configuration."""
from celery import Celery
from app.config import settings

celery_app = Celery(
    "incident_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # Soft limit at 4 minutes
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    
    # Result settings
    result_expires=3600,  # Results expire after 1 hour
    
    # Retry settings
    task_default_retry_delay=30,
    task_max_retries=3,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "detect-incidents-every-60s": {
            "task": "app.workers.tasks.detect_incidents",
            "schedule": 60.0,  # Every 60 seconds
        },
        "cleanup-old-data-daily": {
            "task": "app.workers.tasks.cleanup_old_data",
            "schedule": 86400.0,  # Every 24 hours
        },
    },
)

