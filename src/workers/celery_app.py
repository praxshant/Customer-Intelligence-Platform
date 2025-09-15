# src/workers/celery_app.py
import os
from celery import Celery
from celery.schedules import crontab
import httpx

from src.ml.training import train_and_save

REDIS_URL = os.getenv("CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
ML_SERVING_URL = os.getenv("ML_SERVING_URL", "http://cicop-ml-serving:8001")

celery_app = Celery(
    "cicop",
    broker=REDIS_URL,
    backend=RESULT_BACKEND,
)

@celery_app.task
def sample_task(x: int, y: int) -> int:
    return x + y

@celery_app.task
def healthcheck() -> str:
    return "ok"


@celery_app.task
def retrain_model_task() -> dict:
    """Trigger model training and persist artifact to models/model.pkl"""
    result = train_and_save()
    return {"model_path": result.model_path, "metrics": result.metrics}


@celery_app.task
def reload_ml_server_task() -> dict:
    """Call ML server to reload model artifact after retraining"""
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(f"{ML_SERVING_URL}/reload_model")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"error": str(e)}


# Beat schedule: daily at 02:00 server time
celery_app.conf.beat_schedule = {
    "daily-retrain-and-reload": {
        "task": "src.workers.celery_app.retrain_model_task",
        "schedule": crontab(hour=2, minute=0),
    },
    "daily-reload-after-retrain": {
        "task": "src.workers.celery_app.reload_ml_server_task",
        "schedule": crontab(hour=2, minute=30),
    },
}
