from celery import Celery
from kombu import Queue

from al_medlit.core.config import settings

app = Celery("al_medlit")
app.conf.update(
    broker_connection_retry_on_startup=True,
    broker_url=settings.celery_broker_url,
    enable_utc=True,
    task_always_eager=settings.celery_task_always_eager,
    task_create_missing_queues=True,
    task_default_queue="control",
    task_queues=tuple(
        Queue(name)
        for name in (
            "control",
            "classical-cpu",
            "torch-cpu",
            "transformer-cpu",
            "peft-accelerator",
            "qlora-cuda",
        )
    ),
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    imports=("al_medlit.training.tasks",),
    beat_schedule={
        "reconcile-training-runs": {
            "task": "al_medlit.training.reconcile_training_runs",
            "schedule": 60.0,
            "options": {"queue": "control"},
        },
        "reconcile-feedback-scoring": {
            "task": "al_medlit.workflow.reconcile_feedback_scoring",
            "schedule": 60.0,
            "options": {"queue": "control"},
        },
        "garbage-collect-unreferenced-model-blobs": {
            "task": "al_medlit.model_artifacts.garbage_collect",
            "schedule": 6 * 60 * 60.0,
            "options": {"queue": "control"},
        },
        "expire-abandoned-artifact-reservations": {
            "task": "al_medlit.model_artifacts.expire_reservations",
            "schedule": 60 * 60.0,
            "options": {"queue": "control"},
        },
        "reclaim-orphaned-storage-objects": {
            "task": "al_medlit.storage.reclaim_orphaned_objects",
            "schedule": 15 * 60.0,
            "options": {"queue": "control"},
        },
    },
)

celery_app = app


@app.task(name="al_medlit.tasks.ping")
def ping() -> str:
    return "pong"
