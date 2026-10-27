"""Retain scheduler diagnostics before reclaiming completed remote inference jobs."""

import io
import logging
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from al_medlit.core.storage import ObjectStorage
from al_medlit.inference.models import InferenceRun
from al_medlit.inference.service import lock_inference_run
from al_medlit.lineage.service import register_stored_artifact
from al_medlit.training.compute.slurm import SSHSlurmComputeBackend
from al_medlit.training.compute_profiles import build_compute_backend
from al_medlit.training.models import ComputeProfile

logger = logging.getLogger(__name__)


def cleanup_remote_inference_jobs(
    db: Session,
    storage: ObjectStorage,
    *,
    limit: int = 100,
) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=7)
    runs = (
        db.query(InferenceRun)
        .join(ComputeProfile, ComputeProfile.id == InferenceRun.compute_profile_id)
        .filter(
            ComputeProfile.backend == "ssh_slurm",
            InferenceRun.status.in_(("succeeded", "failed", "cancelled")),
            InferenceRun.completed_at < cutoff,
            (
                (InferenceRun.status != "succeeded")
                | (
                    InferenceRun.diagnostics_artifact_id.is_not(None)
                    & InferenceRun.metrics["inference_log_artifact_id"].as_integer().is_not(None)
                )
            ),
            InferenceRun.metrics["remote_cleanup_completed_at"].as_string().is_(None),
        )
        .order_by(InferenceRun.id)
        .limit(min(max(limit, 1), 1000))
        .all()
    )
    cleaned = 0
    for run in runs:
        run_id = run.id
        try:
            run = lock_inference_run(db, run_id)
            if run.status == "succeeded" and (
                run.diagnostics_artifact_id is None
                or run.metrics.get("inference_log_artifact_id") is None
            ):
                db.rollback()
                continue
            backend = build_compute_backend(run.compute_profile)
            if not isinstance(backend, SSHSlurmComputeBackend):
                continue
            if "scheduler_log_artifact_id" not in run.metrics:
                with tempfile.TemporaryDirectory(prefix="al-medlit-scheduler-") as directory:
                    root = Path(directory)
                    backend.collect_scheduler_logs(job_key=f"inference-{run.id}", output_root=root)
                    logs = [
                        f"{name}\n" + (root / name).read_text(encoding="utf-8", errors="replace")
                        for name in ("scheduler.stdout", "scheduler.stderr")
                        if (root / name).is_file()
                    ]
                    if logs:
                        payload = "\n".join(logs).encode()
                        stored = storage.put_stream(
                            f"projects/{run.project_id}/inference/runs/{run.id}/scheduler.log",
                            io.BytesIO(payload),
                            length=len(payload),
                            content_type="text/plain",
                        )
                        artifact = register_stored_artifact(
                            db,
                            project_id=run.project_id,
                            artifact_type="inference_scheduler_log",
                            stored=stored,
                            manifest={"run_id": run.id},
                            created_by_user_id=run.created_by_user_id,
                        )
                        run.metrics = {**run.metrics, "scheduler_log_artifact_id": artifact.id}
                        db.commit()
                        run = lock_inference_run(db, run_id)
            backend.cleanup_job_directory(f"inference-{run.id}")
            run.metrics = {
                **run.metrics,
                "remote_cleanup_completed_at": datetime.now(UTC).isoformat(),
            }
            db.commit()
            cleaned += 1
        except Exception:
            db.rollback()
            logger.exception("Unable to clean remote inference run %s", run_id)
    profiles = db.query(ComputeProfile).filter(ComputeProfile.backend == "ssh_slurm").all()
    for profile in profiles:
        try:
            backend = build_compute_backend(profile)
            if isinstance(backend, SSHSlurmComputeBackend):
                backend.cleanup_expired_base_models()
        except Exception:
            logger.exception("Unable to expire remote base models for profile %s", profile.id)
    return cleaned
