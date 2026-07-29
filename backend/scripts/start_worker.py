"""Verify an optional runtime before starting its dedicated Celery queue."""

from __future__ import annotations

import json
import os
from pathlib import Path

from al_medlit.training.runtime_profiles import RUNTIME_PROFILES
from scripts.runtime_preflight import (
    collect_runtime_readiness,
    compute_profile_fragment,
    normalize_worker_image_digest,
)


def main() -> None:
    profile_key = os.environ.get("AL_MEDLIT_RUNTIME_PROFILE", "")
    if profile_key not in RUNTIME_PROFILES:
        raise SystemExit("AL_MEDLIT_RUNTIME_PROFILE must name a supported worker runtime")
    descriptor = RUNTIME_PROFILES[profile_key]
    queue = os.environ.get("AL_MEDLIT_WORKER_QUEUE", descriptor.queue)
    if queue != descriptor.queue:
        raise SystemExit(
            f"Worker queue {queue!r} does not match runtime profile {profile_key!r}"
        )

    image_digest = normalize_worker_image_digest(
        os.environ.get("AL_MEDLIT_WORKER_IMAGE_DIGEST")
    )
    report = collect_runtime_readiness(
        profile_key,
        scratch_root=os.environ.get(
            "AL_MEDLIT_LOCAL_ATTEMPT_ROOT",
            "/var/lib/al-medlit/attempts",
        ),
        worker_image_digest=image_digest,
    )
    report_path = Path(
        os.environ.get(
            "AL_MEDLIT_RUNTIME_REPORT_PATH",
            "/var/lib/al-medlit/runtime-report.json",
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(compute_profile_fragment(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)
    if not report.ready:
        missing = ", ".join(report.missing_dependencies) or "none"
        image_attestation = "present" if report.worker_image_digest else "missing"
        raise SystemExit(
            f"Runtime {profile_key} is not ready "
            f"(missing dependencies: {missing}; device: {report.device}; "
            f"storage: {report.storage_access_verified}; "
            f"image digest: {image_attestation})"
        )

    os.execvp(
        "celery",
        (
            "celery",
            "-A",
            "al_medlit.celery_app:app",
            "worker",
            "--loglevel=info",
            f"--queues={queue}",
        ),
    )


if __name__ == "__main__":
    main()
