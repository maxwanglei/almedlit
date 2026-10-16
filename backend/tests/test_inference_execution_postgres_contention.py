"""Opt-in PostgreSQL races around the inference run execution claim."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from io import BytesIO
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker
from test_inference_execution import DECODER_CONFIG, WINDOW_CONFIG, _RemoteResultBackend
from test_system_administration_postgres_contention import (
    _LOCK_TIMEOUT_SECONDS,
    _THREAD_TIMEOUT_SECONDS,
    _new_user,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)
from test_training_run_execution import MemoryObjectStorage

from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import DocumentSentence
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.evidence.models import EvidenceTarget, EvidenceTargetVersion
from al_medlit.inference import execution, service
from al_medlit.inference.models import EvidenceCandidatePrediction, InferenceRun, InferenceWindow
from al_medlit.lineage.models import (
    AnnotationSet,
    CorpusSnapshot,
    CorpusSnapshotDocument,
    LineageArtifact,
)
from al_medlit.project.models import Project, ProjectTask
from al_medlit.training.compute_profiles import build_compute_backend
from al_medlit.training.models import (
    ComputeProfile,
    ModelCheckpoint,
    TrainingExperiment,
    TrainingJob,
)
from al_medlit.training.windowing import EvidenceBlockWindowBuilder
from al_medlit.workspace import service as workspace_service

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def _synthetic_run(
    db: Session,
    storage: MemoryObjectStorage,
    *,
    status: str,
    profile_config: dict | None = None,
) -> tuple[int, int, dict]:
    """Create a one-document synthetic run plus the remote result its runner would emit."""

    suffix = uuid4().hex
    creator = _new_user(db, "inference-claim")
    workspace = workspace_service.create_team_workspace(db, creator, f"claim-{suffix[:8]}")
    project = Project(
        workspace_id=workspace.id,
        name=f"claim-{suffix[:8]}",
        annotation_schema={"labels": {}},
        settings={},
    )
    db.add(project)
    db.flush()
    task = ProjectTask(
        project_id=project.id,
        annotation_type="evidence_block",
        display_name="Evidence block",
    )
    db.add(task)
    db.flush()
    target = EvidenceTarget(
        project_id=project.id,
        task_id=task.id,
        key=f"target-{suffix}",
        name="Evidence target",
        is_active=True,
        created_by_user_id=creator.id,
    )
    db.add(target)
    db.flush()
    target_version = EvidenceTargetVersion(
        target_id=target.id,
        version_number=1,
        text="Does the treatment help?",
        created_by_user_id=creator.id,
    )
    db.add(target_version)
    db.flush()
    document = corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            text="Alpha improved. Beta improved. Gamma improved. Delta improved.",
        ),
    )
    structure = document.active_structure_version
    sentences = (
        db.query(DocumentSentence)
        .filter(DocumentSentence.structure_version_id == structure.id)
        .order_by(DocumentSentence.ordinal)
        .all()
    )
    checkpoint_object = storage.put_stream(
        f"claim/{suffix}/checkpoint.zip",
        BytesIO(b"synthetic checkpoint"),
        length=len(b"synthetic checkpoint"),
        content_type="application/zip",
    )
    corpus_artifact = LineageArtifact(
        project_id=project.id,
        artifact_type="corpus_snapshot",
        content_hash="c" * 64,
        storage_key=f"claim/{suffix}/corpus.json",
        content_type="application/json",
        size_bytes=1,
        manifest={},
        created_by_user_id=creator.id,
    )
    annotation_artifact = LineageArtifact(
        project_id=project.id,
        artifact_type="annotation_set",
        content_hash="a" * 64,
        storage_key=f"claim/{suffix}/annotations.json",
        content_type="application/json",
        size_bytes=1,
        manifest={},
        created_by_user_id=creator.id,
    )
    checkpoint_artifact = LineageArtifact(
        project_id=project.id,
        artifact_type="model_checkpoint",
        content_hash=checkpoint_object.checksum_sha256,
        storage_key=checkpoint_object.key,
        content_type=checkpoint_object.content_type,
        size_bytes=checkpoint_object.size_bytes,
        manifest={},
        created_by_user_id=creator.id,
    )
    db.add_all([corpus_artifact, annotation_artifact, checkpoint_artifact])
    db.flush()
    snapshot = CorpusSnapshot(
        project_id=project.id,
        artifact_id=corpus_artifact.id,
        name="Claim snapshot",
        document_count=1,
    )
    db.add(snapshot)
    db.flush()
    db.add(
        CorpusSnapshotDocument(
            snapshot_id=snapshot.id,
            document_id=document.id,
            structure_version_id=structure.id,
            split="inference",
            group_key=str(document.id),
            source_hash="s" * 64,
        )
    )
    annotation_set = AnnotationSet(
        project_id=project.id,
        artifact_id=annotation_artifact.id,
        corpus_snapshot_id=snapshot.id,
        name="Claim annotations",
        target_version_ids=[target_version.id],
        block_count=0,
        reviewed_region_count=0,
    )
    profile = ComputeProfile(
        project_id=project.id,
        name="claim-local",
        backend="local",
        config=profile_config or {},
        status="active",
        created_by_user_id=creator.id,
    )
    db.add_all([annotation_set, profile])
    db.flush()
    experiment = TrainingExperiment(
        project_id=project.id,
        annotation_set_id=annotation_set.id,
        compute_profile_id=profile.id,
        name="Claim experiment",
        model_type="evidence_block_sentence_tagger",
        mode="conditioned",
        target_version_ids=[target_version.id],
        config={},
        idempotency_key=f"experiment-{suffix}",
        created_by_user_id=creator.id,
    )
    db.add(experiment)
    db.flush()
    job = TrainingJob(
        experiment_id=experiment.id,
        compute_profile_id=profile.id,
        idempotency_key=f"job-{suffix}",
        status="succeeded",
    )
    db.add(job)
    db.flush()
    checkpoint = ModelCheckpoint(
        project_id=project.id,
        training_job_id=job.id,
        artifact_id=checkpoint_artifact.id,
        model_type="evidence_block_sentence_tagger",
        training_mode="conditioned",
        trained_target_version_ids=[target_version.id],
        max_context_tokens=4096,
        manifest={"synthetic_mode": True},
        readiness="ready",
    )
    db.add(checkpoint)
    db.flush()
    run = InferenceRun(
        project_id=project.id,
        corpus_snapshot_id=snapshot.id,
        checkpoint_id=checkpoint.id,
        compute_profile_id=profile.id,
        name="Claim run",
        target_version_ids=[target_version.id],
        window_config=WINDOW_CONFIG,
        decoder_config=DECODER_CONFIG,
        status=status,
        idempotency_key=f"run-{suffix}",
        created_by_user_id=creator.id,
    )
    db.add(run)
    db.commit()
    remote_result = {
        "schema_version": "inference-window-logits-v1",
        "checkpoint_checksum_sha256": checkpoint_object.checksum_sha256,
        "synthetic_mode": False,
        "windows": [
            {
                "stable_key": "window-0",
                "document_id": document.id,
                "structure_version_id": structure.id,
                "target_version_id": target_version.id,
                "start_sentence_ordinal": 0,
                "end_sentence_ordinal": 3,
                "token_count": 8,
                "logits": {
                    str(sentence.ordinal): logits
                    for sentence, logits in zip(
                        sentences,
                        ([0.0, 6.0, 0.0], [0.0, 0.0, 6.0], [6.0, 0.0, 0.0], [6.0, 0.0, 0.0]),
                        strict=True,
                    )
                },
            }
        ],
    }
    return run.id, project.id, remote_result


class _Worker:
    """Run one operation per named thread in its own session and collect the outcome."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory
        self.results: dict[str, tuple[str, Any]] = {}
        self.backend_pids: dict[str, int] = {}
        self.threads: dict[str, threading.Thread] = {}

    def start(self, role: str, operation: Callable[[Session], Any]) -> None:
        def target() -> None:
            with self.session_factory() as db:
                try:
                    db.execute(text(f"SET lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
                    db.execute(text(f"SET statement_timeout = '{_THREAD_TIMEOUT_SECONDS - 2}s'"))
                    self.backend_pids[role] = db.scalar(text("SELECT pg_backend_pid()"))
                    self.results[role] = ("ok", operation(db))
                    db.commit()
                except Exception as exc:  # noqa: BLE001 - race outcomes are asserted by the test
                    db.rollback()
                    self.results[role] = ("error", exc)

        thread = threading.Thread(target=target, name=role, daemon=True)
        self.threads[role] = thread
        thread.start()

    def join(self, *roles: str) -> None:
        deadline = time.monotonic() + _THREAD_TIMEOUT_SECONDS
        for role in roles:
            self.threads[role].join(max(0, deadline - time.monotonic()))
        assert not any(self.threads[role].is_alive() for role in roles), self.results

    def wait_until_blocked_or_done(self, role: str) -> None:
        """Return once the role waits on a row lock or has finished without waiting."""

        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        with self.session_factory() as observer:
            while time.monotonic() < deadline:
                if not self.threads[role].is_alive():
                    return
                pid = self.backend_pids.get(role)
                if pid is not None and observer.scalar(
                    text("SELECT wait_event_type FROM pg_stat_activity WHERE pid = :pid"),
                    {"pid": pid},
                ) == "Lock":
                    return
                time.sleep(0.01)
        raise AssertionError(f"{role} neither blocked on a lock nor finished: {self.results!r}")


def _artifact_count(db: Session, project_id: int, artifact_type: str) -> int:
    return (
        db.query(LineageArtifact)
        .filter(
            LineageArtifact.project_id == project_id,
            LineageArtifact.artifact_type == artifact_type,
        )
        .count()
    )


def test_concurrent_local_executions_run_once(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
    tmp_path,
) -> None:
    storage = MemoryObjectStorage()
    with postgres_session_factory() as db:
        run_id, project_id, _ = _synthetic_run(db, storage, status="queued")

    first_window_build = threading.Event()
    second_execution_done = threading.Event()
    original_build = EvidenceBlockWindowBuilder.build

    def paused_build(self, *args, **kwargs):
        # The first executor has already read the run's existing window keys here.
        if threading.current_thread().name == "first":
            first_window_build.set()
            assert second_execution_done.wait(timeout=_LOCK_TIMEOUT_SECONDS)
        return original_build(self, *args, **kwargs)

    monkeypatch.setattr(EvidenceBlockWindowBuilder, "build", paused_build)

    def execute(db: Session) -> str:
        return execution.execute_local_inference(
            db,
            storage,
            run_id=run_id,
            work_root=tmp_path / threading.current_thread().name,
        ).status

    worker = _Worker(postgres_session_factory)
    worker.start("first", execute)
    try:
        assert first_window_build.wait(timeout=_LOCK_TIMEOUT_SECONDS), worker.results
        worker.start("second", execute)
        worker.join("second")
    finally:
        second_execution_done.set()
        worker.join("first")

    assert worker.results == {
        "first": ("ok", "succeeded"),
        "second": ("ok", "running"),
    }, worker.results
    with postgres_session_factory() as db:
        run = db.get(InferenceRun, run_id)
        assert run.status == "succeeded"
        assert run.metrics["window_count"] == 1
        assert db.query(InferenceWindow).filter(InferenceWindow.run_id == run_id).count() == 1
        assert _artifact_count(db, project_id, "inference_diagnostics") == 1
        assert (
            db.query(EvidenceCandidatePrediction)
            .filter(EvidenceCandidatePrediction.run_id == run_id)
            .count()
            == run.metrics["candidate_count"]
        )


def test_cancel_during_local_execution_is_not_overwritten(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
    tmp_path,
) -> None:
    storage = MemoryObjectStorage()
    with postgres_session_factory() as db:
        run_id, _, _ = _synthetic_run(
            db,
            storage,
            status="queued",
            profile_config={"runtime_root": str(tmp_path / "attempts")},
        )

    storing_diagnostics = threading.Event()
    cancel_committed = threading.Event()
    original_store = service.store_inference_diagnostics

    def paused_store(*args, **kwargs):
        storing_diagnostics.set()
        assert cancel_committed.wait(timeout=_LOCK_TIMEOUT_SECONDS)
        return original_store(*args, **kwargs)

    monkeypatch.setattr(service, "store_inference_diagnostics", paused_store)

    def execute(db: Session) -> str:
        return execution.execute_local_inference(
            db,
            storage,
            run_id=run_id,
            work_root=tmp_path / "work",
        ).status

    def cancel(db: Session) -> str:
        run = service.get_inference_run(db, run_id)
        return service.cancel_inference_run_with_backend(
            db,
            run_id=run.id,
            backend=build_compute_backend(run.compute_profile),
        ).status

    worker = _Worker(postgres_session_factory)
    worker.start("executor", execute)
    try:
        assert storing_diagnostics.wait(timeout=_LOCK_TIMEOUT_SECONDS), worker.results
        worker.start("cancel", cancel)
        worker.join("cancel")
    finally:
        cancel_committed.set()
        worker.join("executor")

    assert worker.results == {
        "executor": ("ok", "cancelled"),
        "cancel": ("ok", "cancelled"),
    }, worker.results
    with postgres_session_factory() as db:
        run = db.get(InferenceRun, run_id)
        assert run.status == "cancelled"
        assert run.completed_at is not None


def test_concurrent_remote_finalizations_store_one_result(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
    tmp_path,
) -> None:
    storage = MemoryObjectStorage()
    with postgres_session_factory() as db:
        run_id, project_id, remote_result = _synthetic_run(db, storage, status="succeeded")

    first_retrieving = threading.Event()
    resume_first = threading.Event()
    original_retrieve = execution._retrieve_remote_result

    def paused_retrieve(backend, *, run_id, output_root):
        if threading.current_thread().name == "first":
            first_retrieving.set()
            assert resume_first.wait(timeout=_LOCK_TIMEOUT_SECONDS)
        return original_retrieve(backend, run_id=run_id, output_root=output_root)

    monkeypatch.setattr(execution, "_retrieve_remote_result", paused_retrieve)

    def finalize(db: Session) -> int | None:
        return execution.finalize_remote_inference(
            db,
            storage,
            run_id=run_id,
            backend=_RemoteResultBackend(run_id, remote_result),
            work_root=tmp_path / threading.current_thread().name,
        ).diagnostics_artifact_id

    worker = _Worker(postgres_session_factory)
    worker.start("first", finalize)
    try:
        assert first_retrieving.wait(timeout=_LOCK_TIMEOUT_SECONDS), worker.results
        worker.start("second", finalize)
        worker.wait_until_blocked_or_done("second")
    finally:
        resume_first.set()
        worker.join("first", "second")

    assert {role: outcome[0] for role, outcome in worker.results.items()} == {
        "first": "ok",
        "second": "ok",
    }, worker.results
    with postgres_session_factory() as db:
        run = db.get(InferenceRun, run_id)
        first_id, second_id = worker.results["first"][1], worker.results["second"][1]
        assert first_id == second_id == run.diagnostics_artifact_id
        assert _artifact_count(db, project_id, "inference_diagnostics") == 1
        assert _artifact_count(db, project_id, "inference_log") == 1
        assert db.query(InferenceWindow).filter(InferenceWindow.run_id == run_id).count() == 1
        assert (
            db.query(EvidenceCandidatePrediction)
            .filter(EvidenceCandidatePrediction.run_id == run_id)
            .count()
            == run.metrics["candidate_count"]
            == 1
        )
