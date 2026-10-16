"""Opt-in PostgreSQL races between workflow transitions and stale row reads."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker
from test_system_administration_postgres_contention import (
    _LOCK_TIMEOUT_SECONDS,
    _THREAD_TIMEOUT_SECONDS,
    _new_user,
    _set_transaction_timeouts,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)

from al_medlit.auth.models import User
from al_medlit.core.exceptions import ConflictError, ValidationError
from al_medlit.project.models import Project
from al_medlit.workflow import models, schemas
from al_medlit.workflow.services import cycles, guidelines, rounds, selection
from al_medlit.workspace import service as workspace_service

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


@dataclass(frozen=True, slots=True)
class _Scope:
    project_id: int
    manager_id: int
    annotator_ids: tuple[int, int]
    task_definition_id: int
    task_version_id: int
    dataset_version_id: int
    dataset_item_id: int


def _scope(db: Session) -> _Scope:
    manager = _new_user(db, "stale-lock-manager")
    annotators = (_new_user(db, "stale-lock-left"), _new_user(db, "stale-lock-right"))
    workspace = workspace_service.create_team_workspace(db, manager, "Stale lock contention")
    project = Project(name=f"Stale lock {uuid4().hex[:8]}", workspace_id=workspace.id)
    db.add(project)
    db.flush()
    task_definition = models.TaskDefinition(
        project_id=project.id, key="stale-lock-task", name="Stale Lock Task"
    )
    dataset = models.Dataset(project_id=project.id, name="Stale Lock Dataset", source_type="test")
    db.add_all([task_definition, dataset])
    db.flush()
    task_version = models.TaskVersion(
        project_id=project.id,
        task_definition_id=task_definition.id,
        version_number=1,
        task_kind="classification",
        content_hash=uuid4().hex,
    )
    dataset_version = models.DatasetVersion(
        project_id=project.id,
        dataset_id=dataset.id,
        version_number=1,
        source_revision="stale-lock-v1",
        source_format="jsonl",
        content_hash=uuid4().hex,
        item_count=1,
    )
    db.add_all([task_version, dataset_version])
    db.flush()
    dataset_item = models.DatasetItem(
        project_id=project.id,
        dataset_version_id=dataset_version.id,
        stable_key="stale-lock-item",
        payload={"text": "stale lock"},
        content_hash=uuid4().hex,
    )
    db.add(dataset_item)
    db.commit()
    return _Scope(
        project_id=project.id,
        manager_id=manager.id,
        annotator_ids=(annotators[0].id, annotators[1].id),
        task_definition_id=task_definition.id,
        task_version_id=task_version.id,
        dataset_version_id=dataset_version.id,
        dataset_item_id=dataset_item.id,
    )


def _round(db: Session, scope: _Scope, **values: Any) -> models.AnnotationRound:
    annotation_round = models.AnnotationRound(
        project_id=scope.project_id,
        name=f"Stale lock round {uuid4().hex[:8]}",
        sequence=1,
        dataset_version_id=scope.dataset_version_id,
        task_version_id=scope.task_version_id,
        reannotation_mode="full_dataset",
        annotator_user_ids=list(scope.annotator_ids),
        created_by_user_id=scope.manager_id,
        **values,
    )
    db.add(annotation_round)
    db.flush()
    return annotation_round


def _attempt(
    session_factory: sessionmaker[Session], operation: Callable[[Session], Any]
) -> tuple[str, Any]:
    with session_factory() as db:
        _set_transaction_timeouts(db)
        try:
            return ("ok", operation(db))
        except (ConflictError, ValidationError) as exc:
            db.rollback()
            return (type(exc).__name__, str(exc))


def _pause_after_load(
    monkeypatch, module: ModuleType, model_type: type
) -> tuple[threading.Event, threading.Event]:
    loaded = threading.Event()
    resume = threading.Event()
    original_scoped = module._scoped

    def paused_scoped(db, loaded_type, *args, **kwargs):
        record = original_scoped(db, loaded_type, *args, **kwargs)
        if loaded_type is model_type and db.info.pop("pause_after_load", False):
            loaded.set()
            assert resume.wait(timeout=_LOCK_TIMEOUT_SECONDS)
        return record

    monkeypatch.setattr(module, "_scoped", paused_scoped)
    return loaded, resume


def _commit_between_load_and_write(
    session_factory: sessionmaker[Session],
    *,
    stale: Callable[[Session], Any],
    winner: Callable[[Session], Any],
    loaded: threading.Event,
    resume: threading.Event,
) -> tuple[tuple[str, Any], tuple[str, Any]]:
    """Commit ``winner`` after ``stale`` loaded its row and before it locks or writes."""

    def paused_stale(db: Session) -> Any:
        db.info["pause_after_load"] = True
        return stale(db)

    with ThreadPoolExecutor(max_workers=1) as executor:
        stale_future = executor.submit(_attempt, session_factory, paused_stale)
        try:
            assert loaded.wait(timeout=_LOCK_TIMEOUT_SECONDS), "The stale writer never loaded"
            winner_result = _attempt(session_factory, winner)
        finally:
            resume.set()
        return winner_result, stale_future.result(timeout=_THREAD_TIMEOUT_SECONDS)


def _wait_until_finished_or_blocked(
    session_factory: sessionmaker[Session],
    pids: dict[str, int],
    finished: threading.Event,
) -> None:
    """Return once the waiter finished or waits on a lock that the holder owns."""
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    with session_factory() as observer:
        while not finished.is_set():
            if "waiter" in pids and observer.scalar(
                text("SELECT :holder = ANY(pg_blocking_pids(:waiter))"),
                pids,
            ):
                return
            assert time.monotonic() < deadline, "The waiter neither finished nor blocked"
            time.sleep(0.01)


def _commit_after_concurrent_attempt(
    session_factory: sessionmaker[Session],
    monkeypatch,
    module: ModuleType,
    *,
    holder: Callable[[Session], Any],
    waiter: Callable[[Session], Any],
) -> tuple[tuple[str, Any], tuple[str, Any]]:
    """Pause ``holder`` at its commit until ``waiter`` finished or blocked behind it."""
    at_commit = threading.Event()
    release = threading.Event()
    waiter_finished = threading.Event()
    pids: dict[str, int] = {}
    original_commit = module._commit

    def paused_commit(db: Session, message: str) -> None:
        if db.info.pop("pause_at_commit", False):
            at_commit.set()
            assert release.wait(timeout=_LOCK_TIMEOUT_SECONDS)
        original_commit(db, message)

    monkeypatch.setattr(module, "_commit", paused_commit)

    def run_holder(db: Session) -> Any:
        db.info["pause_at_commit"] = True
        pids["holder"] = db.scalar(text("SELECT pg_backend_pid()"))
        return holder(db)

    def run_waiter(db: Session) -> Any:
        pids["waiter"] = db.scalar(text("SELECT pg_backend_pid()"))
        try:
            return waiter(db)
        finally:
            waiter_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        holder_future = executor.submit(_attempt, session_factory, run_holder)
        try:
            assert at_commit.wait(timeout=_LOCK_TIMEOUT_SECONDS), "The holder never committed"
            waiter_future = executor.submit(_attempt, session_factory, run_waiter)
            _wait_until_finished_or_blocked(session_factory, pids, waiter_finished)
        finally:
            release.set()
        return (
            holder_future.result(timeout=_THREAD_TIMEOUT_SECONDS),
            waiter_future.result(timeout=_THREAD_TIMEOUT_SECONDS),
        )


def _pilot_revisions(db: Session, scope: _Scope) -> tuple[int, int]:
    guideline = models.Guideline(
        project_id=scope.project_id,
        task_definition_id=scope.task_definition_id,
        name=f"Stale lock guideline {uuid4().hex[:8]}",
        created_by_user_id=scope.manager_id,
    )
    db.add(guideline)
    db.flush()
    pilot_round = _round(db, scope, assistance_policy="blind", status="closed")
    revision_ids = []
    for version_number in (1, 2):
        revision = models.GuidelineRevision(
            project_id=scope.project_id,
            guideline_id=guideline.id,
            version_number=version_number,
            task_version_id=scope.task_version_id,
            markdown=f"Revision {version_number}",
            content_hash=uuid4().hex,
            status="pilot",
            created_by_user_id=scope.manager_id,
        )
        db.add(revision)
        db.flush()
        db.add(
            models.GuidelineImpactEvaluation(
                project_id=scope.project_id,
                guideline_revision_id=revision.id,
                pilot_round_id=pilot_round.id,
                status="completed",
                passed=True,
                created_by_user_id=scope.manager_id,
            )
        )
        revision_ids.append(revision.id)
    db.commit()
    return revision_ids[0], revision_ids[1]


@pytest.mark.parametrize(
    ("waiter_revision", "waiter_status", "expected_statuses"),
    [
        (1, "active", ("retired", "active")),
        (0, "retired", ("retired", "pilot")),
    ],
    ids=["activate-two-revisions", "activate-and-retire-one-revision"],
)
def test_guideline_transitions_apply_in_commit_order(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
    waiter_revision: int,
    waiter_status: str,
    expected_statuses: tuple[str, str],
) -> None:
    with postgres_session_factory() as db:
        scope = _scope(db)
        revision_ids = _pilot_revisions(db, scope)

    def transition(index: int, status: str) -> Callable[[Session], str]:
        def run(db: Session) -> str:
            return guidelines.transition_guideline_revision(
                db,
                scope.project_id,
                revision_ids[index],
                schemas.GuidelineRevisionTransition(status=status),
                db.get(User, scope.manager_id),
            ).status

        return run

    holder, waiter = _commit_after_concurrent_attempt(
        postgres_session_factory,
        monkeypatch,
        guidelines,
        holder=transition(0, "active"),
        waiter=transition(waiter_revision, waiter_status),
    )

    assert holder == ("ok", "active")
    assert waiter == ("ok", waiter_status)
    with postgres_session_factory() as db:
        statuses = tuple(
            db.get(models.GuidelineRevision, revision_id).status for revision_id in revision_ids
        )
    assert statuses == expected_statuses


@pytest.mark.parametrize(
    ("initial", "winning", "stale", "expected_stale", "expected_cycle"),
    [
        (
            ("planned", "pool"),
            ("cancelled", "pool"),
            ("active", "pool"),
            ("ConflictError", "Cannot transition learning cycle from cancelled to active"),
            ("cancelled", "pool"),
        ),
        (
            ("active", "pool"),
            ("active", "select"),
            ("cancelled", "pool"),
            ("ValidationError", "Cancelling a cycle cannot change its current stage"),
            ("active", "select"),
        ),
    ],
    ids=["advance-after-cancel", "cancel-after-advance"],
)
def test_cycle_transition_rejects_a_state_changed_after_its_read(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
    initial: tuple[str, str],
    winning: tuple[str, str],
    stale: tuple[str, str],
    expected_stale: tuple[str, str],
    expected_cycle: tuple[str, str],
) -> None:
    with postgres_session_factory() as db:
        scope = _scope(db)
        cycle = models.LearningCycle(
            project_id=scope.project_id,
            name="Stale lock cycle",
            sequence=1,
            status=initial[0],
            current_stage=initial[1],
            task_version_id=scope.task_version_id,
            source_dataset_version_id=scope.dataset_version_id,
            created_by_user_id=scope.manager_id,
        )
        db.add(cycle)
        db.commit()
        cycle_id = cycle.id
    loaded, resume = _pause_after_load(monkeypatch, cycles, models.LearningCycle)

    def transition(target: tuple[str, str]) -> Callable[[Session], tuple[str, str]]:
        def run(db: Session) -> tuple[str, str]:
            updated = cycles.transition_learning_cycle(
                db,
                scope.project_id,
                cycle_id,
                schemas.CycleTransition(status=target[0], current_stage=target[1]),
            )
            return updated.status, updated.current_stage

        return run

    winner, stale_result = _commit_between_load_and_write(
        postgres_session_factory,
        stale=transition(stale),
        winner=transition(winning),
        loaded=loaded,
        resume=resume,
    )

    assert winner == ("ok", winning)
    assert stale_result == expected_stale
    with postgres_session_factory() as db:
        cycle = db.get(models.LearningCycle, cycle_id)
        assert (cycle.status, cycle.current_stage) == expected_cycle


def test_selection_set_publication_rejects_a_run_completed_after_its_read(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
) -> None:
    with postgres_session_factory() as db:
        scope = _scope(db)
        run = models.SelectionRun(
            project_id=scope.project_id,
            dataset_version_id=scope.dataset_version_id,
            task_version_id=scope.task_version_id,
            strategy="random",
            created_by_user_id=scope.manager_id,
        )
        db.add(run)
        db.commit()
        run_id = run.id
    loaded, resume = _pause_after_load(monkeypatch, selection, models.SelectionRun)

    def publish(db: Session) -> int:
        return selection.create_selection_set(
            db,
            schemas.SelectionSetCreate(
                project_id=scope.project_id,
                selection_run_id=run_id,
                items=[schemas.SelectionItem(dataset_item_id=scope.dataset_item_id, rank=1)],
            ),
            db.get(User, scope.manager_id),
        ).version_number

    winner, stale = _commit_between_load_and_write(
        postgres_session_factory,
        stale=publish,
        winner=publish,
        loaded=loaded,
        resume=resume,
    )

    assert winner == ("ok", 1)
    assert stale == ("ConflictError", "A completed selection run cannot be rewritten")
    with postgres_session_factory() as db:
        versions = [
            version
            for (version,) in db.query(models.SelectionSetVersion.version_number)
            .filter(models.SelectionSetVersion.selection_run_id == run_id)
            .all()
        ]
    assert versions == [1]


def _open_round_ready_to_close(
    db: Session, scope: _Scope, assistance_policy: str
) -> tuple[int, int, int]:
    annotation_round = _round(db, scope, assistance_policy=assistance_policy, status="open")
    item = models.RoundItem(
        project_id=scope.project_id,
        annotation_round_id=annotation_round.id,
        dataset_item_id=scope.dataset_item_id,
        selection_reason={"strategy": "all"},
        metadata_={},
    )
    db.add(item)
    db.commit()
    round_id, item_id = annotation_round.id, item.id
    left, right = (db.get(User, user_id) for user_id in scope.annotator_ids)
    covering = rounds.create_annotation_decision(
        db,
        schemas.AnnotationDecisionCreate(
            project_id=scope.project_id, round_item_id=item_id, output={"label": "left"}
        ),
        left,
    )
    rounds.create_round_submission(
        db,
        schemas.RoundSubmissionCreate(
            project_id=scope.project_id,
            annotation_round_id=round_id,
            decision_ids=[covering.id],
        ),
        left,
    )
    unsubmitted = rounds.create_annotation_decision(
        db,
        schemas.AnnotationDecisionCreate(
            project_id=scope.project_id, round_item_id=item_id, output={"label": "right"}
        ),
        right,
    )
    return round_id, item_id, unsubmitted.id


@pytest.mark.parametrize("assistance_policy", ["blind", "reveal_after_first_pass"])
@pytest.mark.parametrize(
    ("work", "expected_rejection"),
    [
        (
            "decision",
            "Annotation decisions require an open round; only adjudication may follow closure",
        ),
        (
            "submission",
            "Only open rounds accept annotation submissions; "
            "closed rounds accept adjudication submissions",
        ),
    ],
    ids=["decision", "submission"],
)
def test_round_close_rejects_annotation_work_that_read_the_round_open(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
    assistance_policy: str,
    work: str,
    expected_rejection: str,
) -> None:
    with postgres_session_factory() as db:
        scope = _scope(db)
        round_id, item_id, unsubmitted_id = _open_round_ready_to_close(
            db, scope, assistance_policy
        )
    loaded, resume = _pause_after_load(monkeypatch, rounds, models.AnnotationRound)

    def close(db: Session) -> str:
        return rounds.transition_annotation_round(db, scope.project_id, round_id, "closed").status

    def annotate(db: Session) -> int:
        actor = db.get(User, scope.annotator_ids[1])
        if work == "decision":
            return rounds.create_annotation_decision(
                db,
                schemas.AnnotationDecisionCreate(
                    project_id=scope.project_id, round_item_id=item_id, output={"label": "late"}
                ),
                actor,
            ).id
        return rounds.create_round_submission(
            db,
            schemas.RoundSubmissionCreate(
                project_id=scope.project_id,
                annotation_round_id=round_id,
                decision_ids=[unsubmitted_id],
            ),
            actor,
        ).id

    winner, stale = _commit_between_load_and_write(
        postgres_session_factory,
        stale=annotate,
        winner=close,
        loaded=loaded,
        resume=resume,
    )

    assert winner == ("ok", "closed")
    assert stale == ("ConflictError", expected_rejection)
    with postgres_session_factory() as db:
        late_decisions = (
            db.query(models.RoundAnnotationDecision.id)
            .filter(
                models.RoundAnnotationDecision.round_item_id == item_id,
                models.RoundAnnotationDecision.annotator_user_id == scope.annotator_ids[1],
            )
            .count()
        )
        late_submissions = (
            db.query(models.RoundSubmission.id)
            .filter(
                models.RoundSubmission.annotation_round_id == round_id,
                models.RoundSubmission.annotator_user_id == scope.annotator_ids[1],
            )
            .count()
        )
    assert (late_decisions, late_submissions) == (1, 0)
