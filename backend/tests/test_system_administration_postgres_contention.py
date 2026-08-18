"""PostgreSQL contention coverage for cross-scope administration locks.

These tests intentionally use a real PostgreSQL server because SQLite does not
implement ``SELECT ... FOR UPDATE``. They are opt-in to keep the default test
suite independent of Docker:

    AL_MEDLIT_RUN_POSTGRES_TESTS=1 pytest -q \
        tests/test_system_administration_postgres_contention.py
"""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from al_medlit.administration.models import AccountActionToken
from al_medlit.administration.service import (
    complete_account_action,
    issue_password_reset_link,
    set_user_status,
)
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import lock_project_member_for_mutation
from al_medlit.core.exceptions import ForbiddenError, NotFoundError
from al_medlit.project.models import Project
from al_medlit.workflow.models import (
    AnnotationRound,
    Dataset,
    DatasetItem,
    DatasetVersion,
    RoundItem,
    TaskDefinition,
    TaskVersion,
)
from al_medlit.workflow.services.rounds import transition_annotation_round
from al_medlit.workspace import service as workspace_service
from al_medlit.workspace.models import Workspace, WorkspaceInvite, WorkspaceMember

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]

_LOCK_TIMEOUT_SECONDS = 5
_THREAD_TIMEOUT_SECONDS = 15


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_postgres(url: str, timeout_seconds: int = 30) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        engine = create_engine(url, pool_pre_ping=True)
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return
        except SQLAlchemyError:
            time.sleep(0.5)
        finally:
            engine.dispose()
    raise AssertionError("Timed out waiting for PostgreSQL to accept connections")


@pytest.fixture(scope="module")
def postgres_session_factory() -> Iterator[sessionmaker[Session]]:
    port = _free_port()
    container_name = f"al-medlit-contention-{uuid4().hex[:8]}"
    postgres_url = f"postgresql+psycopg2://al_medlit:secret@127.0.0.1:{port}/al_medlit"
    docker_run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "-e",
            "POSTGRES_DB=al_medlit",
            "-e",
            "POSTGRES_USER=al_medlit",
            "-e",
            "POSTGRES_PASSWORD=secret",
            "-p",
            f"127.0.0.1:{port}:5432",
            "postgres:16-alpine",
        ],
        cwd=_backend_root(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert docker_run.returncode == 0, docker_run.stderr

    engine = None
    try:
        _wait_for_postgres(postgres_url)
        migration = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=_backend_root(),
            env=os.environ | {"AL_MEDLIT_DATABASE_URL": postgres_url},
            text=True,
            capture_output=True,
            check=False,
            timeout=90,
        )
        assert migration.returncode == 0, migration.stderr

        engine = create_engine(postgres_url, pool_pre_ping=True, pool_size=6, max_overflow=2)
        yield sessionmaker(bind=engine, autoflush=False, autocommit=False)
    finally:
        if engine is not None:
            engine.dispose()
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            cwd=_backend_root(),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )


def _new_user(
    db: Session,
    prefix: str,
    *,
    is_superuser: bool = False,
) -> User:
    suffix = uuid4().hex[:10]
    user = User(
        username=f"{prefix}-{suffix}",
        email=f"{prefix}-{suffix}@example.test",
        password_hash="usable-test-password-hash",
        display_name=prefix.replace("-", " ").title(),
        is_active=True,
        is_superuser=is_superuser,
    )
    db.add(user)
    db.flush()
    return user


def _set_transaction_timeouts(db: Session) -> None:
    db.execute(text(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
    db.execute(text(f"SET LOCAL statement_timeout = '{_THREAD_TIMEOUT_SECONDS - 2}s'"))


def _transaction_worker(
    session_factory: sessionmaker[Session],
    barrier: threading.Barrier,
    operation: Callable[[Session], Any],
) -> tuple[str, Any]:
    db = session_factory()
    try:
        _set_transaction_timeouts(db)
        barrier.wait(timeout=_LOCK_TIMEOUT_SECONDS)
        value = operation(db)
        db.commit()
        return ("ok", value)
    except Exception as exc:  # noqa: BLE001 - race outcomes are asserted by the caller
        db.rollback()
        return ("error", exc)
    finally:
        db.close()


def _run_concurrently(
    session_factory: sessionmaker[Session],
    left: Callable[[Session], Any],
    right: Callable[[Session], Any],
) -> tuple[tuple[str, Any], tuple[str, Any]]:
    barrier = threading.Barrier(2)
    results: list[tuple[str, Any] | None] = [None, None]

    def run(index: int, operation: Callable[[Session], Any]) -> None:
        results[index] = _transaction_worker(session_factory, barrier, operation)

    threads = [
        threading.Thread(target=run, args=(0, left), daemon=True),
        threading.Thread(target=run, args=(1, right), daemon=True),
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + _THREAD_TIMEOUT_SECONDS
    for thread in threads:
        thread.join(max(0, deadline - time.monotonic()))

    assert not any(thread.is_alive() for thread in threads), "Concurrent operation deadlocked"
    assert results[0] is not None and results[1] is not None
    return results[0], results[1]


def _raw_token(action_url: str) -> str:
    return action_url.rsplit("/", 1)[-1]


def _token_hash(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def test_invitee_deactivation_and_invite_acceptance_do_not_deadlock(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    with postgres_session_factory() as db:
        system_admin = _new_user(db, "system-admin", is_superuser=True)
        inviter = _new_user(db, "workspace-admin")
        invitee = _new_user(db, "invitee")
        workspace = workspace_service.create_team_workspace(db, inviter, "Contention team")
        invite = workspace_service.create_invite(
            db,
            workspace.id,
            created_by=inviter.id,
            role="annotator",
            expires_minutes=60,
        )
        db.commit()
        system_admin_id = system_admin.id
        invitee_id = invitee.id
        workspace_id = workspace.id
        invite_id = invite.id
        invite_token = invite.token

    def deactivate(db: Session) -> str:
        set_user_status(
            db,
            actor_user_id=system_admin_id,
            user_id=invitee_id,
            is_active=False,
        )
        return "deactivated"

    def accept(db: Session) -> str:
        current_invite = workspace_service.get_open_invite(db, invite_token)
        current_user = db.get(User, invitee_id)
        assert current_user is not None
        workspace_service.accept_invite(db, current_invite, current_user)
        return "accepted"

    deactivation_result, acceptance_result = _run_concurrently(
        postgres_session_factory,
        deactivate,
        accept,
    )

    assert deactivation_result == ("ok", "deactivated")
    assert acceptance_result[0] == "ok" or isinstance(acceptance_result[1], ForbiddenError)
    with postgres_session_factory() as db:
        invitee = db.get(User, invitee_id)
        invite = db.get(WorkspaceInvite, invite_id)
        membership = (
            db.query(WorkspaceMember)
            .filter_by(workspace_id=workspace_id, user_id=invitee_id)
            .first()
        )
        assert invitee is not None and invitee.is_active is False
        assert invite is not None
        if acceptance_result[0] == "ok":
            assert invite.accepted_by == invitee_id
            assert membership is not None
        else:
            assert invite.accepted_at is None
            assert membership is None


def test_inviter_deactivation_and_invite_acceptance_do_not_deadlock(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    with postgres_session_factory() as db:
        system_admin = _new_user(db, "authority-system-admin", is_superuser=True)
        inviter = _new_user(db, "authority-workspace-admin")
        invitee = _new_user(db, "authority-invitee")
        workspace = workspace_service.create_team_workspace(db, inviter, "Authority team")
        invite = workspace_service.create_invite(
            db,
            workspace.id,
            created_by=inviter.id,
            role="annotator",
            expires_minutes=60,
        )
        db.commit()
        system_admin_id = system_admin.id
        inviter_id = inviter.id
        invitee_id = invitee.id
        workspace_id = workspace.id
        invite_id = invite.id
        invite_token = invite.token

    def deactivate_inviter(db: Session) -> str:
        set_user_status(
            db,
            actor_user_id=system_admin_id,
            user_id=inviter_id,
            is_active=False,
        )
        return "deactivated"

    def accept(db: Session) -> str:
        current_invite = workspace_service.get_open_invite(db, invite_token)
        current_user = db.get(User, invitee_id)
        assert current_user is not None
        workspace_service.accept_invite(db, current_invite, current_user)
        return "accepted"

    deactivation_result, acceptance_result = _run_concurrently(
        postgres_session_factory,
        deactivate_inviter,
        accept,
    )

    assert deactivation_result == ("ok", "deactivated")
    assert acceptance_result[0] == "ok" or isinstance(acceptance_result[1], ForbiddenError)
    with postgres_session_factory() as db:
        inviter = db.get(User, inviter_id)
        invite = db.get(WorkspaceInvite, invite_id)
        membership = (
            db.query(WorkspaceMember)
            .filter_by(workspace_id=workspace_id, user_id=invitee_id)
            .first()
        )
        assert inviter is not None and inviter.is_active is False
        assert invite is not None
        if acceptance_result[0] == "ok":
            assert invite.accepted_by == invitee_id
            assert membership is not None
        else:
            assert invite.accepted_at is None
            assert membership is None


def test_global_deactivation_and_workspace_governance_do_not_deadlock(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    with postgres_session_factory() as db:
        system_admin = _new_user(db, "governance-system-admin", is_superuser=True)
        workspace_admin = _new_user(db, "governance-workspace-admin")
        workspace = workspace_service.create_team_workspace(
            db,
            workspace_admin,
            "Governance contention team",
        )
        db.commit()
        system_admin_id = system_admin.id
        workspace_admin_id = workspace_admin.id
        workspace_id = workspace.id
        original_join_code = workspace.join_code

    def deactivate(db: Session) -> str:
        set_user_status(
            db,
            actor_user_id=system_admin_id,
            user_id=workspace_admin_id,
            is_active=False,
        )
        return "deactivated"

    def rotate(db: Session) -> str:
        workspace = workspace_service.rotate_join_code(
            db,
            workspace_id,
            actor_user_id=workspace_admin_id,
        )
        return workspace.join_code or ""

    deactivation_result, rotation_result = _run_concurrently(
        postgres_session_factory,
        deactivate,
        rotate,
    )

    assert deactivation_result == ("ok", "deactivated")
    assert rotation_result[0] == "ok" or isinstance(rotation_result[1], ForbiddenError)
    with postgres_session_factory() as db:
        workspace_admin = db.get(User, workspace_admin_id)
        workspace = db.get(Workspace, workspace_id)
        assert workspace_admin is not None and workspace_admin.is_active is False
        assert workspace is not None
        if rotation_result[0] == "ok":
            assert rotation_result[1] != original_join_code
            assert workspace.join_code == rotation_result[1]
        else:
            assert workspace.join_code == original_join_code


def test_account_action_completion_and_replacement_do_not_deadlock(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    with postgres_session_factory() as db:
        system_admin = _new_user(db, "replacement-system-admin", is_superuser=True)
        target = _new_user(db, "replacement-target")
        action = issue_password_reset_link(
            db,
            actor_user_id=system_admin.id,
            user_id=target.id,
        )
        db.commit()
        system_admin_id = system_admin.id
        target_id = target.id
        original_token = _raw_token(action.url)

    def complete(db: Session) -> str:
        complete_account_action(
            db,
            token=original_token,
            password="Completed-password-123!",
        )
        return "completed"

    def replace(db: Session) -> str:
        replacement = issue_password_reset_link(
            db,
            actor_user_id=system_admin_id,
            user_id=target_id,
        )
        return _raw_token(replacement.url)

    completion_result, replacement_result = _run_concurrently(
        postgres_session_factory,
        complete,
        replace,
    )

    assert replacement_result[0] == "ok"
    assert completion_result[0] == "ok" or isinstance(completion_result[1], NotFoundError)
    replacement_token = replacement_result[1]
    with postgres_session_factory() as db:
        target = db.get(User, target_id)
        actions = (
            db.query(AccountActionToken)
            .filter(AccountActionToken.user_id == target_id)
            .order_by(AccountActionToken.id)
            .all()
        )
        current = [
            action
            for action in actions
            if action.consumed_at is None and action.revoked_at is None
        ]
        original = next(
            action for action in actions if action.token_hash == _token_hash(original_token)
        )
        assert target is not None and target.is_active is True
        assert len(current) == 1
        assert current[0].token_hash == _token_hash(replacement_token)
        if completion_result[0] == "ok":
            assert original.consumed_at is not None
            assert target.session_version == 1
        else:
            assert original.revoked_at is not None
            assert target.session_version == 0


def test_account_action_completion_and_deactivation_do_not_deadlock(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    with postgres_session_factory() as db:
        system_admin = _new_user(db, "deactivation-system-admin", is_superuser=True)
        target = _new_user(db, "deactivation-target")
        action = issue_password_reset_link(
            db,
            actor_user_id=system_admin.id,
            user_id=target.id,
        )
        db.commit()
        system_admin_id = system_admin.id
        target_id = target.id
        action_token = _raw_token(action.url)

    def complete(db: Session) -> str:
        complete_account_action(
            db,
            token=action_token,
            password="Completed-before-deactivation-123!",
        )
        return "completed"

    def deactivate(db: Session) -> str:
        set_user_status(
            db,
            actor_user_id=system_admin_id,
            user_id=target_id,
            is_active=False,
        )
        return "deactivated"

    completion_result, deactivation_result = _run_concurrently(
        postgres_session_factory,
        complete,
        deactivate,
    )

    assert deactivation_result == ("ok", "deactivated")
    assert completion_result[0] == "ok" or isinstance(completion_result[1], NotFoundError)
    with postgres_session_factory() as db:
        target = db.get(User, target_id)
        action = (
            db.query(AccountActionToken)
            .filter(AccountActionToken.token_hash == _token_hash(action_token))
            .one()
        )
        assert target is not None and target.is_active is False
        assert action.consumed_at is not None or action.revoked_at is not None
        expected_version = 2 if completion_result[0] == "ok" else 1
        assert target.session_version == expected_version


def test_lower_id_round_assignee_and_higher_id_superuser_deactivation_do_not_deadlock(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    with postgres_session_factory() as db:
        target = _new_user(db, "round-lower-target")
        system_admin = _new_user(db, "round-higher-admin", is_superuser=True)
        assert target.id < system_admin.id
        workspace = workspace_service.create_team_workspace(
            db,
            target,
            "Round lock contention team",
        )
        project = Project(
            name=f"Round contention {uuid4().hex[:8]}",
            workspace_id=workspace.id,
        )
        db.add(project)
        db.flush()
        task_definition = TaskDefinition(
            project_id=project.id,
            key="contention-task",
            name="Contention Task",
        )
        dataset = Dataset(
            project_id=project.id,
            name="Contention Dataset",
            source_type="test",
        )
        db.add_all([task_definition, dataset])
        db.flush()
        task_version = TaskVersion(
            project_id=project.id,
            task_definition_id=task_definition.id,
            version_number=1,
            task_kind="classification",
            content_hash=uuid4().hex,
        )
        dataset_version = DatasetVersion(
            project_id=project.id,
            dataset_id=dataset.id,
            version_number=1,
            source_revision="contention-v1",
            source_format="jsonl",
            content_hash=uuid4().hex,
            item_count=1,
        )
        db.add_all([task_version, dataset_version])
        db.flush()
        dataset_item = DatasetItem(
            project_id=project.id,
            dataset_version_id=dataset_version.id,
            stable_key="contention-item",
            payload={"text": "contention"},
            content_hash=uuid4().hex,
        )
        annotation_round = AnnotationRound(
            project_id=project.id,
            name="Contended draft",
            sequence=1,
            dataset_version_id=dataset_version.id,
            task_version_id=task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=[target.id, system_admin.id],
            status="draft",
            created_by_user_id=system_admin.id,
        )
        db.add_all([dataset_item, annotation_round])
        db.flush()
        db.add(
            RoundItem(
                project_id=project.id,
                annotation_round_id=annotation_round.id,
                dataset_item_id=dataset_item.id,
                selection_reason={"strategy": "all"},
                metadata_={},
            )
        )
        db.commit()
        system_admin_id = system_admin.id
        target_id = target.id
        project_id = project.id
        annotation_round_id = annotation_round.id

    def deactivate(db: Session) -> str:
        set_user_status(
            db,
            actor_user_id=system_admin_id,
            user_id=target_id,
            is_active=False,
        )
        return "deactivated"

    def open_round(db: Session) -> str:
        actor = db.get(User, system_admin_id)
        assert actor is not None
        candidate = (
            db.query(AnnotationRound.annotator_user_ids)
            .filter(AnnotationRound.id == annotation_round_id)
            .first()
        )
        related_user_ids = list(candidate[0] or []) if candidate is not None else []
        lock_project_member_for_mutation(
            db,
            actor,
            project_id,
            related_user_ids=related_user_ids,
        )
        opened = transition_annotation_round(
            db,
            project_id,
            annotation_round_id,
            "open",
        )
        return opened.status

    deactivation_result, open_result = _run_concurrently(
        postgres_session_factory,
        deactivate,
        open_round,
    )

    assert deactivation_result == ("ok", "deactivated")
    assert open_result == ("ok", "open")
    with postgres_session_factory() as db:
        target = db.get(User, target_id)
        annotation_round = db.get(AnnotationRound, annotation_round_id)
        assert target is not None and target.is_active is False
        assert annotation_round is not None and annotation_round.status == "open"
        assert annotation_round.annotator_user_ids == [system_admin_id]
