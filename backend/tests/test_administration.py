from datetime import UTC, datetime, timedelta

import pytest


def _make_user(
    db,
    username: str,
    *,
    password: str = "a-secure-password",
    is_active: bool = True,
    is_superuser: bool = False,
):
    from al_medlit.auth.models import User
    from al_medlit.auth.security import hash_password

    user = User(
        username=username,
        password_hash=hash_password(password),
        display_name=username.title(),
        email=f"{username}@example.test",
        is_active=is_active,
        is_superuser=is_superuser,
    )
    db.add(user)
    db.flush()
    return user


def _headers(user) -> dict[str, str]:
    from al_medlit.auth.security import create_access_token

    return {
        "Authorization": (
            "Bearer "
            + create_access_token(user.id, session_version=user.session_version)
        )
    }


def _create_admin(db, username: str = "system-admin"):
    admin = _make_user(db, username, is_superuser=True)
    db.commit()
    return admin


def _token_from_action(action: dict) -> str:
    return action["url"].rsplit("/", 1)[-1]


def test_membershipless_superuser_can_list_users_but_workspace_admin_cannot(client, db):
    from al_medlit.workspace import service as workspace_service

    admin = _create_admin(db)
    workspace_admin = _make_user(db, "workspace-admin")
    workspace_service.create_team_workspace(db, workspace_admin, "Managed Team")
    db.commit()

    response = client.get("/api/admin/users", headers=_headers(admin))

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["page"] == 1
    assert payload["page_size"] == 25
    assert {item["username"] for item in payload["items"]} == {
        "system-admin",
        "workspace-admin",
    }

    forbidden = client.get("/api/admin/users", headers=_headers(workspace_admin))
    assert forbidden.status_code == 403


def test_administration_locks_superusers_and_target_in_one_canonical_user_query(
    db,
    monkeypatch,
):
    from sqlalchemy.orm import Query

    from al_medlit.administration import service
    from al_medlit.auth.models import User

    target = _make_user(db, "lower-id-target")
    admin = _make_user(db, "higher-id-admin", is_superuser=True)
    db.commit()
    assert target.id < admin.id
    user_lock_queries: list[str] = []
    real_with_for_update = Query.with_for_update

    def track_with_for_update(query, *args, **kwargs):
        locked_query = real_with_for_update(query, *args, **kwargs)
        entity = query.column_descriptions[0].get("entity")
        if entity is User:
            user_lock_queries.append(str(locked_query.statement))
        return locked_query

    monkeypatch.setattr(Query, "with_for_update", track_with_for_update)
    _actor, active_superusers, locked_users = (
        service._lock_and_require_active_superuser(
            db,
            admin.id,
            target_user_ids=(target.id,),
        )
    )

    assert [user.id for user in active_superusers] == [admin.id]
    assert list(locked_users) == [target.id, admin.id]
    assert len(user_lock_queries) == 1
    assert "ORDER BY users.id" in user_lock_queries[0]


def test_password_reset_link_reuses_canonically_locked_target(db, monkeypatch):
    from sqlalchemy.orm import Query

    from al_medlit.administration import service
    from al_medlit.auth.models import User

    target = _make_user(db, "reset-lock-target")
    admin = _make_user(db, "reset-lock-admin", is_superuser=True)
    db.commit()
    user_lock_count = 0
    real_with_for_update = Query.with_for_update

    def track_with_for_update(query, *args, **kwargs):
        nonlocal user_lock_count
        if query.column_descriptions[0].get("entity") is User:
            user_lock_count += 1
        return real_with_for_update(query, *args, **kwargs)

    monkeypatch.setattr(Query, "with_for_update", track_with_for_update)
    service.issue_password_reset_link(
        db,
        actor_user_id=admin.id,
        user_id=target.id,
    )

    assert user_lock_count == 1


def test_user_list_filters_and_detail_include_all_memberships(client, db):
    from al_medlit.workspace import service as workspace_service

    admin = _create_admin(db)
    member = _make_user(db, "study-member")
    first = workspace_service.create_team_workspace(db, admin, "Alpha Team")
    second = workspace_service.create_team_workspace(db, admin, "Beta Team")
    workspace_service.add_member(db, first.id, member.id, role="annotator")
    workspace_service.add_member(db, second.id, member.id, role="trainer")
    db.commit()

    response = client.get(
        f"/api/admin/users?search=study&workspace_id={first.id}&is_active=true",
        headers=_headers(admin),
    )
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["membership_count"] == 2

    detail = client.get(f"/api/admin/users/{member.id}", headers=_headers(admin))
    assert detail.status_code == 200
    assert detail.json()["membership_count"] == 2
    assert {
        (membership["workspace_name"], membership["workspace_kind"], membership["role"])
        for membership in detail.json()["memberships"]
    } == {
        ("Alpha Team", "team", "annotator"),
        ("Beta Team", "team", "trainer"),
    }


def test_instance_policy_is_safe_and_immediately_controls_registration(
    client,
    db,
    monkeypatch,
):
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "allow_self_registration", True)
    admin = _create_admin(db)
    response = client.get("/api/admin/settings", headers=_headers(admin))

    assert response.status_code == 200
    assert response.json()["allow_self_registration"] is True
    assert response.json()["default_invite_expiry_minutes"] == 10_080
    assert response.json()["account_action_expiry_minutes"] == 60
    assert set(response.json()) == {
        "allow_self_registration",
        "default_invite_expiry_minutes",
        "account_action_expiry_minutes",
        "deployment_profile",
        "storage_backend",
        "storage_encryption",
        "task_execution",
        "jwt_lifetime_minutes",
    }

    updated = client.patch(
        "/api/admin/settings",
        headers=_headers(admin),
        json={
            "allow_self_registration": False,
            "default_invite_expiry_minutes": 1_440,
            "account_action_expiry_minutes": 30,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["allow_self_registration"] is False

    registration = client.post(
        "/api/auth/register",
        json={"username": "blocked-signup", "password": "a-secure-password"},
    )
    assert registration.status_code == 403


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_invite_expiry_minutes", 59),
        ("default_invite_expiry_minutes", 43_201),
        ("account_action_expiry_minutes", 14),
        ("account_action_expiry_minutes", 1_441),
    ],
)
def test_instance_policy_rejects_expiry_outside_safe_ranges(client, db, field, value):
    admin = _create_admin(db)
    response = client.patch(
        "/api/admin/settings",
        headers=_headers(admin),
        json={field: value},
    )
    assert response.status_code == 422


def test_admin_created_account_uses_hashed_single_use_activation_link(client, db):
    from al_medlit.administration.models import AccountActionToken
    from al_medlit.auth import service as auth_service

    admin = _create_admin(db)
    created = client.post(
        "/api/admin/users",
        headers=_headers(admin),
        json={
            "username": "new-user",
            "display_name": "New User",
            "email": "new-user@example.test",
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["user"]["is_active"] is False
    assert body["user"]["is_initialized"] is False
    assert body["action"]["purpose"] == "activation"
    raw_token = _token_from_action(body["action"])

    db.expire_all()
    stored = db.query(AccountActionToken).one()
    assert stored.token_hash != raw_token
    assert len(stored.token_hash) == 64

    preview = client.get(
        f"/api/account-actions/{raw_token}",
        headers={"Authorization": ""},
    )
    assert preview.status_code == 200
    assert preview.json()["username"] == "new-user"

    completed = client.post(
        f"/api/account-actions/{raw_token}",
        headers={"Authorization": ""},
        json={"password": "new-secure-password"},
    )
    assert completed.status_code == 200
    assert completed.json()["completed"] is True
    assert completed.json()["purpose"] == "activation"
    assert completed.json()["token_type"] == "bearer"
    assert client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {completed.json()['access_token']}"},
    ).status_code == 200
    assert client.get(
        f"/api/account-actions/{raw_token}",
        headers={"Authorization": ""},
    ).status_code == 404

    db.expire_all()
    activated = auth_service.get_user_by_username(db, "new-user")
    assert activated is not None
    assert activated.is_active is True
    assert auth_service.authenticate_user(db, "new-user", "new-secure-password") is not None

    detail = client.get(
        f"/api/admin/users/{activated.id}",
        headers=_headers(admin),
    )
    assert detail.json()["is_initialized"] is True


def test_replacing_activation_link_revokes_previous_link(client, db):
    admin = _create_admin(db)
    created = client.post(
        "/api/admin/users",
        headers=_headers(admin),
        json={"username": "replace-link"},
    ).json()
    old_token = _token_from_action(created["action"])
    user_id = created["user"]["id"]

    replacement = client.post(
        f"/api/admin/users/{user_id}/activation-link",
        headers=_headers(admin),
    )
    assert replacement.status_code == 200
    new_token = _token_from_action(replacement.json())
    assert new_token != old_token
    assert client.get(
        f"/api/account-actions/{old_token}",
        headers={"Authorization": ""},
    ).status_code == 404
    assert client.get(
        f"/api/account-actions/{new_token}",
        headers={"Authorization": ""},
    ).status_code == 200


def test_expired_account_action_is_rejected(client, db):
    from al_medlit.administration.models import AccountActionToken

    admin = _create_admin(db)
    created = client.post(
        "/api/admin/users",
        headers=_headers(admin),
        json={"username": "expired-link"},
    ).json()
    raw_token = _token_from_action(created["action"])
    db.expire_all()
    action = db.query(AccountActionToken).one()
    action.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()

    assert client.get(
        f"/api/account-actions/{raw_token}",
        headers={"Authorization": ""},
    ).status_code == 404


def test_account_action_completion_revalidates_token_after_user_lock(db, monkeypatch):
    from al_medlit.administration import service
    from al_medlit.administration.schemas import AdminUserCreate

    admin = _create_admin(db)
    created = service.create_inactive_user(
        db,
        actor_user_id=admin.id,
        data=AdminUserCreate(username="lock-order-target"),
    )
    db.commit()
    token = _token_from_action(created.action.model_dump(mode="json"))
    validation_modes: list[bool] = []
    real_validate = service._valid_account_action

    def track_validation(session, raw_token, *, for_update):
        validation_modes.append(for_update)
        return real_validate(session, raw_token, for_update=for_update)

    monkeypatch.setattr(service, "_valid_account_action", track_validation)
    service.complete_account_action(
        db,
        token=token,
        password="lock-order-password",
    )
    db.commit()

    assert validation_modes == [False, True]


def test_password_reset_invalidates_existing_sessions_and_updates_login_time(client, db):
    from al_medlit.auth.models import User

    admin = _create_admin(db)
    target = _make_user(db, "reset-target", password="old-secure-password")
    db.commit()
    old_headers = _headers(target)

    issued = client.post(
        f"/api/admin/users/{target.id}/password-reset-link",
        headers=_headers(admin),
    )
    assert issued.status_code == 200
    raw_token = _token_from_action(issued.json())
    completed = client.post(
        f"/api/account-actions/{raw_token}",
        headers={"Authorization": ""},
        json={"password": "replacement-password"},
    )
    assert completed.status_code == 200

    assert client.get("/api/auth/me", headers=old_headers).status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"username": "reset-target", "password": "old-secure-password"},
    ).status_code == 401
    login = client.post(
        "/api/auth/login",
        json={"username": "reset-target", "password": "replacement-password"},
    )
    assert login.status_code == 200

    db.expire_all()
    refreshed = db.query(User).filter(User.username == "reset-target").one()
    assert refreshed.session_version == 1
    assert refreshed.last_login_at is not None


def test_legacy_session_token_is_valid_only_while_user_version_is_zero(client, db):
    import jwt

    from al_medlit.auth.security import create_access_token
    from al_medlit.core.config import settings

    user = _make_user(db, "legacy-session")
    db.commit()
    modern_token = create_access_token(user.id)
    claims = jwt.decode(
        modern_token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )
    claims.pop("sv")
    legacy_token = jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    headers = {"Authorization": f"Bearer {legacy_token}"}

    assert client.get("/api/auth/me", headers=headers).status_code == 200
    user.session_version = 1
    db.commit()
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_deactivation_preserves_membership_and_history_but_withdraws_mutable_work(
    client,
    db,
):
    from al_medlit.administration.models import AccountActionToken, AdminAuditEvent
    from al_medlit.corpus.models import Document
    from al_medlit.project.models import Project, ProjectTask, TaskAssignment
    from al_medlit.workflow.models import (
        AnnotationRound,
        Dataset,
        DatasetVersion,
        FeedbackEvent,
        ReviewCase,
        TaskDefinition,
        TaskVersion,
    )
    from al_medlit.workspace import service as workspace_service
    from al_medlit.workspace.models import WorkspaceMember

    admin = _create_admin(db)
    target = _make_user(db, "offboard-target")
    workspace = workspace_service.create_team_workspace(db, admin, "Offboarding Team")
    workspace_service.add_member(db, workspace.id, target.id, role="annotator")
    project = Project(name="Offboarding Project", workspace_id=workspace.id)
    db.add(project)
    db.flush()
    task = ProjectTask(
        project_id=project.id,
        annotation_type="entity",
        display_name="Entities",
    )
    mutable_document = Document(project_id=project.id, title="Mutable", text="mutable")
    completed_document = Document(project_id=project.id, title="Done", text="done")
    db.add_all([task, mutable_document, completed_document])
    db.flush()
    mutable = TaskAssignment(
        project_id=project.id,
        task_id=task.id,
        document_id=mutable_document.id,
        assignee_user_id=target.id,
        annotator_id=target.username,
        status="in_progress",
        metadata_={"existing": "value"},
    )
    completed = TaskAssignment(
        project_id=project.id,
        task_id=task.id,
        document_id=completed_document.id,
        assignee_user_id=target.id,
        annotator_id=target.username,
        status="completed",
    )
    db.add_all([mutable, completed])
    feedback_event = FeedbackEvent(
        project_id=project.id,
        event_type="manual_review",
        payload={},
    )
    db.add(feedback_event)
    db.flush()
    open_review_case = ReviewCase(
        project_id=project.id,
        feedback_event_id=feedback_event.id,
        assigned_to_user_id=target.id,
        case_type="quality",
        status="open",
        resolution={"existing": "review-context"},
    )
    resolved_review_case = ReviewCase(
        project_id=project.id,
        feedback_event_id=feedback_event.id,
        assigned_to_user_id=target.id,
        case_type="quality",
        status="resolved",
        resolution={"outcome": "kept"},
        resolved_at=datetime.now(UTC),
    )
    db.add_all([open_review_case, resolved_review_case])
    other_annotator = _make_user(db, "remaining-round-annotator")
    workspace_service.add_member(
        db,
        workspace.id,
        other_annotator.id,
        role="annotator",
    )
    task_definition = TaskDefinition(
        project_id=project.id,
        key="offboarding-task",
        name="Offboarding Task",
    )
    dataset = Dataset(
        project_id=project.id,
        name="Offboarding Dataset",
        source_type="test",
    )
    db.add_all([task_definition, dataset])
    db.flush()
    task_version = TaskVersion(
        project_id=project.id,
        task_definition_id=task_definition.id,
        version_number=1,
        task_kind="classification",
        content_hash="offboarding-task-v1",
    )
    dataset_version = DatasetVersion(
        project_id=project.id,
        dataset_id=dataset.id,
        version_number=1,
        source_revision="test-v1",
        source_format="jsonl",
        content_hash="offboarding-dataset-v1",
        item_count=0,
    )
    db.add_all([task_version, dataset_version])
    db.flush()
    open_round = AnnotationRound(
        project_id=project.id,
        name="Open Round",
        sequence=1,
        dataset_version_id=dataset_version.id,
        task_version_id=task_version.id,
        assistance_policy="blind",
        reannotation_mode="full_dataset",
        annotator_user_ids=[target.id, other_annotator.id],
        status="open",
        opened_at=datetime.now(UTC),
    )
    closed_round = AnnotationRound(
        project_id=project.id,
        name="Closed Round",
        sequence=2,
        dataset_version_id=dataset_version.id,
        task_version_id=task_version.id,
        assistance_policy="blind",
        reannotation_mode="full_dataset",
        annotator_user_ids=[target.id],
        status="closed",
        opened_at=datetime.now(UTC),
        closed_at=datetime.now(UTC),
    )
    draft_round = AnnotationRound(
        project_id=project.id,
        name="Draft Round",
        sequence=3,
        dataset_version_id=dataset_version.id,
        task_version_id=task_version.id,
        assistance_policy="blind",
        reannotation_mode="full_dataset",
        annotator_user_ids=[target.id],
        status="draft",
    )
    db.add_all([open_round, closed_round, draft_round])
    db.commit()
    old_headers = _headers(target)

    reset = client.post(
        f"/api/admin/users/{target.id}/password-reset-link",
        headers=_headers(admin),
    )
    assert reset.status_code == 200
    response = client.patch(
        f"/api/admin/users/{target.id}/status",
        headers=_headers(admin),
        json={"is_active": False},
    )
    assert response.status_code == 200
    assert response.json()["is_active"] is False
    assert client.get("/api/auth/me", headers=old_headers).status_code == 401

    db.expire_all()
    assert db.get(WorkspaceMember, workspace.members[1].id) is not None
    withdrawn = db.get(TaskAssignment, mutable.id)
    assert withdrawn.status == "withdrawn"
    assert withdrawn.metadata_["existing"] == "value"
    assert withdrawn.metadata_["account_deactivation"]["reason"] == "user_deactivated"
    assert db.get(TaskAssignment, completed.id).status == "completed"
    unassigned_review = db.get(ReviewCase, open_review_case.id)
    assert unassigned_review.status == "open"
    assert unassigned_review.assigned_to_user_id is None
    assert unassigned_review.resolution["existing"] == "review-context"
    assert (
        unassigned_review.resolution["account_deactivation"]["reason"]
        == "user_deactivated"
    )
    preserved_review = db.get(ReviewCase, resolved_review_case.id)
    assert preserved_review.status == "resolved"
    assert preserved_review.assigned_to_user_id == target.id
    assert preserved_review.resolution == {"outcome": "kept"}
    assert db.get(AnnotationRound, open_round.id).annotator_user_ids == [
        other_annotator.id
    ]
    assert db.get(AnnotationRound, open_round.id).status == "open"
    assert db.get(AnnotationRound, draft_round.id).annotator_user_ids == []
    assert db.get(AnnotationRound, draft_round.id).status == "draft"
    assert db.get(AnnotationRound, closed_round.id).annotator_user_ids == [target.id]
    assert db.query(AccountActionToken).one().revoked_at is not None
    event = (
        db.query(AdminAuditEvent)
        .filter(AdminAuditEvent.event_type == "account.deactivated")
        .one()
    )
    assert event.details["withdrawn_assignment_count"] == 1
    assert event.details["withdrawn_annotation_round_count"] == 2
    assert event.details["withdrawn_annotation_round_ids"] == [
        open_round.id,
        draft_round.id,
    ]
    assert event.details["unassigned_review_case_count"] == 1
    round_events = (
        db.query(AdminAuditEvent)
        .filter(
            AdminAuditEvent.event_type
            == "workflow.annotation_round_annotator_withdrawn"
        )
        .order_by(AdminAuditEvent.id)
        .all()
    )
    assert len(round_events) == 2
    assert {round_event.details["reason"] for round_event in round_events} == {
        "user_deactivated"
    }
    assert {
        (round_event.details["annotation_round_id"], round_event.details["round_status"])
        for round_event in round_events
    } == {(open_round.id, "open"), (draft_round.id, "draft")}

    reactivated = client.patch(
        f"/api/admin/users/{target.id}/status",
        headers=_headers(admin),
        json={"is_active": True},
    )
    assert reactivated.status_code == 200
    db.expire_all()
    assert db.get(TaskAssignment, mutable.id).status == "withdrawn"
    assert db.get(AnnotationRound, open_round.id).annotator_user_ids == [
        other_annotator.id
    ]
    assert db.get(AnnotationRound, draft_round.id).annotator_user_ids == []


def test_superuser_cannot_deactivate_self(client, db):
    admin = _create_admin(db)
    response = client.patch(
        f"/api/admin/users/{admin.id}/status",
        headers=_headers(admin),
        json={"is_active": False},
    )
    assert response.status_code == 409


def test_audit_events_are_append_only(db):
    from al_medlit.administration.events import record_admin_event

    event = record_admin_event(db, event_type="test.created", details={"safe": True})
    db.commit()
    event.details = {"safe": False}
    with pytest.raises(ValueError, match="append-only"):
        db.flush()
