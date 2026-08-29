"""Pytest configuration and shared fixtures for AL-MedLit tests.

Production runs on PostgreSQL. These fixtures use SQLite in-memory only for
*unit* tests where speed and isolation matter and no Postgres-specific
behaviour is exercised. Integration tests that depend on JSONB operators,
advisory locks, or full-text search must spin up a real Postgres (e.g. via
testcontainers) rather than reuse this fixture.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from al_medlit.co_learning.error_guideline_learning.events import (
    reset_event_session_factory,
    set_event_session_factory,
)
from al_medlit.core.database import Base, get_db, register_models
from al_medlit.main import create_app

TEST_DATABASE_URL = "sqlite://"


@pytest.fixture(autouse=True)
def fixture_safe_jwt_secret(monkeypatch):
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "jwt_secret", "test-jwt-secret-for-pytest-32-byte-minimum")


class LegacyAuthClient:
    """Add auth to legacy tests that predate token-derived annotator identity."""

    def __init__(self, client: TestClient, session_factory):
        self._client = client
        self._session_factory = session_factory
        self._tokens: dict[str, str] = {}

    def __getattr__(self, name):
        return getattr(self._client, name)

    @property
    def headers(self):
        return self._client.headers

    def _token_for(self, username: str) -> str:
        username = username.strip() or "tester"
        cached = self._tokens.get(username)
        if cached is not None:
            return cached

        from al_medlit.auth.models import User
        from al_medlit.auth.security import create_access_token, hash_password
        from al_medlit.workspace import service as workspace_service

        session = self._session_factory()
        try:
            user = session.query(User).filter(User.username == username).first()
            if user is None:
                user = User(
                    username=username,
                    password_hash=hash_password("pw"),
                    display_name=username,
                    is_active=True,
                )
                session.add(user)
                session.flush()
            elif not user.is_active:
                user.is_active = True
                session.flush()
            default_workspace = workspace_service.ensure_default_workspace(session)
            member = workspace_service.get_member(
                session,
                default_workspace.id,
                user.id,
            )
            if member is None:
                workspace_service.add_member(
                    session,
                    default_workspace.id,
                    user.id,
                    role="admin",
                )
            elif member.role != "admin":
                member.role = "admin"
                session.flush()
            session.commit()
            token = create_access_token(str(user.id))
            self._tokens[username] = token
            return token
        finally:
            session.close()

    @staticmethod
    def _has_explicit_auth(headers) -> bool:
        return bool(headers) and any(key.lower() == "authorization" for key in headers)

    @staticmethod
    def _legacy_public_path(method: str, path: str) -> bool:
        clean_path = path.split("?", 1)[0]
        method = method.upper()
        if (method, clean_path) in {
            ("POST", "/api/auth/login"),
            ("POST", "/api/auth/register"),
            ("GET", "/health"),
        }:
            return True
        return (
            method == "POST"
            and clean_path.startswith("/api/invites/")
            and clean_path.endswith("/accept")
        )

    @staticmethod
    def _username_for_legacy_auth(method: str, path: str, json_payload) -> str:
        if not isinstance(json_payload, dict):
            return "tester"
        clean_path = path.split("?", 1)[0]
        if method.upper() == "POST" and clean_path == "/api/annotations":
            annotator_id = json_payload.get("annotator_id")
            return annotator_id if isinstance(annotator_id, str) else "tester"
        if clean_path.startswith("/api/projects/") and clean_path.endswith("/submissions"):
            annotator_id = json_payload.get("annotator_id")
            return annotator_id if isinstance(annotator_id, str) else "tester"
        if clean_path.startswith("/api/projects/") and clean_path.endswith("/assignments"):
            assigned_by = json_payload.get("assigned_by")
            return assigned_by if isinstance(assigned_by, str) else "manager"
        return "tester"

    def _ensure_assignment_assignee(self, method: str, path: str, json_payload) -> None:
        if method.upper() != "POST" or not isinstance(json_payload, dict):
            return
        clean_path = path.split("?", 1)[0]
        parts = clean_path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "projects"] or parts[3] != "assignments":
            return
        annotator_id = json_payload.get("annotator_id")
        if not isinstance(annotator_id, str) or not annotator_id.strip():
            return
        try:
            project_id = int(parts[2])
        except ValueError:
            return

        from al_medlit.auth.models import User
        from al_medlit.auth.security import hash_password
        from al_medlit.project.models import Project
        from al_medlit.workspace import service as workspace_service

        session = self._session_factory()
        try:
            project = session.get(Project, project_id)
            if project is None or project.workspace_id is None:
                return
            username = annotator_id.strip()
            user = session.query(User).filter(User.username == username).first()
            if user is None:
                user = User(
                    username=username,
                    password_hash=hash_password("pw"),
                    display_name=username,
                    is_active=True,
                )
                session.add(user)
                session.flush()
            if workspace_service.get_member(session, project.workspace_id, user.id) is None:
                workspace_service.add_member(
                    session,
                    project.workspace_id,
                    user.id,
                    role="annotator",
                )
            session.commit()
        finally:
            session.close()

    def request(self, method: str, url, **kwargs):
        path = str(url)
        headers = kwargs.get("headers")
        self._ensure_assignment_assignee(method, path, kwargs.get("json"))
        if (
            "authorization" not in self._client.headers
            and not self._has_explicit_auth(headers)
            and not self._legacy_public_path(method, path)
        ):
            username = self._username_for_legacy_auth(method, path, kwargs.get("json"))
            merged_headers = dict(headers or {})
            merged_headers["Authorization"] = f"Bearer {self._token_for(username)}"
            kwargs["headers"] = merged_headers
        return self._client.request(method, url, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def patch(self, url, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def put(self, url, **kwargs):
        return self.request("PUT", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)


@pytest.fixture(name="test_engine")
def fixture_test_engine():
    register_models()
    db_engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(db_engine, "connect")
    def _enable_sqlite_fks(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=db_engine)
    try:
        yield db_engine
    finally:
        # The production schema contains intentionally cyclic foreign keys
        # (for example, a feedback run points to its immutable output set and
        # that set points back to its run). SQLite cannot ALTER/drop one side
        # of a cycle, so disable enforcement only for test-schema teardown.
        with db_engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            Base.metadata.drop_all(bind=connection)
        db_engine.dispose()


@pytest.fixture(name="testing_session_factory")
def fixture_testing_session_factory(request):
    test_engine = request.getfixturevalue("test_engine")
    return sessionmaker(bind=test_engine, autoflush=False, autocommit=False)


@pytest.fixture(autouse=True)
def fixture_event_session_factory(request):
    testing_session_factory = request.getfixturevalue("testing_session_factory")
    set_event_session_factory(testing_session_factory)
    try:
        yield
    finally:
        reset_event_session_factory()


@pytest.fixture(name="db")
def fixture_db(request):
    """Return a database session that rolls back after each test."""
    testing_session_factory = request.getfixturevalue("testing_session_factory")
    session = testing_session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(name="object_storage")
def fixture_object_storage(tmp_path):
    from al_medlit.core.storage import LocalObjectStorage

    return LocalObjectStorage(tmp_path / "objects")


@pytest.fixture(name="client")
def fixture_client(request):
    """Return a FastAPI TestClient backed by the in-memory test database."""
    from al_medlit.core.storage import get_object_storage

    testing_session_factory = request.getfixturevalue("testing_session_factory")
    object_storage = request.getfixturevalue("object_storage")
    app = create_app()

    def override_get_db():
        session = testing_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_object_storage] = lambda: object_storage
    with TestClient(app, base_url="https://testserver") as c:
        yield LegacyAuthClient(c, testing_session_factory)
    app.dependency_overrides.clear()


@pytest.fixture(name="auth_user")
def fixture_auth_user(db):
    from al_medlit.auth import service as auth_service
    from al_medlit.auth.schemas import UserCreate
    from al_medlit.auth.security import create_access_token
    from al_medlit.workspace import service as workspace_service

    user = auth_service.register_user(
        db,
        UserCreate(username="tester", password="pw", display_name="Test User"),
    )
    ws = workspace_service.create_personal_workspace(db, user)
    db.commit()
    return {
        "id": user.id,
        "username": user.username,
        "workspace_id": ws.id,
        "access_token": create_access_token(user.id),
    }


@pytest.fixture(name="auth_client")
def fixture_auth_client(client, auth_user):
    client.headers.update(
        {"Authorization": f"Bearer {auth_user['access_token']}"},
    )
    return client
