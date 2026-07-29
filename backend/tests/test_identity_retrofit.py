def _make_project_document(db, *, project_name: str, text: str = "hello world"):
    from al_medlit.auth.models import User
    from al_medlit.corpus.models import Document
    from al_medlit.project.models import Project
    from al_medlit.workspace import service as workspace_service
    from al_medlit.workspace.models import WorkspaceMember

    workspace_id = None
    tester = db.query(User).filter(User.username == "tester").first()
    if tester is not None:
        member = (
            db.query(WorkspaceMember)
            .filter(WorkspaceMember.user_id == tester.id)
            .order_by(WorkspaceMember.id)
            .first()
        )
        if member is not None:
            workspace_id = member.workspace_id
    if workspace_id is None:
        workspace_id = workspace_service.ensure_default_workspace(db).id

    project = Project(name=project_name, workspace_id=workspace_id)
    db.add(project)
    db.flush()
    document = Document(project_id=project.id, title="doc", text=text)
    db.add(document)
    db.commit()
    return project, document


def test_submission_uses_token_identity(auth_client, db):
    project, document = _make_project_document(db, project_name="RetrofitProj")

    resp = auth_client.post(
        f"/api/projects/{project.id}/documents/{document.id}/submissions",
        json={"kind": "submission", "annotator_id": "IMPERSONATED", "metadata_": {}},
    )
    assert resp.status_code == 200
    assert resp.json()["annotator_id"] == "tester"


def test_annotation_create_uses_token_identity(auth_client, db):
    project, document = _make_project_document(db, project_name="RetrofitAnnoProj")

    resp = auth_client.post(
        "/api/annotations",
        json={
            "project_id": project.id,
            "document_id": document.id,
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 0,
            "end_offset": 5,
            "text_span": "hello",
            "annotator_id": "IMPERSONATED",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["annotator_id"] == "tester"


def test_write_endpoints_require_auth(client, db):
    project, document = _make_project_document(db, project_name="NoAuthProj", text="hi")

    resp = client.post(
        f"/api/projects/{project.id}/documents/{document.id}/submissions",
        json={"kind": "submission", "metadata_": {}},
        headers={"Authorization": ""},
    )
    assert resp.status_code == 401
