import io
from urllib.parse import unquote

import pytest
from pydantic import ValidationError

from al_medlit.export.registry import export_formats
from al_medlit.export.schemas import ExportCreate
from al_medlit.lineage.models import ExportArtifact
from al_medlit.lineage.service import register_export_artifact


class _SnapshotExportFormat:
    key = "test-snapshot-export"
    content_type = "application/x-ndjson"
    extension = "jsonl"

    def iter_rows(self, context):
        yield {"project_id": context.project_id}


@pytest.fixture
def export_source(client, monkeypatch):
    monkeypatch.setattr(export_formats, "_formats", dict(export_formats._formats))
    export_formats.register(_SnapshotExportFormat(), replace=True)
    project = client.post("/api/projects", json={"name": "safe-export"}).json()
    document = client.post(
        "/api/documents", json={"project_id": project["id"], "text": "A sentence."}
    ).json()
    structure = client.get(f"/api/documents/{document['id']}/structure").json()["structure_version"]
    snapshot = client.post(
        f"/api/projects/{project['id']}/corpus-snapshots",
        json={
            "name": "export-source",
            "documents": [
                {
                    "document_id": document["id"],
                    "structure_version_id": structure["id"],
                    "source_hash": structure["source_hash"],
                    "group_key": "document",
                }
            ],
        },
    )
    assert snapshot.status_code == 200, snapshot.text
    return project["id"], {
        "format_key": _SnapshotExportFormat.key,
        "corpus_snapshot_id": snapshot.json()["id"],
    }


@pytest.mark.parametrize(
    "file_name",
    [
        "",
        " ",
        ".",
        "..",
        "../other.jsonl",
        "folder/report.jsonl",
        "/report.jsonl",
        "folder\\report.jsonl",
        "C:\\report.jsonl",
        "C:report.jsonl",
        "bad\x00.jsonl",
        "bad\r\nX-Injected: yes.jsonl",
        "bad\x7f.jsonl",
        "bad\x85.jsonl",
    ],
)
def test_export_rejects_names_that_are_not_leaf_filenames(file_name):
    with pytest.raises(ValidationError, match="file_name"):
        ExportCreate(format_key="evidence-blocks-v1", corpus_snapshot_id=1, file_name=file_name)


@pytest.mark.parametrize("file_name", ["报告.jsonl", "résumé.jsonl", 'a "quote".jsonl'])
def test_export_accepts_unicode_and_quoted_leaf_names(file_name):
    assert (
        ExportCreate(
            format_key="evidence-blocks-v1", corpus_snapshot_id=1, file_name=file_name
        ).file_name
        == file_name
    )


def test_export_path_traversal_cannot_overwrite_another_project(
    client, export_source, object_storage
):
    project_id, payload = export_source
    victim = client.post("/api/projects", json={"name": "victim-project"}).json()
    victim_key = f"projects/{victim['id']}/protected.jsonl"
    object_storage.put_bytes(victim_key, b"original project object")

    response = client.post(
        f"/api/projects/{project_id}/exports",
        json={**payload, "file_name": f"../../../../{victim_key}"},
    )

    assert object_storage.get_bytes(victim_key) == b"original project object"
    assert response.status_code == 422, response.text


def test_export_storage_key_does_not_depend_on_display_name(client, export_source, db):
    project_id, payload = export_source
    response = client.post(
        f"/api/projects/{project_id}/exports",
        json={**payload, "file_name": "review report.jsonl"},
    )
    assert response.status_code == 200, response.text
    export = db.get(ExportArtifact, response.json()["id"])
    assert export.file_name == "review report.jsonl"
    assert export.artifact.storage_key.startswith(f"projects/{project_id}/exports/")
    assert "review report.jsonl" not in export.artifact.storage_key


@pytest.mark.parametrize("file_name", ["报告.jsonl", "résumé.jsonl", 'a "quote".jsonl'])
def test_download_preserves_unicode_and_quotes_in_safe_header(
    client, export_source, file_name, monkeypatch
):
    project_id, payload = export_source
    response = client.post(
        f"/api/projects/{project_id}/exports", json={**payload, "file_name": file_name}
    )
    assert response.status_code == 200, response.text
    monkeypatch.setattr(client._client._transport, "raise_server_exceptions", False)
    downloaded = client.get(f"/api/exports/{response.json()['id']}/download")
    assert downloaded.status_code == 200, downloaded.text
    header = downloaded.headers["content-disposition"]
    header.encode("ascii")
    assert header.startswith('attachment; filename="')
    assert header.count('"') == 2
    assert unquote(header.split("filename*=UTF-8''", 1)[1]) == file_name
    assert downloaded.json() == {"project_id": project_id}


def test_download_safely_handles_legacy_path_and_control_characters(
    client, export_source, db, object_storage, monkeypatch
):
    project_id, payload = export_source
    stored = object_storage.put_stream("legacy/export.jsonl", io.BytesIO(b"{}\n"))
    export = register_export_artifact(
        db,
        project_id=project_id,
        stored=stored,
        format_key=payload["format_key"],
        file_name='../folder\\report"\r\nX-Injected: yes.jsonl',
        row_count=1,
        actor_user_id=None,
        corpus_snapshot_id=payload["corpus_snapshot_id"],
    )
    monkeypatch.setattr(client._client._transport, "raise_server_exceptions", False)
    downloaded = client.get(f"/api/exports/{export.id}/download")
    assert downloaded.status_code == 200, downloaded.text
    header = downloaded.headers["content-disposition"]
    header.encode("ascii")
    assert "\r" not in header and "\n" not in header
    assert "../" not in header and "\\" not in header
    assert 'filename="report_' in header
    assert unquote(header.split("filename*=UTF-8''", 1)[1]) == 'report"__X-Injected: yes.jsonl'
