import pytest
from test_workflow_ingestion_selection import _create_dataset, _create_project


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("registry_dataset_id", "../invalid/dataset"),
        ("exact_revision", "z" * 40),
        ("config_name", "invalid config"),
        ("expected_content_sha256", "z" * 64),
    ],
)
def test_public_registry_form_validation_is_422_and_can_be_retried(client, field, invalid):
    project = _create_project(client, "Registry validation")
    dataset = _create_dataset(client, project["id"], "Public snapshot", "public_registry")
    path = (
        f"/api/projects/{project['id']}/datasets/{dataset['id']}"
        "/versions/public-registry-snapshot"
    )
    valid = {
        "registry_dataset_id": "owner/dataset",
        "exact_revision": "a" * 40,
        "source_format": "jsonl",
    }
    files = {"file": ("snapshot.jsonl", b'{"text":"A paper"}\n', "application/x-ndjson")}
    response = client.post(path, data={**valid, field: invalid}, files=files)
    assert response.status_code == 422, response.text
    assert any(error["loc"] == ["body", field] for error in response.json()["detail"])
    versions = client.get(
        f"/api/datasets/versions?project_id={project['id']}&dataset_id={dataset['id']}"
    )
    assert versions.status_code == 200
    assert versions.json() == []
    retry = client.post(path, data=valid, files=files)
    assert retry.status_code == 201, retry.text
    assert retry.json()["item_count"] == 1
