"""Keep the API's code-point offsets distinct from browser UTF-16 offsets."""

import pytest

DOCUMENT_TEXT = "𝑃 = 𝑛 😀. Aspirin reduced pain. Symptoms resolved. Follow-up continued."
SENTENCES = [[0, 8], [9, 30], [31, 49], [50, 70]]


@pytest.fixture
def unicode_document(client):
    project_response = client.post(
        "/api/projects",
        json={"name": "unicode-offsets", "annotation_schema": {}, "settings": {}},
    )
    assert project_response.status_code == 200
    project = project_response.json()
    document_response = client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "text": DOCUMENT_TEXT,
            "source": "manual",
            "metadata_": {},
        },
    )
    assert document_response.status_code == 200
    return document_response.json()


def test_document_and_workbench_sentence_offsets_are_code_points(client, unicode_document):
    assert unicode_document["sentences"] == SENTENCES

    workbench_response = client.get(f"/api/annotation-workbench/documents/{unicode_document['id']}")
    assert workbench_response.status_code == 200
    assert workbench_response.json()["document"]["sentences"] == SENTENCES

    structure_response = client.get(f"/api/documents/{unicode_document['id']}/structure")
    assert structure_response.status_code == 200
    sentences = structure_response.json()["sentences"]
    assert [[item["start_offset"], item["end_offset"]] for item in sentences] == SENTENCES
    assert [item["text"] for item in sentences] == [
        "𝑃 = 𝑛 😀.",
        "Aspirin reduced pain.",
        "Symptoms resolved.",
        "Follow-up continued.",
    ]


@pytest.mark.parametrize(
    ("annotation_type", "initial_span", "updated_span"),
    [
        ("entity", (9, 16, "Aspirin"), (25, 29, "pain")),
        (
            "sentence_label",
            (9, 30, "Aspirin reduced pain."),
            (31, 49, "Symptoms resolved."),
        ),
        (
            "passage_label",
            (9, 30, "Aspirin reduced pain."),
            (31, 49, "Symptoms resolved."),
        ),
    ],
)
def test_annotation_create_and_update_require_code_point_offsets(
    client, unicode_document, annotation_type, initial_span, updated_span
):
    start, end, text_span = initial_span
    payload = {
        "project_id": unicode_document["project_id"],
        "document_id": unicode_document["id"],
        "annotation_type": annotation_type,
        "label": "Relevant",
        "start_offset": start,
        "end_offset": end,
        "text_span": text_span,
    }
    # The math italic P/n and emoji each occupy two UTF-16 code units but one
    # Unicode code point, so every subsequent browser offset is three larger.
    legacy_response = client.post(
        "/api/annotations",
        json={**payload, "start_offset": start + 3, "end_offset": end + 3},
    )
    assert legacy_response.status_code == 422
    assert legacy_response.json()["detail"] == (
        "text_span does not match document text at the given offsets"
    )

    response = client.post("/api/annotations", json=payload)
    assert response.status_code == 200
    annotation = response.json()
    assert (
        annotation["start_offset"],
        annotation["end_offset"],
        annotation["text_span"],
    ) == initial_span

    start, end, text_span = updated_span
    annotation_url = f"/api/annotations/{annotation['id']}"
    legacy_update = client.patch(
        annotation_url,
        json={"start_offset": start + 3, "end_offset": end + 3, "text_span": text_span},
    )
    assert legacy_update.status_code == 422
    assert legacy_update.json()["detail"] == (
        "text_span does not match document text at the given offsets"
    )

    response = client.patch(
        annotation_url,
        json={"start_offset": start, "end_offset": end, "text_span": text_span},
    )
    assert response.status_code == 200
    updated = response.json()
    assert (updated["start_offset"], updated["end_offset"], updated["text_span"]) == updated_span

    workbench_response = client.get(f"/api/annotation-workbench/documents/{unicode_document['id']}")
    assert workbench_response.status_code == 200
    saved = workbench_response.json()["annotations"]
    assert len(saved) == 1
    assert (saved[0]["start_offset"], saved[0]["end_offset"], saved[0]["text_span"]) == updated_span
