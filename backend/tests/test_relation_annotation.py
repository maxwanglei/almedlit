"""Relation annotation rules: span-only targets, duplicates, type constraints."""


def _create_project(client, name: str) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": name,
            "description": f"project {name}",
            "annotation_schema": {"labels": {}},
            "settings": {},
        },
    )
    assert response.status_code == 200
    return response.json()


def _create_document(client, project_id: int, text: str) -> dict:
    response = client.post(
        "/api/documents",
        json={
            "project_id": project_id,
            "external_id": "PMID:relation-tests",
            "title": "Relation test document",
            "text": text,
            "source": "manual",
            "metadata_": {},
        },
    )
    assert response.status_code == 200
    return response.json()


def _create_entity_annotation(
    client,
    project_id: int,
    document_id: int,
    *,
    label: str,
    start_offset: int,
    end_offset: int,
    text_span: str,
) -> dict:
    response = client.post(
        "/api/annotations",
        json={
            "project_id": project_id,
            "document_id": document_id,
            "annotation_type": "entity",
            "label": label,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "text_span": text_span,
        },
    )
    assert response.status_code == 200
    return response.json()


def _create_relation(
    client,
    project_id: int,
    document_id: int,
    *,
    label: str,
    head_id: int,
    tail_id: int,
):
    return client.post(
        "/api/annotations",
        json={
            "project_id": project_id,
            "document_id": document_id,
            "annotation_type": "relation",
            "label": label,
            "head_annotation_id": head_id,
            "tail_annotation_id": tail_id,
        },
    )


DOCUMENT_TEXT = "Ibuprofen caused gastrointestinal bleeding in elderly patients."
# Offsets: "Ibuprofen" = 0-9, "gastrointestinal bleeding" = 17-42, "patients" = 54-62

VALID_CONSTRAINTS = {
    "relation_constraints": {
        "causes": {"head": ["Drug"], "tail": ["AdverseEvent"]},
    }
}


def test_relation_targets_must_be_span_annotations(client):
    project = _create_project(client, "relation-span-targets")
    document = _create_document(client, project["id"], DOCUMENT_TEXT)
    head = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=0,
        end_offset=9,
        text_span="Ibuprofen",
    )

    doc_label_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "doc_label",
            "label": "CaseReport",
        },
    )
    assert doc_label_response.status_code == 200
    doc_label = doc_label_response.json()

    response = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=head["id"],
        tail_id=doc_label["id"],
    )
    assert response.status_code == 422
    assert "span annotations" in response.json()["detail"]


def test_relation_cannot_target_another_relation(client):
    project = _create_project(client, "relation-no-relation-targets")
    document = _create_document(client, project["id"], DOCUMENT_TEXT)
    head = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=0,
        end_offset=9,
        text_span="Ibuprofen",
    )
    tail = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="AdverseEvent",
        start_offset=17,
        end_offset=42,
        text_span="gastrointestinal bleeding",
    )
    relation_response = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=head["id"],
        tail_id=tail["id"],
    )
    assert relation_response.status_code == 200
    relation = relation_response.json()

    nested_response = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=relation["id"],
        tail_id=tail["id"],
    )
    assert nested_response.status_code == 422
    assert "span annotations" in nested_response.json()["detail"]


def test_duplicate_relations_are_rejected(client):
    project = _create_project(client, "relation-duplicates")
    document = _create_document(client, project["id"], DOCUMENT_TEXT)
    head = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=0,
        end_offset=9,
        text_span="Ibuprofen",
    )
    tail = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="AdverseEvent",
        start_offset=17,
        end_offset=42,
        text_span="gastrointestinal bleeding",
    )

    first = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=head["id"],
        tail_id=tail["id"],
    )
    assert first.status_code == 200

    duplicate = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=head["id"],
        tail_id=tail["id"],
    )
    assert duplicate.status_code == 422
    assert "already exists" in duplicate.json()["detail"]

    different_label = _create_relation(
        client,
        project["id"],
        document["id"],
        label="treats",
        head_id=head["id"],
        tail_id=tail["id"],
    )
    assert different_label.status_code == 200

    reverse_direction = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=tail["id"],
        tail_id=head["id"],
    )
    assert reverse_direction.status_code == 200


def test_update_cannot_create_duplicate_relation(client):
    project = _create_project(client, "relation-update-duplicates")
    document = _create_document(client, project["id"], DOCUMENT_TEXT)
    head = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=0,
        end_offset=9,
        text_span="Ibuprofen",
    )
    tail = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="AdverseEvent",
        start_offset=17,
        end_offset=42,
        text_span="gastrointestinal bleeding",
    )
    causes = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=head["id"],
        tail_id=tail["id"],
    ).json()
    treats = _create_relation(
        client,
        project["id"],
        document["id"],
        label="treats",
        head_id=head["id"],
        tail_id=tail["id"],
    ).json()

    collide = client.patch(f"/api/annotations/{treats['id']}", json={"label": "causes"})
    assert collide.status_code == 422
    assert "already exists" in collide.json()["detail"]

    self_update = client.patch(f"/api/annotations/{causes['id']}", json={"label": "causes"})
    assert self_update.status_code == 200


def test_relation_constraint_settings_round_trip(client):
    response = client.post(
        "/api/projects",
        json={
            "name": "constraint-shape-valid",
            "tasks": [
                {
                    "annotation_type": "relation",
                    "display_name": "Relations",
                    "labels": [{"name": "causes", "color": "#8e3a3a"}],
                    "settings": VALID_CONSTRAINTS,
                }
            ],
            "settings": {},
        },
    )
    assert response.status_code == 200
    task = response.json()["tasks"][0]
    assert task["settings"] == VALID_CONSTRAINTS


def test_malformed_relation_constraints_are_rejected(client):
    bad_shapes = [
        {"relation_constraints": ["causes"]},
        {"relation_constraints": {"causes": {"head": "Drug", "tail": []}}},
        {"relation_constraints": {"causes": {"heads": ["Drug"]}}},
    ]
    for index, settings in enumerate(bad_shapes):
        response = client.post(
            "/api/projects",
            json={
                "name": f"constraint-shape-bad-{index}",
                "tasks": [
                    {
                        "annotation_type": "relation",
                        "display_name": "Relations",
                        "labels": [{"name": "causes", "color": "#8e3a3a"}],
                        "settings": settings,
                    }
                ],
                "settings": {},
            },
        )
        assert response.status_code == 422, f"shape {index} was accepted"


def test_task_patch_validates_relation_constraints(client):
    project_response = client.post(
        "/api/projects",
        json={
            "name": "constraint-shape-patch",
            "tasks": [
                {
                    "annotation_type": "relation",
                    "display_name": "Relations",
                    "labels": [{"name": "causes", "color": "#8e3a3a"}],
                }
            ],
            "settings": {},
        },
    )
    assert project_response.status_code == 200
    project = project_response.json()
    task_id = project["tasks"][0]["id"]

    good_patch = client.patch(
        f"/api/projects/{project['id']}/tasks/{task_id}",
        json={"settings": VALID_CONSTRAINTS},
    )
    assert good_patch.status_code == 200
    assert good_patch.json()["settings"] == VALID_CONSTRAINTS

    bad_patch = client.patch(
        f"/api/projects/{project['id']}/tasks/{task_id}",
        json={"settings": {"relation_constraints": {"causes": {"head": 7}}}},
    )
    assert bad_patch.status_code == 422


def _create_strict_constrained_project(client, name: str) -> dict:
    response = client.post(
        "/api/projects",
        json={
            "name": name,
            "annotation_validation_mode": "strict",
            "tasks": [
                {
                    "annotation_type": "entity",
                    "display_name": "Entities",
                    "labels": [
                        {"name": "Drug", "color": "#16a34a"},
                        {"name": "AdverseEvent", "color": "#8e3a3a"},
                        {"name": "Disease", "color": "#4a5a8a"},
                    ],
                },
                {
                    "annotation_type": "relation",
                    "display_name": "Relations",
                    "labels": [
                        {"name": "causes", "color": "#8e3a3a"},
                        {"name": "treats", "color": "#16a34a"},
                    ],
                    "settings": {
                        "relation_constraints": {
                            "causes": {"head": ["Drug"], "tail": ["AdverseEvent"]},
                        }
                    },
                },
            ],
            "settings": {},
        },
    )
    assert response.status_code == 200
    return response.json()


def test_strict_mode_enforces_relation_constraints(client):
    project = _create_strict_constrained_project(client, "strict-relation-constraints")
    document = _create_document(client, project["id"], DOCUMENT_TEXT)
    drug = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=0,
        end_offset=9,
        text_span="Ibuprofen",
    )
    adverse_event = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="AdverseEvent",
        start_offset=17,
        end_offset=42,
        text_span="gastrointestinal bleeding",
    )
    disease = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Disease",
        start_offset=54,
        end_offset=62,
        text_span="patients",
    )

    allowed = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=drug["id"],
        tail_id=adverse_event["id"],
    )
    assert allowed.status_code == 200

    bad_head = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=adverse_event["id"],
        tail_id=drug["id"],
    )
    assert bad_head.status_code == 422
    assert "head" in bad_head.json()["detail"]

    bad_tail = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=drug["id"],
        tail_id=disease["id"],
    )
    assert bad_tail.status_code == 422
    assert "tail" in bad_tail.json()["detail"]

    unconstrained_label = _create_relation(
        client,
        project["id"],
        document["id"],
        label="treats",
        head_id=drug["id"],
        tail_id=disease["id"],
    )
    assert unconstrained_label.status_code == 200


def test_relaxed_mode_ignores_relation_constraints(client):
    project_response = client.post(
        "/api/projects",
        json={
            "name": "relaxed-relation-constraints",
            "annotation_validation_mode": "relaxed",
            "tasks": [
                {
                    "annotation_type": "relation",
                    "display_name": "Relations",
                    "labels": [{"name": "causes", "color": "#8e3a3a"}],
                    "settings": {
                        "relation_constraints": {
                            "causes": {"head": ["Drug"], "tail": ["AdverseEvent"]},
                        }
                    },
                },
            ],
            "settings": {},
        },
    )
    assert project_response.status_code == 200
    project = project_response.json()
    document = _create_document(client, project["id"], DOCUMENT_TEXT)
    first = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Anything",
        start_offset=0,
        end_offset=9,
        text_span="Ibuprofen",
    )
    second = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Whatever",
        start_offset=17,
        end_offset=42,
        text_span="gastrointestinal bleeding",
    )
    response = _create_relation(
        client,
        project["id"],
        document["id"],
        label="causes",
        head_id=first["id"],
        tail_id=second["id"],
    )
    assert response.status_code == 200
