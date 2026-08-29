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
            "external_id": "PMID:test-flow",
            "title": "API flow document",
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


def test_project_document_and_guideline_routes(client):
    project = _create_project(client, "api-flow-project")

    get_project_response = client.get(f"/api/projects/{project['id']}")
    assert get_project_response.status_code == 200
    assert get_project_response.json()["name"] == project["name"]

    document_text = "Patients receiving low-dose aspirin therapy were excluded."
    document = _create_document(
        client,
        project["id"],
        document_text,
    )
    assert document["sentences"] == [[0, len(document_text)]]

    list_documents_response = client.get("/api/documents", params={"project_id": project["id"]})
    assert list_documents_response.status_code == 200
    assert [item["id"] for item in list_documents_response.json()] == [document["id"]]

    get_document_response = client.get(f"/api/documents/{document['id']}")
    assert get_document_response.status_code == 200
    assert get_document_response.json()["text"] == document["text"]

    missing_document_response = client.get("/api/documents/9999")
    assert missing_document_response.status_code == 404

    guideline_response = client.post(
        "/api/guidelines",
        json={
            "project_id": project["id"],
            "version_label": "v1",
            "markdown": "# Guideline\n\nAnnotate drug mentions.",
            "author_id": "tester",
            "status": "active",
        },
    )
    assert guideline_response.status_code == 200

    list_guidelines_response = client.get(
        "/api/guidelines",
        params={"project_id": project["id"]},
    )
    assert list_guidelines_response.status_code == 200
    guidelines = list_guidelines_response.json()
    assert len(guidelines) == 1
    assert guidelines[0]["version_label"] == "v1"


def test_active_guideline_version_supersedes_prior_active_version(client):
    project = _create_project(client, "guideline-active-version-project")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )

    first_response = client.post(
        "/api/guidelines",
        json={
            "project_id": project["id"],
            "version_label": "v1",
            "markdown": "# Guideline v1",
            "author_id": "tester",
            "status": "active",
        },
    )
    assert first_response.status_code == 200
    first = first_response.json()

    second_response = client.post(
        "/api/guidelines",
        json={
            "project_id": project["id"],
            "version_label": "v2",
            "markdown": "# Guideline v2",
            "author_id": "tester",
            "status": "active",
        },
    )
    assert second_response.status_code == 200
    second = second_response.json()

    list_response = client.get("/api/guidelines", params={"project_id": project["id"]})
    assert list_response.status_code == 200
    guidelines = list_response.json()
    statuses_by_id = {guideline["id"]: guideline["status"] for guideline in guidelines}

    assert statuses_by_id[first["id"]] == "superseded"
    assert statuses_by_id[second["id"]] == "active"
    assert [guideline["id"] for guideline in guidelines if guideline["status"] == "active"] == [
        second["id"]
    ]

    workbench_response = client.get(f"/api/annotation-workbench/documents/{document['id']}")
    assert workbench_response.status_code == 200
    assert workbench_response.json()["active_guideline"]["id"] == second["id"]


def test_annotation_workbench_returns_canvas_state(client):
    project_response = client.post(
        "/api/projects",
        json={
            "name": "annotation-workbench-project",
            "description": "annotation workbench project",
            "annotation_schema": {
                "labels": {
                    "entity": [{"name": "Drug", "color": "#7aa2f7"}],
                    "relation": [{"name": "treats", "color": "#2ecc71"}],
                }
            },
            "settings": {},
        },
    )
    assert project_response.status_code == 200
    project = project_response.json()

    document_text = "Patients receiving aspirin improved."
    document = _create_document(client, project["id"], document_text)

    guideline_response = client.post(
        "/api/guidelines",
        json={
            "project_id": project["id"],
            "version_label": "v1",
            "markdown": "# Guideline\n\nAnnotate drug mentions.",
            "author_id": "tester",
            "status": "active",
        },
    )
    assert guideline_response.status_code == 200
    guideline = guideline_response.json()

    head_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )
    tail_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="PatientGroup",
        start_offset=0,
        end_offset=8,
        text_span="Patients",
    )
    relation_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "relation",
            "label": "treats",
            "head_annotation_id": head_annotation["id"],
            "tail_annotation_id": tail_annotation["id"],
        },
    )
    assert relation_response.status_code == 200
    relation = relation_response.json()

    response = client.get(f"/api/annotation-workbench/documents/{document['id']}")
    assert response.status_code == 200
    workbench = response.json()

    assert workbench["project"]["id"] == project["id"]
    assert workbench["project"]["annotation_schema"]["labels"]["entity"][0]["name"] == "Drug"
    assert workbench["document"]["id"] == document["id"]
    assert workbench["document"]["text"] == document_text
    assert workbench["document"]["sentences"] == document["sentences"]
    assert workbench["active_guideline"]["id"] == guideline["id"]
    assert workbench["active_guideline"]["markdown"] == guideline["markdown"]

    specs_by_name = {item["name"]: item for item in workbench["annotation_type_specs"]}
    assert specs_by_name["entity"]["requires_span"] is True
    assert specs_by_name["relation"]["requires_head_tail"] is True

    annotations_by_id = {item["id"]: item for item in workbench["annotations"]}
    assert set(annotations_by_id) == {
        head_annotation["id"],
        tail_annotation["id"],
        relation["id"],
    }
    assert annotations_by_id[relation["id"]]["head_annotation_id"] == head_annotation["id"]
    assert annotations_by_id[relation["id"]]["tail_annotation_id"] == tail_annotation["id"]
    assert workbench["correction_locked_annotation_ids"] == []


def test_annotation_workbench_reports_correction_locked_annotations(client):
    project = _create_project(client, "annotation-workbench-correction-lock-project")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    original_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )
    corrected_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )
    correction_response = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "original_annotation_id": original_annotation["id"],
            "corrected_annotation_id": corrected_annotation["id"],
            "correction_source": "adjudication",
            "error_type": "boundary_error",
        },
    )
    assert correction_response.status_code == 200

    response = client.get(f"/api/annotation-workbench/documents/{document['id']}")
    assert response.status_code == 200
    workbench = response.json()

    assert sorted(workbench["correction_locked_annotation_ids"]) == sorted(
        [original_annotation["id"], corrected_annotation["id"]]
    )


def test_annotation_workbench_returns_404_for_missing_document(client):
    response = client.get("/api/annotation-workbench/documents/9999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_annotation_workbench_allows_missing_active_guideline(client):
    project = _create_project(client, "annotation-workbench-no-guideline-project")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )

    response = client.get(f"/api/annotation-workbench/documents/{document['id']}")
    assert response.status_code == 200
    workbench = response.json()
    assert workbench["active_guideline"] is None
    assert workbench["annotations"] == []


def test_project_routes_normalize_and_validate_annotation_schema(client):
    legacy_response = client.post(
        "/api/projects",
        json={
            "name": "legacy-project-schema",
            "description": "legacy schema shape",
            "annotation_schema": {
                "annotation_types": ["entity"],
                "labels": [{"name": "Drug", "color": "#7aa2f7"}],
            },
            "settings": {},
        },
    )
    assert legacy_response.status_code == 200
    assert legacy_response.json()["annotation_schema"] == {
        "labels": {
            "entity": [
                {
                    "name": "Drug",
                    "color": "#7aa2f7",
                    "description": None,
                }
            ]
        }
    }

    invalid_response = client.post(
        "/api/projects",
        json={
            "name": "invalid-project-schema",
            "description": "invalid schema shape",
            "annotation_schema": {
                "labels": {
                    "foo": [{"name": "Unsupported", "color": "#000000"}],
                }
            },
            "settings": {},
        },
    )
    assert invalid_response.status_code == 422
    assert "Unknown annotation_type keys in labels" in invalid_response.text

    extra_key_response = client.post(
        "/api/projects",
        json={
            "name": "extra-key-project-schema",
            "description": "schema with unrecognized top-level key",
            "annotation_schema": {
                "labels": {},
                "totally_unknown": True,
            },
            "settings": {},
        },
    )
    assert extra_key_response.status_code == 422
    assert "totally_unknown" in extra_key_response.text


def test_project_route_updates_editable_project_settings(client):
    project = _create_project(client, "editable-project-settings")

    update_response = client.patch(
        f"/api/projects/{project['id']}",
        json={
            "description": "updated description",
            "annotation_validation_mode": "strict",
            "settings": {"review_required": True},
        },
    )
    assert update_response.status_code == 200
    updated_project = update_response.json()
    assert updated_project["description"] == "updated description"
    assert updated_project["annotation_validation_mode"] == "strict"
    assert updated_project["settings"] == {"review_required": True}
    assert updated_project["annotation_schema"] == project["annotation_schema"]

    other_project = _create_project(client, "editable-project-conflict")
    conflict_response = client.patch(
        f"/api/projects/{project['id']}",
        json={"name": other_project["name"]},
    )
    assert conflict_response.status_code == 409

    invalid_mode_response = client.patch(
        f"/api/projects/{project['id']}",
        json={"annotation_validation_mode": "locked"},
    )
    assert invalid_mode_response.status_code == 422


def test_project_tasks_drive_workbench_configuration(client):
    project_response = client.post(
        "/api/projects",
        json={
            "name": "dynamic-workbench-project",
            "description": "explicit task config",
            "annotation_validation_mode": "strict",
            "tasks": [
                {
                    "annotation_type": "doc_label",
                    "display_name": "Document Annotation",
                    "labels": [{"name": "ClinicalTrial", "color": "#2563eb"}],
                    "sort_order": 0,
                },
                {
                    "annotation_type": "entity",
                    "display_name": "Entity Annotation",
                    "labels": [{"name": "Drug", "color": "#16a34a"}],
                    "sort_order": 1,
                },
            ],
            "settings": {},
        },
    )
    assert project_response.status_code == 200
    project = project_response.json()
    assert project["annotation_validation_mode"] == "strict"
    assert [task["annotation_type"] for task in project["tasks"]] == [
        "doc_label",
        "entity",
    ]
    assert project["annotation_schema"]["labels"]["doc_label"][0]["name"] == "ClinicalTrial"

    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )

    workbench_response = client.get(f"/api/annotation-workbench/documents/{document['id']}")
    assert workbench_response.status_code == 200
    workbench = workbench_response.json()
    assert [task["annotation_type"] for task in workbench["tasks"]] == [
        "doc_label",
        "entity",
    ]
    assert [spec["name"] for spec in workbench["annotation_type_specs"]] == [
        "doc_label",
        "entity",
    ]
    assert workbench["tasks"][0]["annotation_type_spec"]["requires_span"] is False
    assert workbench["tasks"][1]["annotation_type_spec"]["requires_span"] is True


def test_project_task_crud_syncs_annotation_schema(client):
    project = _create_project(client, "project-task-crud")

    create_task_response = client.post(
        f"/api/projects/{project['id']}/tasks",
        json={
            "annotation_type": "doc_label",
            "display_name": "Document Annotation",
            "labels": [{"name": "CaseReport", "color": "#2563eb"}],
        },
    )
    assert create_task_response.status_code == 200
    task = create_task_response.json()
    assert task["display_name"] == "Document Annotation"

    get_project_response = client.get(f"/api/projects/{project['id']}")
    assert get_project_response.status_code == 200
    assert get_project_response.json()["annotation_schema"]["labels"]["doc_label"][0][
        "name"
    ] == "CaseReport"

    duplicate_response = client.post(
        f"/api/projects/{project['id']}/tasks",
        json={
            "annotation_type": "doc_label",
            "display_name": "Duplicate",
            "labels": [{"name": "Other", "color": "#9333ea"}],
        },
    )
    assert duplicate_response.status_code == 409

    disable_response = client.patch(
        f"/api/projects/{project['id']}/tasks/{task['id']}",
        json={"enabled": False},
    )
    assert disable_response.status_code == 200
    assert disable_response.json()["enabled"] is False

    enabled_tasks_response = client.get(
        f"/api/projects/{project['id']}/tasks",
        params={"enabled_only": True},
    )
    assert enabled_tasks_response.status_code == 200
    assert enabled_tasks_response.json() == []

    synced_project_response = client.get(f"/api/projects/{project['id']}")
    assert synced_project_response.status_code == 200
    assert synced_project_response.json()["annotation_schema"] == {"labels": {}}

    delete_response = client.delete(f"/api/projects/{project['id']}/tasks/{task['id']}")
    assert delete_response.status_code == 204

    list_tasks_response = client.get(f"/api/projects/{project['id']}/tasks")
    assert list_tasks_response.status_code == 200
    assert list_tasks_response.json() == []


def test_project_task_sort_order_drives_schema_and_workbench(client):
    project_response = client.post(
        "/api/projects",
        json={
            "name": "project-task-order",
            "tasks": [
                {
                    "annotation_type": "entity",
                    "display_name": "Entity Annotation",
                    "labels": [{"name": "Drug", "color": "#16a34a"}],
                    "sort_order": 0,
                },
                {
                    "annotation_type": "doc_label",
                    "display_name": "Document Annotation",
                    "labels": [{"name": "ClinicalTrial", "color": "#2563eb"}],
                    "sort_order": 1,
                },
            ],
            "settings": {},
        },
    )
    assert project_response.status_code == 200
    project = project_response.json()
    tasks_by_type = {task["annotation_type"]: task for task in project["tasks"]}

    doc_label_order_response = client.patch(
        f"/api/projects/{project['id']}/tasks/{tasks_by_type['doc_label']['id']}",
        json={"sort_order": 0},
    )
    assert doc_label_order_response.status_code == 200
    entity_order_response = client.patch(
        f"/api/projects/{project['id']}/tasks/{tasks_by_type['entity']['id']}",
        json={"sort_order": 1},
    )
    assert entity_order_response.status_code == 200

    get_project_response = client.get(f"/api/projects/{project['id']}")
    assert get_project_response.status_code == 200
    assert list(get_project_response.json()["annotation_schema"]["labels"]) == [
        "doc_label",
        "entity",
    ]

    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    workbench_response = client.get(f"/api/annotation-workbench/documents/{document['id']}")
    assert workbench_response.status_code == 200
    assert [task["annotation_type"] for task in workbench_response.json()["tasks"]] == [
        "doc_label",
        "entity",
    ]


def test_annotation_validation_modes(client):
    strict_project_response = client.post(
        "/api/projects",
        json={
            "name": "strict-annotation-validation",
            "annotation_validation_mode": "strict",
            "tasks": [
                {
                    "annotation_type": "entity",
                    "display_name": "Entity Annotation",
                    "labels": [{"name": "Drug", "color": "#16a34a"}],
                }
            ],
            "settings": {},
        },
    )
    assert strict_project_response.status_code == 200
    strict_project = strict_project_response.json()
    strict_document = _create_document(
        client,
        strict_project["id"],
        "Patients receiving aspirin improved.",
    )

    valid_response = client.post(
        "/api/annotations",
        json={
            "project_id": strict_project["id"],
            "document_id": strict_document["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 19,
            "end_offset": 26,
            "text_span": "aspirin",
        },
    )
    assert valid_response.status_code == 200
    annotation = valid_response.json()

    invalid_label_response = client.post(
        "/api/annotations",
        json={
            "project_id": strict_project["id"],
            "document_id": strict_document["id"],
            "annotation_type": "entity",
            "label": "Disease",
            "start_offset": 0,
            "end_offset": 8,
            "text_span": "Patients",
        },
    )
    assert invalid_label_response.status_code == 422
    assert "Label 'Disease' is not enabled" in invalid_label_response.json()["detail"]

    invalid_type_response = client.post(
        "/api/annotations",
        json={
            "project_id": strict_project["id"],
            "document_id": strict_document["id"],
            "annotation_type": "doc_label",
            "label": "ClinicalTrial",
        },
    )
    assert invalid_type_response.status_code == 422
    assert "Annotation type 'doc_label' is not enabled" in (
        invalid_type_response.json()["detail"]
    )

    invalid_patch_response = client.patch(
        f"/api/annotations/{annotation['id']}",
        json={"label": "Disease"},
    )
    assert invalid_patch_response.status_code == 422
    assert "Label 'Disease' is not enabled" in invalid_patch_response.json()["detail"]

    relaxed_project = _create_project(client, "relaxed-annotation-validation")
    relaxed_document = _create_document(
        client,
        relaxed_project["id"],
        "Patients receiving aspirin improved.",
    )
    relaxed_response = client.post(
        "/api/annotations",
        json={
            "project_id": relaxed_project["id"],
            "document_id": relaxed_document["id"],
            "annotation_type": "entity",
            "label": "AdHocLabel",
            "start_offset": 19,
            "end_offset": 26,
            "text_span": "aspirin",
        },
    )
    assert relaxed_response.status_code == 200


def test_task_assignments_and_progress(client):
    project_response = client.post(
        "/api/projects",
        json={
            "name": "assignment-progress-project",
            "tasks": [
                {
                    "annotation_type": "entity",
                    "display_name": "Entity Annotation",
                    "labels": [{"name": "Drug", "color": "#16a34a"}],
                },
                {
                    "annotation_type": "doc_label",
                    "display_name": "Document Annotation",
                    "labels": [{"name": "ClinicalTrial", "color": "#2563eb"}],
                    "sort_order": 1,
                },
            ],
            "settings": {},
        },
    )
    assert project_response.status_code == 200
    project = project_response.json()
    tasks_by_type = {task["annotation_type"]: task for task in project["tasks"]}
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )

    assignment_response = client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": tasks_by_type["entity"]["id"],
            "document_id": document["id"],
            "annotator_id": "curator1",
            "assigned_by": "manager1",
        },
    )
    assert assignment_response.status_code == 200
    assignment = assignment_response.json()
    assert assignment["status"] == "assigned"

    duplicate_response = client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": tasks_by_type["entity"]["id"],
            "document_id": document["id"],
            "annotator_id": "curator1",
        },
    )
    assert duplicate_response.status_code == 409

    update_response = client.patch(
        f"/api/projects/{project['id']}/assignments/{assignment['id']}",
        json={"status": "submitted"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["status"] == "submitted"

    reopen_response = client.patch(
        f"/api/projects/{project['id']}/assignments/{assignment['id']}",
        json={"status": "in_progress"},
    )
    assert reopen_response.status_code == 409
    assert "cannot be reopened" in reopen_response.json()["detail"]

    forged_provenance = client.patch(
        f"/api/projects/{project['id']}/assignments/{assignment['id']}",
        json={"assigned_by": "forged-manager", "assigned_by_user_id": 999999},
    )
    assert forged_provenance.status_code == 422
    null_status = client.patch(
        f"/api/projects/{project['id']}/assignments/{assignment['id']}",
        json={"status": None},
    )
    assert null_status.status_code == 422

    doc_label_assignment_response = client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": tasks_by_type["doc_label"]["id"],
            "document_id": document["id"],
            "annotator_id": "curator2",
            "status": "completed",
        },
    )
    assert doc_label_assignment_response.status_code == 200

    filtered_response = client.get(
        f"/api/projects/{project['id']}/assignments",
        params={"status": "submitted"},
    )
    assert filtered_response.status_code == 200
    assert [item["id"] for item in filtered_response.json()] == [assignment["id"]]

    progress_response = client.get(f"/api/projects/{project['id']}/progress")
    assert progress_response.status_code == 200
    progress = progress_response.json()
    assert progress["total"] == 2
    assert progress["by_status"] == {"submitted": 1, "completed": 1}
    assert {item["annotator_id"]: item["by_status"] for item in progress["by_annotator"]} == {
        "curator1": {"submitted": 1},
        "curator2": {"completed": 1},
    }

    workbench_response = client.get(f"/api/annotation-workbench/documents/{document['id']}")
    assert workbench_response.status_code == 200
    # The editing workbench is always the current actor's private layer, even
    # for managers. Cross-annotator assignment comparison belongs in manager
    # workflows rather than being overlaid as editable local work.
    assert workbench_response.json()["assignments"] == []

    other_project = _create_project(client, "assignment-other-project")
    other_document = _create_document(
        client,
        other_project["id"],
        "Controls receiving placebo improved.",
    )
    cross_document_response = client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": tasks_by_type["entity"]["id"],
            "document_id": other_document["id"],
            "annotator_id": "curator3",
        },
    )
    assert cross_document_response.status_code == 422
    assert "does not belong to project" in cross_document_response.json()["detail"]

    other_task = client.post(
        f"/api/projects/{other_project['id']}/tasks",
        json={
            "annotation_type": "entity",
            "display_name": "Entity Annotation",
            "labels": [{"name": "Drug", "color": "#16a34a"}],
        },
    ).json()
    cross_task_response = client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": other_task["id"],
            "document_id": document["id"],
            "annotator_id": "curator3",
        },
    )
    assert cross_task_response.status_code == 404
    assert "Project task not found" in cross_task_response.json()["detail"]


def test_annotation_correction_and_error_guideline_routes(client):
    project = _create_project(client, "error-guideline-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving low-dose aspirin therapy were excluded.",
    )

    model_annotation_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 19,
            "end_offset": 43,
            "text_span": "low-dose aspirin therapy",
            "source": "model",
            "status": "rejected",
        },
    )
    assert model_annotation_response.status_code == 200
    model_annotation = model_annotation_response.json()

    human_annotation_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 28,
            "end_offset": 35,
            "text_span": "aspirin",
            "source": "human",
            "status": "gold",
            "annotator_id": "curator1",
        },
    )
    assert human_annotation_response.status_code == 200
    human_annotation = human_annotation_response.json()

    list_annotations_response = client.get(
        "/api/annotations",
        params={"document_id": document["id"]},
    )
    assert list_annotations_response.status_code == 200
    assert len(list_annotations_response.json()) == 2

    correction_response = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "original_annotation_id": model_annotation["id"],
            "corrected_annotation_id": human_annotation["id"],
            "correction_source": "adjudication",
            "correction_note": "Model included dose/context phrase.",
            "error_type": "boundary_error",
            "severity": "medium",
            "metadata_": {},
        },
    )
    assert correction_response.status_code == 200
    correction = correction_response.json()

    list_corrections_response = client.get(
        "/api/annotations/corrections",
        params={"project_id": project["id"]},
    )
    assert list_corrections_response.status_code == 200
    assert [item["id"] for item in list_corrections_response.json()] == [correction["id"]]

    patterns_response = client.get(
        "/api/co-learning/error-guideline/patterns",
        params={"project_id": project["id"]},
    )
    assert patterns_response.status_code == 200
    patterns = patterns_response.json()
    assert len(patterns) == 1
    pattern = patterns[0]
    assert pattern["error_type"] == "boundary_error"
    assert pattern["example_count"] == 1

    draft_atom_response = client.post(
        f"/api/co-learning/error-guideline/patterns/{pattern['id']}/draft-guideline-atom",
        json={"manager_id": "lei", "use_llm": False, "rule_text_override": None},
    )
    assert draft_atom_response.status_code == 200
    atom = draft_atom_response.json()
    assert atom["status"] == "pending"

    repeat_draft_atom_response = client.post(
        f"/api/co-learning/error-guideline/patterns/{pattern['id']}/draft-guideline-atom",
        json={"manager_id": "lei", "use_llm": False, "rule_text_override": None},
    )
    assert repeat_draft_atom_response.status_code == 200
    assert repeat_draft_atom_response.json()["id"] == atom["id"]

    list_atoms_response = client.get(
        "/api/co-learning/error-guideline/guideline-atoms",
        params={"project_id": project["id"]},
    )
    assert list_atoms_response.status_code == 200
    assert len(list_atoms_response.json()) == 1

    approve_atom_response = client.post(
        f"/api/co-learning/error-guideline/guideline-atoms/{atom['id']}/approve",
        json={"approved_by": "lei"},
    )
    assert approve_atom_response.status_code == 200
    approved_atom = approve_atom_response.json()
    assert approved_atom["status"] == "accepted"

    repeat_approve_response = client.post(
        f"/api/co-learning/error-guideline/guideline-atoms/{atom['id']}/approve",
        json={"approved_by": "lei"},
    )
    assert repeat_approve_response.status_code == 200
    assert repeat_approve_response.json()["status"] == "accepted"

    training_actions_response = client.get(
        "/api/co-learning/error-guideline/training-actions",
        params={"project_id": project["id"]},
    )
    assert training_actions_response.status_code == 200
    training_actions = training_actions_response.json()
    assert len(training_actions) == 1
    assert training_actions[0]["guideline_atom_id"] == atom["id"]

    micro_questions_response = client.get(
        "/api/co-learning/error-guideline/micro-question-templates",
        params={"project_id": project["id"]},
    )
    assert micro_questions_response.status_code == 200
    micro_questions = micro_questions_response.json()
    assert len(micro_questions) == 1
    assert micro_questions[0]["guideline_atom_id"] == atom["id"]

    missing_correction_response = client.post(
        "/api/co-learning/error-guideline/patterns/from-correction/9999",
    )
    assert missing_correction_response.status_code == 404


def test_annotation_routes_validate_and_support_crud(client):
    project = _create_project(client, "annotation-crud-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving low-dose aspirin therapy were excluded.",
    )

    invalid_entity_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug",
        },
    )
    assert invalid_entity_response.status_code == 422
    assert "requires start_offset and end_offset" in invalid_entity_response.json()["detail"]

    invalid_relation_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "relation",
            "label": "treats",
        },
    )
    assert invalid_relation_response.status_code == 422
    assert (
        "requires head_annotation_id and tail_annotation_id"
        in invalid_relation_response.json()["detail"]
    )

    invalid_type_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "bogus",
            "label": "Drug",
        },
    )
    assert invalid_type_response.status_code == 422
    invalid_type_detail = invalid_type_response.json()["detail"]
    assert any(
        err["loc"] == ["body", "annotation_type"] and "literal_error" in err["type"]
        for err in invalid_type_detail
    )

    head_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 28,
            "end_offset": 35,
            "text_span": "aspirin",
        },
    )
    assert head_response.status_code == 200
    head_annotation = head_response.json()

    tail_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Disease",
            "start_offset": 0,
            "end_offset": 8,
            "text_span": "Patients",
        },
    )
    assert tail_response.status_code == 200
    tail_annotation = tail_response.json()

    relation_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "relation",
            "label": "treats",
            "head_annotation_id": head_annotation["id"],
            "tail_annotation_id": tail_annotation["id"],
        },
    )
    assert relation_response.status_code == 200
    relation = relation_response.json()
    assert relation["head_annotation_id"] == head_annotation["id"]
    assert relation["tail_annotation_id"] == tail_annotation["id"]

    get_response = client.get(f"/api/annotations/{head_annotation['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["label"] == "Drug"

    missing_get_response = client.get("/api/annotations/9999")
    assert missing_get_response.status_code == 404

    patch_response = client.patch(
        f"/api/annotations/{head_annotation['id']}",
        json={"label": "Drug2"},
    )
    assert patch_response.status_code == 200
    updated_annotation = patch_response.json()
    assert updated_annotation["label"] == "Drug2"
    assert updated_annotation["start_offset"] == 28
    assert updated_annotation["end_offset"] == 35

    delete_response = client.delete(f"/api/annotations/{relation['id']}")
    assert delete_response.status_code == 204

    repeat_delete_response = client.delete(f"/api/annotations/{relation['id']}")
    assert repeat_delete_response.status_code == 404

    for cleanup_id in (head_annotation["id"], tail_annotation["id"]):
        cleanup_response = client.delete(f"/api/annotations/{cleanup_id}")
        assert cleanup_response.status_code == 204


def test_relation_annotations_require_existing_same_scope_targets(client):
    project = _create_project(client, "annotation-relation-target-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    other_document = _create_document(
        client,
        project["id"],
        "Controls receiving placebo improved.",
    )

    head_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )
    tail_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Disease",
        start_offset=0,
        end_offset=8,
        text_span="Patients",
    )
    other_document_annotation = _create_entity_annotation(
        client,
        project["id"],
        other_document["id"],
        label="Disease",
        start_offset=0,
        end_offset=8,
        text_span="Controls",
    )

    missing_target_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "relation",
            "label": "treats",
            "head_annotation_id": 9999,
            "tail_annotation_id": tail_annotation["id"],
        },
    )
    assert missing_target_response.status_code == 422
    assert missing_target_response.json()["detail"] == (
        "One or more referenced annotations are unavailable"
    )

    cross_document_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "relation",
            "label": "treats",
            "head_annotation_id": head_annotation["id"],
            "tail_annotation_id": other_document_annotation["id"],
        },
    )
    assert cross_document_response.status_code == 422
    assert cross_document_response.json()["detail"] == (
        "One or more referenced annotations are unavailable"
    )

    relation_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "relation",
            "label": "treats",
            "head_annotation_id": head_annotation["id"],
            "tail_annotation_id": tail_annotation["id"],
        },
    )
    assert relation_response.status_code == 200
    relation = relation_response.json()

    patch_missing_response = client.patch(
        f"/api/annotations/{relation['id']}",
        json={"head_annotation_id": 9999},
    )
    assert patch_missing_response.status_code == 422
    assert patch_missing_response.json()["detail"] == (
        "One or more referenced annotations are unavailable"
    )

    patch_cross_document_response = client.patch(
        f"/api/annotations/{relation['id']}",
        json={"tail_annotation_id": other_document_annotation["id"]},
    )
    assert patch_cross_document_response.status_code == 422
    assert patch_cross_document_response.json()["detail"] == (
        "One or more referenced annotations are unavailable"
    )


def test_delete_annotation_rejects_relation_target_until_relation_deleted(client):
    project = _create_project(client, "annotation-delete-relation-target-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    head_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )
    tail_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Disease",
        start_offset=0,
        end_offset=8,
        text_span="Patients",
    )

    relation_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "relation",
            "label": "treats",
            "head_annotation_id": head_annotation["id"],
            "tail_annotation_id": tail_annotation["id"],
        },
    )
    assert relation_response.status_code == 200
    relation = relation_response.json()

    blocked_delete_response = client.delete(f"/api/annotations/{head_annotation['id']}")
    assert blocked_delete_response.status_code == 409
    assert "delete those relations first" in blocked_delete_response.json()["detail"]

    get_head_response = client.get(f"/api/annotations/{head_annotation['id']}")
    assert get_head_response.status_code == 200

    delete_relation_response = client.delete(f"/api/annotations/{relation['id']}")
    assert delete_relation_response.status_code == 204

    delete_head_response = client.delete(f"/api/annotations/{head_annotation['id']}")
    assert delete_head_response.status_code == 204


def test_annotation_response_includes_timestamps(client):
    from datetime import datetime

    project = _create_project(client, "annotation-timestamps-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )
    assert "created_at" in annotation
    assert "updated_at" in annotation
    datetime.fromisoformat(annotation["created_at"])
    datetime.fromisoformat(annotation["updated_at"])


def test_annotation_create_rejects_invalid_source(client):
    project = _create_project(client, "annotation-invalid-source-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 19,
            "end_offset": 26,
            "text_span": "aspirin",
            "source": "bogus",
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(
        err["loc"] == ["body", "source"] and "literal_error" in err["type"]
        for err in detail
    )


def test_annotation_create_rejects_out_of_range_confidence(client):
    project = _create_project(client, "annotation-invalid-confidence-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    base_payload = {
        "project_id": project["id"],
        "document_id": document["id"],
        "annotation_type": "entity",
        "label": "Drug",
        "start_offset": 19,
        "end_offset": 26,
        "text_span": "aspirin",
    }
    for invalid_confidence in (-0.1, 1.5):
        response = client.post(
            "/api/annotations",
            json={**base_payload, "confidence": invalid_confidence},
        )
        assert response.status_code == 422, invalid_confidence
        detail = response.json()["detail"]
        assert any(err["loc"] == ["body", "confidence"] for err in detail), invalid_confidence


def test_create_project_rejects_duplicate_name(client):
    name = "duplicate-name-project"
    first = client.post(
        "/api/projects",
        json={
            "name": name,
            "description": "first",
            "annotation_schema": {"labels": {}},
            "settings": {},
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/api/projects",
        json={
            "name": name,
            "description": "second",
            "annotation_schema": {"labels": {}},
            "settings": {},
        },
    )
    assert second.status_code == 409
    assert f"Project with name '{name}' already exists" in second.json()["detail"]


def test_create_document_rejects_missing_project(client):
    response = client.post(
        "/api/documents",
        json={
            "project_id": 9999,
            "external_id": "PMID:none",
            "title": "orphan",
            "text": "Text body.",
            "source": "manual",
            "metadata_": {},
        },
    )
    assert response.status_code == 422
    assert "Project 9999 not found" in response.json()["detail"]


def test_create_guideline_version_rejects_missing_project(client):
    response = client.post(
        "/api/guidelines",
        json={
            "project_id": 9999,
            "version_label": "v1",
            "markdown": "# Guideline",
            "author_id": "tester",
            "status": "active",
        },
    )
    assert response.status_code == 422
    assert "Project 9999 not found" in response.json()["detail"]


def test_annotation_create_rejects_invalid_spans(client):
    project = _create_project(client, "annotation-invalid-span-flow")
    document_text = "Patients receiving aspirin improved."
    document = _create_document(client, project["id"], document_text)
    base_payload = {
        "project_id": project["id"],
        "document_id": document["id"],
        "annotation_type": "entity",
        "label": "Drug",
    }

    cases = [
        (
            {"start_offset": 100, "end_offset": 10, "text_span": "x"},
            "start_offset must be less than end_offset",
        ),
        (
            {"start_offset": -1, "end_offset": 5, "text_span": "x"},
            "Offsets must be non-negative",
        ),
        (
            {"start_offset": 19, "end_offset": 19, "text_span": ""},
            "start_offset must be less than end_offset",
        ),
        (
            {"start_offset": 0, "end_offset": len(document_text) + 1, "text_span": "x"},
            f"exceeds document length {len(document_text)}",
        ),
        (
            {"start_offset": 19, "end_offset": 26, "text_span": "ASPIRIN"},
            "text_span does not match document text",
        ),
    ]
    for overrides, expected_msg in cases:
        response = client.post("/api/annotations", json={**base_payload, **overrides})
        assert response.status_code == 422, overrides
        assert expected_msg in response.json()["detail"], overrides

    happy_response = client.post(
        "/api/annotations",
        json={
            **base_payload,
            "start_offset": 19,
            "end_offset": 26,
            "text_span": "aspirin",
        },
    )
    assert happy_response.status_code == 200


def test_patch_annotation_rejects_invalid_span(client):
    project = _create_project(client, "annotation-patch-invalid-span-flow")
    document_text = "Patients receiving aspirin improved."
    document = _create_document(client, project["id"], document_text)
    annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )

    reversed_patch = client.patch(
        f"/api/annotations/{annotation['id']}",
        json={"start_offset": 100, "end_offset": 10},
    )
    assert reversed_patch.status_code == 422
    assert "start_offset must be less than end_offset" in reversed_patch.json()["detail"]

    text_mismatch_patch = client.patch(
        f"/api/annotations/{annotation['id']}",
        json={"text_span": "ibuprofen"},
    )
    assert text_mismatch_patch.status_code == 422
    assert "text_span does not match document text" in text_mismatch_patch.json()["detail"]


def test_annotation_create_rejects_cross_project_document(client):
    project_a = _create_project(client, "annotation-scope-project-a")
    project_b = _create_project(client, "annotation-scope-project-b")
    document_b = _create_document(
        client,
        project_b["id"],
        "Patients receiving aspirin improved.",
    )

    response = client.post(
        "/api/annotations",
        json={
            "project_id": project_a["id"],
            "document_id": document_b["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 19,
            "end_offset": 26,
            "text_span": "aspirin",
        },
    )
    assert response.status_code == 422
    assert (
        f"Document {document_b['id']} does not belong to project {project_a['id']}"
        in response.json()["detail"]
    )


def test_annotation_create_rejects_missing_document(client):
    project = _create_project(client, "annotation-scope-missing-document")

    response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": 9999,
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 0,
            "end_offset": 1,
            "text_span": "x",
        },
    )
    assert response.status_code == 422
    assert "Document 9999 not found" in response.json()["detail"]


def test_annotation_create_rejects_cross_project_guideline_version(client):
    project_a = _create_project(client, "annotation-scope-guideline-project-a")
    project_b = _create_project(client, "annotation-scope-guideline-project-b")
    document_a = _create_document(
        client,
        project_a["id"],
        "Patients receiving aspirin improved.",
    )
    guideline_response = client.post(
        "/api/guidelines",
        json={
            "project_id": project_b["id"],
            "version_label": "v1",
            "markdown": "# Guideline",
            "author_id": "tester",
            "status": "active",
        },
    )
    assert guideline_response.status_code == 200
    guideline = guideline_response.json()

    response = client.post(
        "/api/annotations",
        json={
            "project_id": project_a["id"],
            "document_id": document_a["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 19,
            "end_offset": 26,
            "text_span": "aspirin",
            "guideline_version_id": guideline["id"],
        },
    )
    assert response.status_code == 422
    assert (
        f"GuidelineVersion {guideline['id']} does not belong to project {project_a['id']}"
        in response.json()["detail"]
    )


def test_correction_create_rejects_cross_project_document(client):
    project_a = _create_project(client, "correction-scope-project-a")
    project_b = _create_project(client, "correction-scope-project-b")
    document_b = _create_document(
        client,
        project_b["id"],
        "Patients receiving aspirin improved.",
    )

    response = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": project_a["id"],
            "document_id": document_b["id"],
            "correction_source": "adjudication",
            "error_type": "boundary_error",
        },
    )
    assert response.status_code == 422
    assert (
        f"Document {document_b['id']} does not belong to project {project_a['id']}"
        in response.json()["detail"]
    )


def test_correction_create_rejects_cross_scope_annotation_refs(client):
    project = _create_project(client, "correction-scope-cross-doc")
    document_a = _create_document(client, project["id"], "Document A.")
    document_b = _create_document(client, project["id"], "Document B.")
    annotation_b = _create_entity_annotation(
        client,
        project["id"],
        document_b["id"],
        label="Drug",
        start_offset=0,
        end_offset=1,
        text_span="D",
    )

    response = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": project["id"],
            "document_id": document_a["id"],
            "original_annotation_id": annotation_b["id"],
            "correction_source": "adjudication",
            "error_type": "boundary_error",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "One or more referenced annotations are unavailable"
    )


def test_correction_create_rejects_missing_annotation_ref(client):
    project = _create_project(client, "correction-scope-missing-ref")
    document = _create_document(client, project["id"], "Document text.")

    response = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "original_annotation_id": 9999,
            "correction_source": "adjudication",
            "error_type": "boundary_error",
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "One or more referenced annotations are unavailable"
    )


def test_correction_create_rejects_invalid_source_and_severity(client):
    project = _create_project(client, "correction-invalid-enum-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )
    base_payload = {
        "project_id": project["id"],
        "document_id": document["id"],
        "original_annotation_id": annotation["id"],
        "corrected_annotation_id": annotation["id"],
        "error_type": "boundary_error",
    }

    bad_source = client.post(
        "/api/annotations/corrections",
        json={**base_payload, "correction_source": "bogus"},
    )
    assert bad_source.status_code == 422
    assert any(
        err["loc"] == ["body", "correction_source"] and "literal_error" in err["type"]
        for err in bad_source.json()["detail"]
    )

    bad_severity = client.post(
        "/api/annotations/corrections",
        json={**base_payload, "severity": "extreme"},
    )
    assert bad_severity.status_code == 422
    assert any(
        err["loc"] == ["body", "severity"] and "literal_error" in err["type"]
        for err in bad_severity.json()["detail"]
    )


def test_annotation_create_rejects_invalid_status(client):
    project = _create_project(client, "annotation-invalid-status-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 19,
            "end_offset": 26,
            "text_span": "aspirin",
            "status": "bogus",
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(
        err["loc"] == ["body", "status"] and "literal_error" in err["type"]
        for err in detail
    )


def test_patch_relation_rejects_self_reference(client):
    project = _create_project(client, "annotation-self-reference-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    head_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )
    tail_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Disease",
        start_offset=0,
        end_offset=8,
        text_span="Patients",
    )

    relation_response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "relation",
            "label": "treats",
            "head_annotation_id": head_annotation["id"],
            "tail_annotation_id": tail_annotation["id"],
        },
    )
    assert relation_response.status_code == 200
    relation = relation_response.json()

    head_self_response = client.patch(
        f"/api/annotations/{relation['id']}",
        json={"head_annotation_id": relation["id"]},
    )
    assert head_self_response.status_code == 422
    assert "cannot reference itself: head_annotation_id" in (
        head_self_response.json()["detail"]
    )

    tail_self_response = client.patch(
        f"/api/annotations/{relation['id']}",
        json={"tail_annotation_id": relation["id"]},
    )
    assert tail_self_response.status_code == 422
    assert "cannot reference itself: tail_annotation_id" in (
        tail_self_response.json()["detail"]
    )


def test_delete_annotation_rejects_target_of_correction(client):
    project = _create_project(client, "annotation-delete-correction-target-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )
    original_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )
    corrected_annotation = _create_entity_annotation(
        client,
        project["id"],
        document["id"],
        label="Drug",
        start_offset=19,
        end_offset=26,
        text_span="aspirin",
    )

    correction_response = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "original_annotation_id": original_annotation["id"],
            "corrected_annotation_id": corrected_annotation["id"],
            "correction_source": "adjudication",
            "error_type": "boundary_error",
        },
    )
    assert correction_response.status_code == 200

    blocked_original_response = client.delete(f"/api/annotations/{original_annotation['id']}")
    assert blocked_original_response.status_code == 409
    assert "delete those corrections first" in blocked_original_response.json()["detail"]

    blocked_corrected_response = client.delete(f"/api/annotations/{corrected_annotation['id']}")
    assert blocked_corrected_response.status_code == 409
    assert "delete those corrections first" in blocked_corrected_response.json()["detail"]


def test_entity_annotation_rejects_head_or_tail_ids(client):
    project = _create_project(client, "annotation-entity-rejects-head-tail-flow")
    document = _create_document(
        client,
        project["id"],
        "Patients receiving aspirin improved.",
    )

    response = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 19,
            "end_offset": 26,
            "text_span": "aspirin",
            "head_annotation_id": 1,
        },
    )
    assert response.status_code == 422
    assert "must not set head_annotation_id or tail_annotation_id" in response.json()["detail"]


def test_manual_error_guideline_create_endpoints_are_retry_safe(client):
    project = _create_project(client, "manual-error-guideline-create-flow")

    pattern_payload = {
        "project_id": project["id"],
        "task_type": "entity",
        "error_type": "label_confusion",
        "label_type": "Drug",
        "description": "Wrong label type was assigned.",
        "example_count": 2,
        "severity": "high",
        "detected_from": "manual",
        "status": "active",
        "example_ids": [{"document_id": 7, "correction_id": 11}],
        "metadata_": {"source": "manual-review"},
    }
    create_pattern_response = client.post(
        "/api/co-learning/error-guideline/patterns",
        json=pattern_payload,
    )
    assert create_pattern_response.status_code == 200
    pattern = create_pattern_response.json()

    retry_pattern_response = client.post(
        "/api/co-learning/error-guideline/patterns",
        json=pattern_payload,
    )
    assert retry_pattern_response.status_code == 200
    assert retry_pattern_response.json()["id"] == pattern["id"]

    conflicting_pattern_response = client.post(
        "/api/co-learning/error-guideline/patterns",
        json={**pattern_payload, "description": "Conflicting active pattern."},
    )
    assert conflicting_pattern_response.status_code == 409
    assert "active error pattern already exists" in conflicting_pattern_response.json()["detail"]

    list_patterns_response = client.get(
        "/api/co-learning/error-guideline/patterns",
        params={"project_id": project["id"]},
    )
    assert list_patterns_response.status_code == 200
    assert len(list_patterns_response.json()) == 1

    training_action_payload = {
        "project_id": project["id"],
        "error_pattern_id": pattern["id"],
        "guideline_atom_id": None,
        "action_type": "manual_retrain",
        "target_model": "bert_ner",
        "example_ids": [{"document_id": 7, "correction_id": 11}],
        "priority": "high",
        "status": "planned",
        "notes": "Retry-safe manual training action.",
    }
    create_training_action_response = client.post(
        "/api/co-learning/error-guideline/training-actions",
        json=training_action_payload,
    )
    assert create_training_action_response.status_code == 200
    training_action = create_training_action_response.json()

    retry_training_action_response = client.post(
        "/api/co-learning/error-guideline/training-actions",
        json=training_action_payload,
    )
    assert retry_training_action_response.status_code == 200
    assert retry_training_action_response.json()["id"] == training_action["id"]

    list_training_actions_response = client.get(
        "/api/co-learning/error-guideline/training-actions",
        params={"project_id": project["id"]},
    )
    assert list_training_actions_response.status_code == 200
    assert len(list_training_actions_response.json()) == 1

    micro_question_payload = {
        "project_id": project["id"],
        "guideline_atom_id": None,
        "error_type": "label_confusion",
        "target_annotation_type": "entity",
        "question_text": "Which label is correct for this biomedical mention?",
        "answer_options": ["Drug", "AdverseEvent"],
        "status": "active",
    }
    create_micro_question_response = client.post(
        "/api/co-learning/error-guideline/micro-question-templates",
        json=micro_question_payload,
    )
    assert create_micro_question_response.status_code == 200
    micro_question = create_micro_question_response.json()

    retry_micro_question_response = client.post(
        "/api/co-learning/error-guideline/micro-question-templates",
        json=micro_question_payload,
    )
    assert retry_micro_question_response.status_code == 200
    assert retry_micro_question_response.json()["id"] == micro_question["id"]

    list_micro_questions_response = client.get(
        "/api/co-learning/error-guideline/micro-question-templates",
        params={"project_id": project["id"]},
    )
    assert list_micro_questions_response.status_code == 200
    assert len(list_micro_questions_response.json()) == 1
