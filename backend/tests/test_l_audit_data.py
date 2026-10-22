from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError as PydanticValidationError
from test_export_safety import export_source as export_source
from test_guideline_impact_submissions import pilot as pilot
from test_lineage_coverage import frozen_scope as frozen_scope
from test_paper_entity_rounds import paper_tasks as paper_tasks
from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.importers import pubmed
from al_medlit.importers.service import normalize_pmids
from al_medlit.workflow.preparation_schemas import TrainingPreparationRequest


def test_seed_rejects_values_outside_postgres_integer():
    for seed in (-(2**31) - 1, 2**31):
        with pytest.raises(PydanticValidationError, match="seed"):
            TrainingPreparationRequest.model_validate(
                {
                    "name": "test",
                    "task_version_id": 1,
                    "seed": seed,
                    "sources": [{"dataset_version_id": 1, "label_field": "label"}],
                }
            )


@pytest.mark.parametrize("fetch", [pubmed.fetch_pubmed_metadata, pubmed.fetch_pmc_documents])
def test_malformed_xml_is_an_upstream_error(fetch):
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, text="bad")))
    with pytest.raises(pubmed.ImporterFetchError, match="XML"):
        fetch(client, ["123"])


def test_id_converter_rejects_non_string_pmcid():
    with pytest.raises(ValueError):
        pubmed.parse_idconv_json({"records": [{"pmid": "123", "pmcid": {"bad": "shape"}}]})


def test_pmc_accessions_are_not_pubmed_accessions():
    assert normalize_pmids(["PMC7654321", "PMID:123", "123", "456"]) == ["123", "456"]


def test_offset_only_patch_refreshes_text(client):
    project = client.post("/api/projects", json={"name": "offset-patch"}).json()
    document = client.post(
        "/api/documents", json={"project_id": project["id"], "text": "Alpha beta."}
    ).json()
    annotation = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Term",
            "start_offset": 0,
            "end_offset": 5,
            "text_span": "Alpha",
        },
    ).json()
    response = client.patch(
        f"/api/annotations/{annotation['id']}", json={"start_offset": 6, "end_offset": 10}
    )
    assert response.status_code == 200, response.text
    assert response.json()["text_span"] == "beta"


def test_snapshot_only_builtin_export_is_validation_error(client, export_source):
    project_id, payload = export_source
    response = client.post(
        f"/api/projects/{project_id}/exports", json={**payload, "format_key": "evidence-blocks-v1"}
    )
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("value", ["bad", None, -1])
def test_training_window_bad_options_are_validation_errors(value):
    from al_medlit.core.exceptions import ValidationError
    from al_medlit.export.formats.evidence_block import TrainingWindowsJSONLFormat
    from al_medlit.export.registry import ExportContext

    with pytest.raises(ValidationError, match="options"):
        list(
            TrainingWindowsJSONLFormat().iter_rows(
                ExportContext(
                    db=None,
                    project_id=1,
                    corpus_snapshot=None,
                    annotation_set=SimpleNamespace(),
                    options={"max_tokens": value},
                )
            )
        )


def test_import_without_available_text_matches_preview(client, monkeypatch):
    from al_medlit.importers import service

    project = client.post("/api/projects", json={"name": "No text"}).json()
    metadata = {"123": pubmed.PubMedMeta(pmid="123", title="No text")}
    monkeypatch.setattr(service, "_gather", lambda *_: (metadata, {}, {}))
    preview = client.post(
        f"/api/projects/{project['id']}/import/pubmed/preview", json={"pmids": ["123"]}
    )
    imported = client.post(
        f"/api/projects/{project['id']}/import/pubmed",
        json={"pmids": ["123"], "include_abstract_only": True},
    )
    assert preview.status_code == imported.status_code == 200
    assert (
        preview.json()["items"][0]["status"] == imported.json()["skipped"][0]["status"] == "error"
    )


def test_missing_export_object_is_not_found(client, export_source, object_storage, db):
    from al_medlit.lineage.models import ExportArtifact

    project_id, payload = export_source
    exported = client.post(f"/api/projects/{project_id}/exports", json=payload).json()
    record = db.get(ExportArtifact, exported["id"])
    object_storage.delete(record.artifact.storage_key)
    response = client.get(f"/api/exports/{record.id}/download")
    assert response.status_code == 404, response.text


def test_guideline_impact_rejects_unsupported_protected_map(client, db, pilot):
    from test_workflow import _headers

    from al_medlit.workflow import models

    split = models.SplitMap(
        project_id=pilot.project.id,
        dataset_version_id=pilot.round.dataset_version_id,
        name="protected",
        strategy="manual",
        seed=42,
        assignments={},
        protected_splits=["test"],
        content_hash="a" * 64,
    )
    db.add(split)
    db.commit()
    response = client.post(
        "/api/workflow-guidelines/impact-evaluations",
        json={
            "project_id": pilot.project.id,
            "guideline_revision_id": pilot.revision.id,
            "pilot_round_id": pilot.round.id,
            "protected_split_map_id": split.id,
        },
        headers=_headers(pilot.manager),
    )
    assert response.status_code == 422, response.text


def test_review_protects_copied_input_with_different_metadata(client, db):
    from test_legacy_training_holdouts import _prepared_source
    from test_training_preparation import _source

    from al_medlit.workflow import models
    from al_medlit.workflow.services.holdouts import (
        protected_dataset_item_ids,
        source_split_constraints,
    )

    project, task, source, _, prepared, _, assignments = _prepared_source(client, db)
    held = (
        db.query(models.DatasetItem)
        .filter_by(
            dataset_version_id=source["id"],
            stable_key=next(key for key, split in assignments.items() if split == "test"),
        )
        .one()
    )
    copied = _source(
        client,
        project,
        "Copied inputs",
        [{"key": "different", "text": held.payload["text"], "metadata": "changed", "label": "yes"}],
    )
    version = db.get(models.DatasetVersion, copied["id"])
    item = db.query(models.DatasetItem).filter_by(dataset_version_id=version.id).one()
    assert source_split_constraints(
        db, version, [item], task["id"], input_mapping={"text": "text"}
    ) == {item.stable_key: {"test"}}
    assert protected_dataset_item_ids(db, [item], task["id"]) == {item.id}


def test_new_version_of_task_keeps_prepared_holdouts(client, db):
    from test_legacy_training_holdouts import _prepared_source

    from al_medlit.workflow import models
    from al_medlit.workflow.services.holdouts import protected_dataset_item_ids

    project, task, source, _, _, _, assignments = _prepared_source(client, db)
    second = client.post(
        "/api/tasks/versions",
        json={
            "project_id": project["id"],
            "task_definition_id": task["task_definition_id"],
            "task_kind": task["task_kind"],
            "input_schema": task["input_schema"],
            "output_schema": {"type": "string", "enum": ["yes", "no", "other"]},
        },
    )
    assert second.status_code == 201, second.text
    items = db.query(models.DatasetItem).filter_by(dataset_version_id=source["id"]).all()
    expected = {item.id for item in items if assignments[item.stable_key] == "test"}
    assert protected_dataset_item_ids(db, items, second.json()["id"]) == expected


def test_source_progress_excludes_cancelled_and_adjudication_and_scopes_task(
    db, workflow_scope, workflow_data
):
    from sqlalchemy import update
    from test_source_progress import _decision, _round, _submit

    from al_medlit.workflow import models, schemas, service
    from al_medlit.workflow.services.source_progress import annotation_progress

    scope, data = workflow_scope, workflow_data
    annotation_round, items = _round(db, scope, data, "Cancelled work", scope.annotator)
    decision = _decision(db, scope, items[0], scope.annotator)
    _submit(db, scope, annotation_round, decision, scope.annotator)
    service.transition_annotation_round(db, scope.project.id, annotation_round.id, "cancelled")
    assert annotation_progress(db, scope.project.id)[0]["submitted"] == 0
    live, items = _round(db, scope, data, "Adjudication", scope.annotator)
    decision = _decision(db, scope, items[0], scope.annotator)
    _submit(db, scope, live, decision, scope.annotator)
    db.execute(
        update(models.RoundAnnotationDecision)
        .where(models.RoundAnnotationDecision.id == decision.id)
        .values(decision_kind="adjudication")
    )
    db.commit()
    assert annotation_progress(db, scope.project.id)[0]["submitted"] == 0
    db.execute(
        update(models.RoundAnnotationDecision)
        .where(models.RoundAnnotationDecision.id == decision.id)
        .values(decision_kind="annotation")
    )
    db.commit()
    assert annotation_progress(db, scope.project.id, data.task_version.id)[0]["submitted"] == 1
    other = service.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=scope.project.id,
            task_definition_id=data.task.id,
            task_kind="classification",
            input_schema={"text": "string"},
            output_schema={"label": ["new"]},
        ),
        scope.manager,
    )
    assert annotation_progress(db, scope.project.id, other.id)[0]["submitted"] == 0


def test_co_learning_uses_authenticated_actor_and_bounds_legacy_fields(client):
    project = client.post("/api/projects", json={"name": "Actor provenance"}).json()
    actor = client.get("/api/auth/me").json()["user"]["username"]
    pattern = client.post(
        "/api/co-learning/error-guideline/patterns",
        json={
            "project_id": project["id"],
            "task_type": "entity",
            "error_type": "boundary_error",
            "description": "Boundary mismatch",
        },
    )
    assert pattern.status_code == 200, pattern.text
    url = f"/api/co-learning/error-guideline/patterns/{pattern.json()['id']}/draft-guideline-atom"
    atom = client.post(url, json={"manager_id": "someone-else"})
    assert atom.status_code == 200, atom.text
    assert atom.json()["approved_by"] == actor
    approved = client.post(
        f"/api/co-learning/error-guideline/guideline-atoms/{atom.json()['id']}/approve",
        json={"approved_by": "spoofed"},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["approved_by"] == actor
    assert client.post(url, json={"manager_id": "x" * 121}).status_code == 422
    assert (
        client.post(
            f"/api/co-learning/error-guideline/guideline-atoms/{atom.json()['id']}/approve",
            json={"approved_by": "x" * 121},
        ).status_code
        == 422
    )


def test_correction_without_replacement_uses_original_task_and_phenotype(client):
    project = client.post("/api/projects", json={"name": "Correction phenotype"}).json()
    document = client.post(
        "/api/documents", json={"project_id": project["id"], "text": "Alpha beta."}
    ).json()
    original = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "passage_label",
            "label": "Relevant",
            "start_offset": 0,
            "end_offset": 11,
            "text_span": "Alpha beta.",
        },
    )
    assert original.status_code == 200, original.text
    correction = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "original_annotation_id": original.json()["id"],
        },
    )
    assert correction.status_code == 200, correction.text
    pattern = client.post(
        f"/api/co-learning/error-guideline/patterns/from-correction/{correction.json()['id']}"
    )
    assert pattern.status_code == 200, pattern.text
    assert pattern.json()["task_type"] == "passage_label"
    assert pattern.json()["error_type"] == "false_positive"


def test_equal_start_items_export_in_a_total_order(frozen_scope):
    from al_medlit.evidence.models import EvidenceTarget, EvidenceTargetVersion
    from al_medlit.export.formats.evidence_block import EvidenceBlocksJSONLFormat
    from al_medlit.export.registry import ExportContext
    from al_medlit.lineage.models import AnnotationSet, AnnotationSetItem, LineageArtifact

    scope = frozen_scope
    db = scope.db
    second_target = EvidenceTarget(project=scope.project, task=scope.task, key="harm", name="Harm")
    second_target.versions.append(EvidenceTargetVersion(version_number=1, text="Does it harm?"))
    artifact = LineageArtifact(
        project_id=scope.project.id,
        artifact_type="annotation_set",
        content_hash="a" * 64,
        storage_key="test/annotation-set",
        content_type="application/json",
        size_bytes=1,
    )
    db.add_all([second_target, artifact])
    db.flush()
    annotation_set = AnnotationSet(
        project_id=scope.project.id,
        artifact_id=artifact.id,
        corpus_snapshot_id=scope.snapshot.id,
        name="Tied starts",
        block_count=3,
        reviewed_region_count=0,
    )
    db.add(annotation_set)
    db.flush()
    for target in (second_target.versions[0], scope.target, scope.target):
        sentence = scope.sentences[0]
        db.add(
            AnnotationSetItem(
                annotation_set_id=annotation_set.id,
                document_id=scope.document.id,
                structure_version_id=scope.structure.id,
                target_version_id=target.id,
                start_sentence_id=sentence.id,
                end_sentence_id=sentence.id,
                start_sentence_ordinal=0,
                end_sentence_ordinal=0,
                start_char=sentence.start_offset,
                end_char=sentence.end_offset,
                block_text="Alpha.",
            )
        )
    db.commit()
    expected = sorted(
        (item.target_version_id, item.id) for item in db.query(AnnotationSetItem).all()
    )
    observed = []
    for reverse in ("ON", "OFF"):
        db.execute(__import__("sqlalchemy").text(f"PRAGMA reverse_unordered_selects={reverse}"))
        db.expire(annotation_set, ["items"])
        assert [(item.target_version_id, item.id) for item in annotation_set.items] == expected
        observed.append(
            list(
                EvidenceBlocksJSONLFormat().iter_rows(
                    ExportContext(
                        db=db,
                        project_id=scope.project.id,
                        corpus_snapshot=scope.snapshot,
                        annotation_set=annotation_set,
                    )
                )
            )
        )
    assert observed[0] == observed[1]
