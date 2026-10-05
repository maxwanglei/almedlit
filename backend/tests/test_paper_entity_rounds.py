"""The paper editor retains task-specific decisions against immutable source text."""

from types import SimpleNamespace

import pytest
from test_workflow import workflow_scope as workflow_scope

from al_medlit.annotation.models import Annotation
from al_medlit.core.exceptions import ValidationError
from al_medlit.corpus.models import Document
from al_medlit.project.models import Project
from al_medlit.workflow import models, schemas, service
from al_medlit.workflow.services.datasets import create_project_corpus_dataset_version


@pytest.fixture
def paper_tasks(db, workflow_scope):
    scope = workflow_scope
    document = Document(
        project_id=scope.project.id, title="Drug response", text="Drug 🧬 improved.",
        source="test", metadata_={},
    )
    other_document = Document(
        project_id=scope.project.id, title="Another paper", text="No entities here.",
        source="test", metadata_={},
    )
    db.add_all([document, other_document])
    db.commit()
    source = service.create_dataset(db, schemas.DatasetCreate(
        project_id=scope.project.id, name="Shared papers", source_type="project_corpus",
    ), scope.manager)
    version = create_project_corpus_dataset_version(
        db, project_id=scope.project.id, dataset_id=source.id, actor=scope.manager,
        document_ids=[document.id, other_document.id],
    )
    classification = service.create_task_definition(db, schemas.TaskDefinitionCreate(
        project_id=scope.project.id, name="Relevance", key="relevance",
    ), scope.manager)
    classification_version = service.create_task_version(db, schemas.TaskVersionCreate(
        project_id=scope.project.id, task_definition_id=classification.id,
        task_kind="classification",
        input_schema={"type": "object", "required": ["text"], "properties": {
            "text": {"type": "string"},
        }},
        output_schema={"type": "string", "enum": ["relevant", "irrelevant"]},
    ), scope.manager)
    entities = service.create_task_definition(db, schemas.TaskDefinitionCreate(
        project_id=scope.project.id, name="Entities", key="entities",
    ), scope.manager)
    entity_version = service.create_task_version(db, schemas.TaskVersionCreate(
        project_id=scope.project.id, task_definition_id=entities.id,
        task_kind="span_extraction",
        input_schema={"type": "object", "required": ["text"], "properties": {
            "text": {"type": "string"},
        }},
        output_schema={"type": "object", "required": ["entities"], "properties": {
            "entities": {"type": "array", "items": {
                "type": "object", "required": ["start", "end", "label"], "properties": {
                    "start": {"type": "integer"}, "end": {"type": "integer"},
                    "label": {"type": "string"},
                },
            }},
        }},
        annotation_ui={
            "preset": "document_entities", "offset_unit": "utf16_code_unit",
            "end_offset": "exclusive",
        },
        label_rules={"values": ["Drug", "Gene"], "closed_set": True},
    ), scope.manager)
    return SimpleNamespace(
        dataset=source, version=version, document=document,
        classification=classification_version, entities=entity_version,
    )


def _round(db, scope, source_version, task):
    annotation_round = service.create_annotation_round(db, schemas.AnnotationRoundCreate(
        project_id=scope.project.id, name=f"Task {task.id}",
        dataset_version_id=source_version.id, task_version_id=task.id,
        assistance_policy="blind", reannotation_mode="full_dataset",
        annotator_user_ids=[scope.annotator.id],
    ), scope.manager)
    service.transition_annotation_round(db, scope.project.id, annotation_round.id, "open")
    return annotation_round, service.list_round_items(db, scope.project.id, annotation_round.id)


def _decision(db, scope, item, output, prior=None):
    return service.create_annotation_decision(db, schemas.AnnotationDecisionCreate(
        project_id=scope.project.id, round_item_id=item.id, output=output,
        supersedes_decision_id=prior.id if prior else None,
    ), scope.annotator)


def _publish(db, scope, annotation_round, decision):
    submission = service.create_round_submission(db, schemas.RoundSubmissionCreate(
        project_id=scope.project.id, annotation_round_id=annotation_round.id,
        decision_ids=[decision.id],
    ), scope.annotator)
    return service.create_round_label_set(db, annotation_round.id, schemas.RoundLabelSetCreate(
        project_id=scope.project.id, name="Submitted subset", source_kind="human",
        submission_ids=[submission.id], publication_mode="submitted_snapshot",
    ), scope.manager)


def test_classification_and_entities_preserve_independent_submitted_subsets(
    db, workflow_scope, paper_tasks,
):
    scope, papers = workflow_scope, paper_tasks
    classification_round, class_items = _round(db, scope, papers.version, papers.classification)
    entity_round, entity_items = _round(db, scope, papers.version, papers.entities)
    assert [item.dataset_item_id for item in class_items] == [
        item.dataset_item_id for item in entity_items
    ]
    classification = _decision(db, scope, class_items[0], "relevant")
    # The emoji consumes two UTF-16 code units. These spans may overlap.
    output = {"entities": [
        {"start": 0, "end": 4, "label": "Drug"},
        {"start": 0, "end": 2, "label": "Drug"},
        {"start": 5, "end": 7, "label": "Gene"},
    ]}
    entities = _decision(db, scope, entity_items[0], output)
    class_snapshot = _publish(db, scope, classification_round, classification)
    entity_snapshot = _publish(db, scope, entity_round, entities)
    assert class_snapshot.label_count == entity_snapshot.label_count == 1
    assert class_snapshot.task_version_id == papers.classification.id
    assert entity_snapshot.task_version_id == papers.entities.id
    assert (
        class_snapshot.dataset_version_id == entity_snapshot.dataset_version_id == papers.version.id
    )

    _decision(db, scope, class_items[1], "irrelevant")  # still an unsubmitted draft
    revised = _decision(db, scope, entity_items[0], {"entities": []}, entities)
    later_snapshot = _publish(db, scope, entity_round, revised)
    assert list(later_snapshot.labels.values()) == [{"entities": []}]
    assert list(entity_snapshot.labels.values()) == [output]
    assert list(class_snapshot.labels.values()) == ["relevant"]
    assert class_snapshot.source_decision_ids == [classification.id]
    assert entity_snapshot.source_decision_ids == [entities.id]
    assert classification_round.status == entity_round.status == "open"
    assert db.query(Annotation).count() == 0
    assert db.query(models.DatasetVersion).count() == 1


@pytest.mark.parametrize(("entity", "message"), [
    ({"start": -1, "end": 2, "label": "Drug"}, "pinned paper text"),
    ({"start": 0, "end": 100, "label": "Drug"}, "pinned paper text"),
    ({"start": 6, "end": 7, "label": "Gene"}, "Unicode character"),
    ({"start": 5, "end": 6, "label": "Gene"}, "Unicode character"),
    ({"start": 2, "end": 2, "label": "Drug"}, "start < end"),
    ({"start": 0, "end": 4, "label": "Unknown"}, "not allowed"),
])
def test_entity_offsets_and_labels_are_validated_against_the_pinned_source(
    db, workflow_scope, paper_tasks, entity, message,
):
    _, items = _round(db, workflow_scope, paper_tasks.version, paper_tasks.entities)
    with pytest.raises(ValidationError, match=message):
        _decision(db, workflow_scope, items[0], {"entities": [entity]})
    assert db.query(models.RoundAnnotationDecision).count() == 0


def test_entity_validation_uses_snapshot_text_when_live_document_changes(
    db, workflow_scope, paper_tasks,
):
    papers = paper_tasks
    papers.document.text = "Changed"
    db.commit()
    _, items = _round(db, workflow_scope, papers.version, papers.entities)
    decision = _decision(db, workflow_scope, items[0], {
        "entities": [{"start": 8, "end": 16, "label": "Drug"}],
    })
    assert decision.output["entities"][0]["end"] == 16


@pytest.mark.parametrize("payload", [{"tokens": ["Drug"]}, {"text": ""}])
def test_entity_round_rejects_non_text_sources_before_creating_any_round(
    db, workflow_scope, paper_tasks, payload,
):
    scope = workflow_scope
    invalid = service.create_dataset_version(db, schemas.DatasetVersionCreate(
        project_id=scope.project.id, dataset_id=paper_tasks.dataset.id,
        source_revision="no-text", source_format="jsonl",
        items=[schemas.DatasetItemCreate(stable_key="bad", payload=payload)],
    ), scope.manager)
    with pytest.raises(ValidationError, match="no paper text"):
        _round(db, scope, invalid, paper_tasks.entities)
    assert db.query(models.AnnotationRound).count() == 0


def test_entity_round_rejects_cross_project_document_reference(db, workflow_scope, paper_tasks):
    scope = workflow_scope
    foreign_project = Project(name="Other papers", workspace_id=scope.workspace.id)
    db.add(foreign_project)
    db.flush()
    foreign = Document(
        project_id=foreign_project.id, title="Private", text="Private paper", source="test",
    )
    db.add(foreign)
    db.commit()
    forged = service.create_dataset_version(db, schemas.DatasetVersionCreate(
        project_id=scope.project.id, dataset_id=paper_tasks.dataset.id,
        source_revision="wrong-project", source_format="jsonl",
        items=[schemas.DatasetItemCreate(stable_key="foreign", payload={
            "text": "Private paper", "document_id": foreign.id,
        })],
    ), scope.manager)
    with pytest.raises(ValidationError, match="must belong to this project"):
        _round(db, scope, forged, paper_tasks.entities)
    assert db.query(models.AnnotationRound).count() == 0


@pytest.mark.parametrize(("field", "value", "message"), [
    ("annotation_ui", {"preset": "document_entities"}, "must specify UTF-16"),
    ("label_rules", {"values": None}, "list of nonempty names"),
])
def test_entity_round_requires_an_explicit_valid_editor_contract(
    db, workflow_scope, paper_tasks, field, value, message,
):
    task = paper_tasks.entities
    fields = {
        name: getattr(task, name)
        for name in (
            "project_id", "task_definition_id", "task_kind", "input_schema",
            "output_schema", "annotation_ui", "label_rules",
        )
    }
    fields[field] = value
    incompatible = service.create_task_version(
        db, schemas.TaskVersionCreate(**fields), workflow_scope.manager,
    )
    with pytest.raises(ValidationError, match=message):
        _round(db, workflow_scope, paper_tasks.version, incompatible)
    assert db.query(models.AnnotationRound).count() == 0
