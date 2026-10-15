from types import SimpleNamespace

import pytest
from sqlalchemy.orm import Session

from al_medlit.annotation.models import Annotation
from al_medlit.auth.models import User
from al_medlit.core.exceptions import ValidationError
from al_medlit.core.storage import LocalObjectStorage
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import DocumentSentence
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.evidence.models import (
    EvidenceBlockAnnotation,
    EvidenceReviewCoverage,
    EvidenceTarget,
    EvidenceTargetVersion,
)
from al_medlit.export.formats.evidence_block import TrainingWindowsJSONLFormat
from al_medlit.export.registry import ExportContext
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.lineage.schemas import (
    AnnotationSetCreate,
    CorpusSnapshotCreate,
    CorpusSnapshotDocumentCreate,
)
from al_medlit.lineage.service import freeze_annotation_set, freeze_corpus_snapshot
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.workspace import service as workspace_service


@pytest.fixture
def frozen_scope(test_engine, tmp_path):
    with Session(test_engine) as db:
        workspace = workspace_service.ensure_default_workspace(db)
        project = Project(name="coverage-freeze", workspace_id=workspace.id)
        task = ProjectTask(
            project=project, annotation_type="evidence_block", display_name="Evidence"
        )
        users = [
            User(username=name, password_hash="test", display_name=name)
            for name in ("alice", "bob")
        ]
        guidelines = [
            GuidelineVersion(project=project, version_label=label) for label in ("v1", "v2")
        ]
        target = EvidenceTarget(project=project, task=task, key="benefit", name="Benefit")
        target_version = EvidenceTargetVersion(version_number=1, text="Does it help?")
        target.versions.append(target_version)
        db.add_all([project, task, *users, *guidelines, target])
        db.commit()
        document = corpus_service.create_document(
            db, DocumentCreate(project_id=project.id, text="Alpha. Beta. Gamma.")
        )
        structure = document.active_structure_version
        sentences = (
            db.query(DocumentSentence)
            .filter(DocumentSentence.structure_version_id == structure.id)
            .order_by(DocumentSentence.ordinal)
            .all()
        )
        assert len(sentences) == 3
        storage = LocalObjectStorage(tmp_path)
        snapshot = freeze_corpus_snapshot(
            db,
            storage,
            project_id=project.id,
            data=CorpusSnapshotCreate(
                name="coverage-corpus",
                documents=[
                    CorpusSnapshotDocumentCreate(
                        document_id=document.id,
                        structure_version_id=structure.id,
                        group_key=str(document.id),
                        source_hash=structure.source_hash,
                    )
                ],
            ),
            actor_user_id=None,
        )
        yield SimpleNamespace(
            db=db,
            project=project,
            task=task,
            users=users,
            guidelines=guidelines,
            target=target_version,
            document=document,
            structure=structure,
            sentences=sentences,
            storage=storage,
            snapshot=snapshot,
        )


def _assignment(scope, guideline_id, *, reviewer=0, status="submitted"):
    assignment = TaskAssignment(
        project_id=scope.project.id,
        task_id=scope.task.id,
        document_id=scope.document.id,
        assignee_user_id=scope.users[reviewer].id,
        annotator_id=scope.users[reviewer].username,
        target_version_id=scope.target.id,
        structure_version_id=scope.structure.id,
        guideline_version_id=guideline_id,
        assignment_scope_key=f"target:{scope.target.id}:guideline:{guideline_id}",
        status=status,
    )
    scope.db.add(assignment)
    return assignment


def _coverage(scope, guideline_id, start, end, *, reviewer=0):
    coverage = EvidenceReviewCoverage(
        project_id=scope.project.id,
        document_id=scope.document.id,
        structure_version_id=scope.structure.id,
        target_version_id=scope.target.id,
        guideline_version_id=guideline_id,
        reviewer_user_id=scope.users[reviewer].id,
        start_sentence_id=scope.sentences[start].id,
        end_sentence_id=scope.sentences[end].id,
        start_sentence_ordinal=start,
        end_sentence_ordinal=end,
    )
    scope.db.add(coverage)
    return coverage


def _gold(scope, guideline_id, ordinal):
    sentence = scope.sentences[ordinal]
    annotation = Annotation(
        project_id=scope.project.id,
        document_id=scope.document.id,
        annotation_type="evidence_block",
        label="evidence_block",
        guideline_version_id=guideline_id,
        start_offset=sentence.start_offset,
        end_offset=sentence.end_offset,
        text_span=scope.document.text[sentence.start_offset : sentence.end_offset],
        status="gold",
        source="human",
        annotator_user_id=scope.users[0].id,
        annotator_id=scope.users[0].username,
        attributes={"adjudication": {"strategy": "a", "source_annotation_ids": []}},
    )
    annotation.evidence_block = EvidenceBlockAnnotation(
        structure_version_id=scope.structure.id,
        target_version_id=scope.target.id,
        start_sentence_id=sentence.id,
        end_sentence_id=sentence.id,
        start_sentence_ordinal=ordinal,
        end_sentence_ordinal=ordinal,
        labels=["supporting"],
        locked=True,
    )
    scope.db.add(annotation)
    scope.db.flush()
    return annotation


def _freeze(scope, guideline_ids=()):
    scope.db.commit()
    return freeze_annotation_set(
        scope.db,
        scope.storage,
        project_id=scope.project.id,
        data=AnnotationSetCreate(
            name="frozen-coverage",
            corpus_snapshot_id=scope.snapshot.id,
            target_version_ids=[scope.target.id],
            guideline_version_ids=list(guideline_ids),
        ),
        actor_user_id=None,
    )


@pytest.mark.parametrize("null_guideline", [False, True])
@pytest.mark.parametrize("open_status", ["assigned", "in_progress"])
def test_freeze_does_not_export_open_round_coverage_as_negatives(
    frozen_scope, null_guideline, open_status
):
    scope = frozen_scope
    frozen_guideline = None if null_guideline else scope.guidelines[0].id
    _assignment(scope, frozen_guideline)
    _coverage(scope, frozen_guideline, 0, 0)
    _assignment(scope, scope.guidelines[1].id, status=open_status)
    _coverage(scope, scope.guidelines[1].id, 1, 2)

    annotation_set = _freeze(scope)
    rows = list(
        TrainingWindowsJSONLFormat().iter_rows(
            ExportContext(
                db=scope.db,
                project_id=scope.project.id,
                corpus_snapshot=scope.snapshot,
                annotation_set=annotation_set,
            )
        )
    )
    assert rows
    assert {
        sentence["ordinal"]: sentence["label"] for row in rows for sentence in row["sentences"]
    } == {0: "O", 1: "IGNORE", 2: "IGNORE"}


def test_freeze_filters_assignments_before_intersecting_reviewers(frozen_scope):
    scope = frozen_scope
    v1, v2 = (guideline.id for guideline in scope.guidelines)
    _assignment(scope, v1)
    _coverage(scope, v1, 0, 0)
    _assignment(scope, v2, reviewer=1)
    _coverage(scope, v2, 1, 2, reviewer=1)

    annotation_set = _freeze(scope, [v1])
    assert [
        (region.start_sentence_ordinal, region.end_sentence_ordinal)
        for region in annotation_set.reviewed_regions
    ] == [(0, 0)]
    assert annotation_set.reviewed_regions[0].metadata_["reviewer_user_ids"] == [scope.users[0].id]


def test_freeze_filters_gold_annotations_by_requested_guideline(frozen_scope):
    scope = frozen_scope
    v1, v2 = (guideline.id for guideline in scope.guidelines)
    _assignment(scope, v1)
    _coverage(scope, v1, 0, 2)
    selected = _gold(scope, v1, 0)
    _gold(scope, v2, 1)
    _gold(scope, None, 2)

    annotation_set = _freeze(scope, [v1])
    assert [item.source_annotation_id for item in annotation_set.items] == [selected.id]
    assert annotation_set.guideline_version_ids == [v1]


def test_freeze_rejects_guideline_without_finalized_assignments(frozen_scope):
    scope = frozen_scope
    v1, v2 = (guideline.id for guideline in scope.guidelines)
    _assignment(scope, v1)
    _coverage(scope, v1, 0, 0)
    _assignment(scope, v2, status="in_progress")
    _coverage(scope, v2, 1, 2)

    with pytest.raises(ValidationError, match="No reviewed evidence coverage"):
        _freeze(scope, [v2])


def test_freeze_requires_scoped_coverage_from_every_finalized_reviewer(frozen_scope):
    scope = frozen_scope
    v1, v2 = (guideline.id for guideline in scope.guidelines)
    _assignment(scope, v1)
    _coverage(scope, v1, 0, 2)
    _assignment(scope, v1, reviewer=1)
    _assignment(scope, v2, reviewer=1, status="in_progress")
    _coverage(scope, v2, 0, 2, reviewer=1)

    with pytest.raises(ValidationError, match="No reviewed evidence coverage"):
        _freeze(scope)


def test_freeze_unions_finalized_rounds_per_reviewer_then_intersects(frozen_scope):
    scope = frozen_scope
    v1, v2 = (guideline.id for guideline in scope.guidelines)
    _assignment(scope, v1)
    _coverage(scope, v1, 0, 0)
    _assignment(scope, v2)
    _coverage(scope, v2, 1, 2)
    _assignment(scope, v1, reviewer=1)
    _coverage(scope, v1, 1, 1, reviewer=1)

    annotation_set = _freeze(scope)
    assert [
        (region.start_sentence_ordinal, region.end_sentence_ordinal)
        for region in annotation_set.reviewed_regions
    ] == [(1, 1)]


def test_equivalent_coverage_has_identical_manifest_and_hash(frozen_scope):
    scope = frozen_scope
    guideline_id = scope.guidelines[0].id
    _assignment(scope, guideline_id)
    _coverage(scope, guideline_id, 0, 2)
    first = _freeze(scope)
    first_hash = first.artifact.content_hash

    scope.db.query(EvidenceReviewCoverage).delete()
    _coverage(scope, guideline_id, 1, 2)
    _coverage(scope, guideline_id, 0, 0)
    _coverage(scope, guideline_id, 0, 1)
    second = _freeze(scope)

    assert second.artifact.content_hash == first_hash
    assert second.id == first.id
    assert [
        (region.start_sentence_ordinal, region.end_sentence_ordinal)
        for region in second.reviewed_regions
    ] == [(0, 2)]
