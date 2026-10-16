import json

import pytest

from al_medlit.annotation.models import Annotation
from al_medlit.auth.models import User
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import DocumentSentence
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.evidence import service as evidence_service
from al_medlit.evidence.models import (
    EvidenceBlockAnnotation,
    EvidenceReviewCoverage,
    EvidenceTarget,
    EvidenceTargetVersion,
)
from al_medlit.evidence.schemas import EvidenceAdjudicationCreate
from al_medlit.export import service as export_service
from al_medlit.export.schemas import ExportCreate
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.lineage.schemas import (
    AnnotationSetCreate,
    CorpusSnapshotCreate,
    CorpusSnapshotDocumentCreate,
)
from al_medlit.lineage.service import freeze_annotation_set, freeze_corpus_snapshot
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.workspace import service as workspace_service


@pytest.mark.parametrize(
    ("workspace_kind", "legacy", "expected_source"),
    [
        ("individual", False, "solo_gold"),
        ("team", False, "adjudicated"),
        ("team", True, "adjudicated"),
    ],
)
def test_frozen_gold_export_preserves_adjudication_provenance(
    db, object_storage, workspace_kind, legacy, expected_source
):
    users = [User(username=f"reviewer-{index}", password_hash="test") for index in range(2)]
    db.add_all(users)
    db.flush()
    if workspace_kind == "individual":
        workspace = workspace_service.create_personal_workspace(db, users[0])
    else:
        workspace = workspace_service.create_team_workspace(db, users[0], "team")
    project = Project(name="provenance", workspace_id=workspace.id)
    task = ProjectTask(project=project, annotation_type="evidence_block", display_name="Evidence")
    guideline = GuidelineVersion(project=project, version_label="v1", markdown="Find evidence.")
    db.add_all([project, task, guideline])
    db.flush()
    target = EvidenceTarget(project_id=project.id, task_id=task.id, key="benefit", name="Benefit")
    target_version = EvidenceTargetVersion(version_number=1, text="Does treatment help?")
    target.versions.append(target_version)
    db.add(target)
    db.commit()
    document = corpus_service.create_document(
        db, DocumentCreate(project_id=project.id, text="Treatment helps.")
    )
    structure = document.active_structure_version
    sentence = db.query(DocumentSentence).filter_by(structure_version_id=structure.id).one()
    sources = []
    reviewers = users[:1] if workspace_kind == "individual" else users
    for reviewer in reviewers:
        source = Annotation(
            project_id=project.id,
            document_id=document.id,
            annotation_type="evidence_block",
            label="evidence_block",
            start_offset=sentence.start_offset,
            end_offset=sentence.end_offset,
            text_span=document.text,
            source="human",
            status="draft",
            annotator_user_id=reviewer.id,
            annotator_id=reviewer.username,
            guideline_version_id=guideline.id,
            structure_version_id=structure.id,
        )
        source.evidence_block = EvidenceBlockAnnotation(
            structure_version_id=structure.id,
            target_version_id=target_version.id,
            start_sentence_id=sentence.id,
            end_sentence_id=sentence.id,
            start_sentence_ordinal=0,
            end_sentence_ordinal=0,
            labels=["supporting"],
        )
        db.add_all(
            [
                source,
                EvidenceReviewCoverage(
                    project_id=project.id,
                    document_id=document.id,
                    structure_version_id=structure.id,
                    target_version_id=target_version.id,
                    guideline_version_id=guideline.id,
                    reviewer_user_id=reviewer.id,
                    start_sentence_id=sentence.id,
                    end_sentence_id=sentence.id,
                    start_sentence_ordinal=0,
                    end_sentence_ordinal=0,
                ),
                TaskAssignment(
                    project_id=project.id,
                    task_id=task.id,
                    document_id=document.id,
                    structure_version_id=structure.id,
                    target_version_id=target_version.id,
                    guideline_version_id=guideline.id,
                    assignee_user_id=reviewer.id,
                    annotator_id=reviewer.username,
                    assignment_scope_key=f"target:{target_version.id}",
                    status="submitted",
                ),
            ]
        )
        sources.append(source)
    db.commit()
    gold = evidence_service.adjudicate(
        db,
        project_id=project.id,
        document_id=document.id,
        actor_user_id=users[0].id,
        actor_username=users[0].username,
        data=EvidenceAdjudicationCreate(
            target_version_id=target_version.id,
            structure_version_id=structure.id,
            guideline_version_id=guideline.id,
            strategy="union",
            source_annotation_ids=[source.id for source in sources],
            solo_gold=workspace_kind == "individual",
        ),
    )
    assert gold.attributes["adjudication"]["source_annotation_ids"]
    if legacy:
        adjudication = dict(gold.attributes["adjudication"])
        adjudication.pop("solo_gold")
        gold.attributes = {"adjudication": adjudication}
        db.commit()
    snapshot = freeze_corpus_snapshot(
        db,
        object_storage,
        project_id=project.id,
        actor_user_id=users[0].id,
        data=CorpusSnapshotCreate(
            name="snapshot",
            documents=[
                CorpusSnapshotDocumentCreate(
                    document_id=document.id,
                    structure_version_id=structure.id,
                    source_hash=structure.source_hash,
                    group_key="document",
                )
            ],
        ),
    )
    annotation_set = freeze_annotation_set(
        db,
        object_storage,
        project_id=project.id,
        actor_user_id=users[0].id,
        data=AnnotationSetCreate(
            name="gold",
            corpus_snapshot_id=snapshot.id,
            target_version_ids=[target_version.id],
            guideline_version_ids=[guideline.id],
        ),
    )
    exported = export_service.create_export(
        db,
        object_storage,
        project_id=project.id,
        actor_user_id=users[0].id,
        data=ExportCreate(format_key="evidence-blocks-v1", annotation_set_id=annotation_set.id),
    )
    rows = [
        json.loads(row)
        for row in object_storage.get_bytes(exported.artifact.storage_key).splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["provenance"]["annotation_id"] == gold.id
    assert rows[0]["provenance"]["source"] == expected_source
    assert annotation_set.items[0].source == expected_source
    manifest = json.loads(object_storage.get_bytes(annotation_set.artifact.storage_key))
    assert manifest["items"][0]["source"] == expected_source
