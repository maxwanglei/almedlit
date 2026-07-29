import hashlib

import pytest
from sqlalchemy.orm import Session

from al_medlit.annotation.models import Annotation
from al_medlit.auth.models import User
from al_medlit.core.exceptions import ConflictError, ValidationError
from al_medlit.core.storage import LocalObjectStorage
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import Document, DocumentSentence, DocumentStructureVersion
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.evidence.models import (
    EvidenceBlockAnnotation,
    EvidenceReviewCoverage,
    EvidenceTarget,
    EvidenceTargetVersion,
)
from al_medlit.export.formats.evidence_block import TrainingWindowsJSONLFormat
from al_medlit.export.registry import ExportContext
from al_medlit.lineage.models import ImmutableRecordError, LineageArtifact
from al_medlit.lineage.schemas import (
    AnnotationSetCreate,
    AnnotationSetReviewRegionCreate,
    CorpusSnapshotCreate,
    CorpusSnapshotDocumentCreate,
)
from al_medlit.lineage.service import (
    add_lineage_edge,
    freeze_annotation_set,
    freeze_corpus_snapshot,
)
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.workspace import service as workspace_service


def _project(db: Session, name: str) -> Project:
    workspace = workspace_service.ensure_default_workspace(db)
    return Project(name=name, workspace_id=workspace.id)


def test_corpus_snapshot_is_idempotent_checksummed_and_immutable(test_engine, tmp_path):
    text = "A sentence."
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    with Session(test_engine) as db:
        project = _project(db, "lineage-project")
        db.add(project)
        db.flush()
        document = Document(project_id=project.id, text=text)
        db.add(document)
        db.flush()
        structure = DocumentStructureVersion(
            document_id=document.id,
            version=1,
            segmenter_name="test",
            segmenter_version="1",
            source_hash=source_hash,
            text_length=len(text),
            status="ready",
        )
        db.add(structure)
        db.commit()
        payload = CorpusSnapshotCreate(
            name="frozen-v1",
            documents=[
                CorpusSnapshotDocumentCreate(
                    document_id=document.id,
                    structure_version_id=structure.id,
                    split="test",
                    group_key=f"document-{document.id}",
                    source_hash=source_hash,
                )
            ],
        )
        storage = LocalObjectStorage(tmp_path)
        first = freeze_corpus_snapshot(
            db,
            storage,
            project_id=project.id,
            data=payload,
            actor_user_id=None,
        )
        second = freeze_corpus_snapshot(
            db,
            storage,
            project_id=project.id,
            data=payload,
            actor_user_id=None,
        )
        assert second.id == first.id
        assert first.documents[0].split == "train"
        assert first.artifact.content_hash == hashlib.sha256(
            storage.get_bytes(first.artifact.storage_key)
        ).hexdigest()

        first.name = "rewritten"
        with pytest.raises(ImmutableRecordError):
            db.commit()
        db.rollback()


def test_lineage_edges_reject_cycles(test_engine):
    with Session(test_engine) as db:
        project = _project(db, "lineage-cycle-project")
        db.add(project)
        db.flush()
        artifacts = [
            LineageArtifact(
                project_id=project.id,
                artifact_type=f"type-{index}",
                content_hash=str(index) * 64,
                storage_key=f"object-{index}",
                content_type="application/json",
                size_bytes=1,
            )
            for index in (1, 2, 3)
        ]
        db.add_all(artifacts)
        db.flush()
        add_lineage_edge(
            db,
            upstream_artifact_id=artifacts[0].id,
            downstream_artifact_id=artifacts[1].id,
        )
        add_lineage_edge(
            db,
            upstream_artifact_id=artifacts[1].id,
            downstream_artifact_id=artifacts[2].id,
        )
        with pytest.raises(ConflictError, match="cycle"):
            add_lineage_edge(
                db,
                upstream_artifact_id=artifacts[2].id,
                downstream_artifact_id=artifacts[0].id,
            )


def test_annotation_freeze_derives_submitted_coverage_and_explicit_gold(
    test_engine,
    tmp_path,
):
    with Session(test_engine) as db:
        project = _project(db, "server-derived-freeze")
        task = ProjectTask(
            project=project,
            annotation_type="evidence_block",
            display_name="Evidence blocks",
        )
        users = [
            User(username=name, password_hash="test", display_name=name)
            for name in ("alice-freeze", "bob-freeze")
        ]
        db.add_all([project, task, *users])
        db.flush()
        target = EvidenceTarget(
            project_id=project.id,
            task_id=task.id,
            key="benefit",
            name="Benefit",
            is_active=True,
        )
        target_version = EvidenceTargetVersion(
            version_number=1,
            text="Does the treatment help?",
        )
        target.versions.append(target_version)
        unreviewed_target = EvidenceTarget(
            project_id=project.id,
            task_id=task.id,
            key="harm",
            name="Harm",
            is_active=True,
        )
        unreviewed_target_version = EvidenceTargetVersion(
            version_number=1,
            text="Does the treatment cause harm?",
        )
        unreviewed_target.versions.append(unreviewed_target_version)
        db.add_all([target, unreviewed_target])
        db.commit()

        document = corpus_service.create_document(
            db,
            DocumentCreate(project_id=project.id, text="Alpha. Beta."),
        )
        structure = document.active_structure_version
        sentences = (
            db.query(DocumentSentence)
            .filter(DocumentSentence.structure_version_id == structure.id)
            .order_by(DocumentSentence.ordinal)
            .all()
        )
        assert len(sentences) == 2

        for user in users:
            db.add(
                TaskAssignment(
                    project_id=project.id,
                    task_id=task.id,
                    document_id=document.id,
                    assignee_user_id=user.id,
                    target_version_id=target_version.id,
                    structure_version_id=structure.id,
                    assignment_scope_key=f"target:{target_version.id}",
                    annotator_id=user.username,
                    status="submitted",
                )
            )
        # Only sentence 1 is reviewed by every submitted annotator.
        db.add_all(
            [
                EvidenceReviewCoverage(
                    project_id=project.id,
                    document_id=document.id,
                    structure_version_id=structure.id,
                    target_version_id=target_version.id,
                    reviewer_user_id=users[0].id,
                    start_sentence_id=sentences[0].id,
                    end_sentence_id=sentences[1].id,
                    start_sentence_ordinal=0,
                    end_sentence_ordinal=1,
                ),
                EvidenceReviewCoverage(
                    project_id=project.id,
                    document_id=document.id,
                    structure_version_id=structure.id,
                    target_version_id=target_version.id,
                    reviewer_user_id=users[1].id,
                    start_sentence_id=sentences[1].id,
                    end_sentence_id=sentences[1].id,
                    start_sentence_ordinal=1,
                    end_sentence_ordinal=1,
                ),
            ]
        )
        gold = Annotation(
            project_id=project.id,
            document_id=document.id,
            annotation_type="evidence_block",
            label="evidence_block",
            start_offset=sentences[1].start_offset,
            end_offset=sentences[1].end_offset,
            text_span="Beta.",
            source="human",
            status="gold",
            annotator_user_id=users[0].id,
            annotator_id=users[0].username,
            attributes={
                "adjudication": {
                    "strategy": "a",
                    "source_annotation_ids": [101],
                }
            },
        )
        gold.evidence_block = EvidenceBlockAnnotation(
            structure_version_id=structure.id,
            target_version_id=target_version.id,
            start_sentence_id=sentences[1].id,
            end_sentence_id=sentences[1].id,
            start_sentence_ordinal=1,
            end_sentence_ordinal=1,
            labels=["supporting"],
            locked=True,
        )
        legacy_gold = Annotation(
            project_id=project.id,
            document_id=document.id,
            annotation_type="evidence_block",
            label="evidence_block",
            start_offset=sentences[0].start_offset,
            end_offset=sentences[0].end_offset,
            text_span="Alpha.",
            source="human",
            status="gold",
            annotator_user_id=users[0].id,
            annotator_id=users[0].username,
            attributes={},
        )
        legacy_gold.evidence_block = EvidenceBlockAnnotation(
            structure_version_id=structure.id,
            target_version_id=target_version.id,
            start_sentence_id=sentences[0].id,
            end_sentence_id=sentences[0].id,
            start_sentence_ordinal=0,
            end_sentence_ordinal=0,
            locked=True,
        )
        db.add_all([gold, legacy_gold])
        db.commit()

        storage = LocalObjectStorage(tmp_path)
        snapshot = freeze_corpus_snapshot(
            db,
            storage,
            project_id=project.id,
            data=CorpusSnapshotCreate(
                name="snapshot",
                documents=[
                    CorpusSnapshotDocumentCreate(
                        document_id=document.id,
                        structure_version_id=structure.id,
                        group_key=f"document-{document.id}",
                        source_hash=structure.source_hash,
                    )
                ],
            ),
            actor_user_id=users[0].id,
        )
        annotation_set = freeze_annotation_set(
            db,
            storage,
            project_id=project.id,
            data=AnnotationSetCreate(
                name="gold-v1",
                corpus_snapshot_id=snapshot.id,
                target_version_ids=[target_version.id, unreviewed_target_version.id],
            ),
            actor_user_id=users[0].id,
        )

        assert annotation_set.block_count == 1
        assert annotation_set.reviewed_region_count == 1
        assert annotation_set.items[0].source_annotation_id == gold.id
        assert annotation_set.items[0].block_text == "Beta."
        assert annotation_set.items[0].labels == ["supporting"]
        assert annotation_set.reviewed_regions[0].start_sentence_ordinal == 1
        assert annotation_set.reviewed_regions[0].end_sentence_ordinal == 1
        training_rows = list(
            TrainingWindowsJSONLFormat().iter_rows(
                ExportContext(
                    db=db,
                    project_id=project.id,
                    corpus_snapshot=snapshot,
                    annotation_set=annotation_set,
                )
            )
        )
        assert training_rows
        assert {row["target"]["id"] for row in training_rows} == {
            target_version.id
        }
        assert all(
            any(sentence["label"] != "IGNORE" for sentence in row["sentences"])
            for row in training_rows
        )

        with pytest.raises(ValidationError, match="server-derived"):
            freeze_annotation_set(
                db,
                storage,
                project_id=project.id,
                data=AnnotationSetCreate(
                    name="forged",
                    corpus_snapshot_id=snapshot.id,
                    target_version_ids=[target_version.id],
                    reviewed_regions=[
                        AnnotationSetReviewRegionCreate(
                            document_id=document.id,
                            structure_version_id=structure.id,
                            target_version_id=target_version.id,
                            start_sentence_ordinal=0,
                            end_sentence_ordinal=1,
                        )
                    ],
                ),
                actor_user_id=users[0].id,
            )
