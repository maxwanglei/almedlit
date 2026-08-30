from dataclasses import dataclass

import pytest
from sqlalchemy import event

from al_medlit.auth.models import User
from al_medlit.core.exceptions import ValidationError
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import DocumentSentence
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.evidence.models import EvidenceTarget, EvidenceTargetVersion
from al_medlit.inference import service as inference_service
from al_medlit.inference.decoder import DecodedBlock, DecodedSentence, DecoderResult
from al_medlit.inference.models import (
    EvidencePredictionReview,
    InferenceRun,
    InferenceWindow,
)
from al_medlit.lineage.models import (
    AnnotationSet,
    CorpusSnapshot,
    CorpusSnapshotDocument,
    LineageArtifact,
)
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.training.models import (
    ComputeProfile,
    ModelCheckpoint,
    TrainingExperiment,
    TrainingJob,
)
from al_medlit.workspace.models import Workspace, WorkspaceMember


@dataclass
class InferenceScope:
    user: User
    project: Project
    target_version: EvidenceTargetVersion
    sentences: list[DocumentSentence]
    run: InferenceRun
    window: InferenceWindow


def _inference_scope(db) -> InferenceScope:
    user = User(
        username="candidate-query-reviewer",
        password_hash="unused",
        display_name="Candidate reviewer",
    )
    db.add(user)
    db.flush()
    workspace = Workspace(
        name="Candidate query workspace",
        kind="team",
        created_by=user.id,
    )
    db.add(workspace)
    db.flush()
    db.add(WorkspaceMember(workspace_id=workspace.id, user_id=user.id, role="annotator"))
    project = Project(
        workspace_id=workspace.id,
        name="Candidate query project",
        annotation_schema={},
        settings={},
    )
    db.add(project)
    db.flush()
    task = ProjectTask(
        project_id=project.id,
        annotation_type="evidence_block",
        display_name="Evidence blocks",
    )
    db.add(task)
    db.flush()
    target = EvidenceTarget(
        project_id=project.id,
        task_id=task.id,
        key="benefit",
        name="Benefit",
        is_active=True,
        created_by_user_id=user.id,
    )
    db.add(target)
    db.flush()
    target_version = EvidenceTargetVersion(
        target_id=target.id,
        version_number=1,
        text="Does the treatment help?",
        created_by_user_id=user.id,
    )
    db.add(target_version)
    db.flush()

    document = corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            text="Alpha improved. Beta improved. Gamma improved. Delta improved.",
        ),
    )
    structure = document.active_structure_version
    sentences = (
        db.query(DocumentSentence)
        .filter(DocumentSentence.structure_version_id == structure.id)
        .order_by(DocumentSentence.ordinal)
        .all()
    )
    assert len(sentences) == 4

    artifacts = [
        LineageArtifact(
            project_id=project.id,
            artifact_type=artifact_type,
            content_hash=content_hash * 64,
            storage_key=f"candidate-query/{artifact_type}.json",
            content_type="application/json",
            size_bytes=1,
            manifest={},
            created_by_user_id=user.id,
        )
        for artifact_type, content_hash in (
            ("corpus_snapshot", "c"),
            ("annotation_set", "a"),
            ("model_checkpoint", "m"),
        )
    ]
    db.add_all(artifacts)
    db.flush()
    snapshot = CorpusSnapshot(
        project_id=project.id,
        artifact_id=artifacts[0].id,
        name="Candidate query snapshot",
        document_count=1,
    )
    db.add(snapshot)
    db.flush()
    db.add(
        CorpusSnapshotDocument(
            snapshot_id=snapshot.id,
            document_id=document.id,
            structure_version_id=structure.id,
            split="inference",
            group_key=str(document.id),
            source_hash="s" * 64,
        )
    )
    annotation_set = AnnotationSet(
        project_id=project.id,
        artifact_id=artifacts[1].id,
        corpus_snapshot_id=snapshot.id,
        name="Candidate query annotations",
        target_version_ids=[target_version.id],
        block_count=0,
        reviewed_region_count=0,
    )
    profile = ComputeProfile(
        project_id=project.id,
        name="candidate-query-local",
        backend="local",
        config={},
        status="active",
        created_by_user_id=user.id,
    )
    db.add_all([annotation_set, profile])
    db.flush()
    experiment = TrainingExperiment(
        project_id=project.id,
        annotation_set_id=annotation_set.id,
        compute_profile_id=profile.id,
        name="Candidate query experiment",
        model_type="evidence_block_sentence_tagger",
        mode="conditioned",
        target_version_ids=[target_version.id],
        config={},
        idempotency_key="candidate-query-experiment",
        created_by_user_id=user.id,
    )
    db.add(experiment)
    db.flush()
    job = TrainingJob(
        experiment_id=experiment.id,
        compute_profile_id=profile.id,
        idempotency_key="candidate-query-job",
        status="succeeded",
    )
    db.add(job)
    db.flush()
    checkpoint = ModelCheckpoint(
        project_id=project.id,
        training_job_id=job.id,
        artifact_id=artifacts[2].id,
        model_type="evidence_block_sentence_tagger",
        training_mode="conditioned",
        trained_target_version_ids=[target_version.id],
        max_context_tokens=512,
        manifest={"synthetic_mode": True},
        readiness="ready",
    )
    db.add(checkpoint)
    db.flush()
    run = InferenceRun(
        project_id=project.id,
        corpus_snapshot_id=snapshot.id,
        checkpoint_id=checkpoint.id,
        compute_profile_id=profile.id,
        name="Candidate query run",
        target_version_ids=[target_version.id],
        window_config={},
        decoder_config={},
        status="succeeded",
        idempotency_key="candidate-query-run",
        created_by_user_id=user.id,
    )
    db.add(run)
    db.flush()
    window = InferenceWindow(
        run_id=run.id,
        document_id=document.id,
        structure_version_id=structure.id,
        target_version_id=target_version.id,
        stable_key="candidate-query-window",
        start_sentence_ordinal=0,
        end_sentence_ordinal=3,
        token_count=12,
        status="pending",
    )
    db.add(window)
    db.commit()
    return InferenceScope(
        user=user,
        project=project,
        target_version=target_version,
        sentences=sentences,
        run=run,
        window=window,
    )


def _decoder_result(sentences: list[DocumentSentence]) -> DecoderResult:
    blocks = tuple(
        DecodedBlock(
            start_sentence_id=sentence.id,
            end_sentence_id=sentence.id,
            start_ordinal=sentence.ordinal,
            end_ordinal=sentence.ordinal,
            start_char=sentence.start_offset,
            end_char=sentence.end_offset,
            confidence=0.9,
            start_confidence=0.9,
            end_confidence=0.9,
            uncertainty=0.1,
            sentence_ordinals=(sentence.ordinal,),
        )
        for sentence in sentences
    )
    decoded_sentences = tuple(
        DecodedSentence(
            id=sentence.id,
            ordinal=sentence.ordinal,
            raw_label="B",
            decoded_label="B",
            probabilities=(0.05, 0.9, 0.05),
            contribution_count=1,
            entropy=0.1,
        )
        for sentence in sentences
    )
    return DecoderResult(
        blocks=blocks,
        suppressed_blocks=(),
        sentences=decoded_sentences,
    )


def _selected_statements(db, operation):
    statements: list[str] = []

    def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    bind = db.get_bind()
    event.listen(bind, "before_cursor_execute", capture)
    try:
        result = operation()
    finally:
        event.remove(bind, "before_cursor_execute", capture)
    return result, statements


def test_persist_decoder_result_bulk_fetches_and_reloads_candidates(db):
    scope = _inference_scope(db)
    result = _decoder_result(scope.sentences)
    run_id = scope.run.id
    document_id = scope.sentences[0].structure_version.document_id
    structure_version_id = scope.sentences[0].structure_version_id
    target_version_id = scope.target_version.id
    window_id = scope.window.id

    candidates, statements = _selected_statements(
        db,
        lambda: inference_service.persist_decoder_result(
            db,
            run_id=run_id,
            document_id=document_id,
            structure_version_id=structure_version_id,
            target_version_id=target_version_id,
            result=result,
            source_window_ids=[window_id],
        ),
    )

    candidate_selects = [
        statement
        for statement in statements
        if "FROM evidence_candidate_predictions" in statement
    ]
    assert len(candidates) == 4
    assert len({candidate.id for candidate in candidates}) == 4
    assert len(candidate_selects) == 2


def test_list_candidates_filters_before_bounded_paging_without_n_plus_one(db):
    scope = _inference_scope(db)
    result = _decoder_result(scope.sentences)
    candidates = inference_service.persist_decoder_result(
        db,
        run_id=scope.run.id,
        document_id=scope.sentences[0].structure_version.document_id,
        structure_version_id=scope.sentences[0].structure_version_id,
        target_version_id=scope.target_version.id,
        result=result,
        source_window_ids=[scope.window.id],
    )
    assignment = TaskAssignment(
        project_id=scope.project.id,
        task_id=scope.target_version.target.task_id,
        document_id=candidates[0].document_id,
        assignee_user_id=scope.user.id,
        target_version_id=scope.target_version.id,
        structure_version_id=candidates[0].structure_version_id,
        assignment_scope_key=f"target:{scope.target_version.id}",
        annotator_id=scope.user.username,
        status="assigned",
    )
    db.add(assignment)
    db.flush()
    for candidate, action in zip(
        candidates,
        ("accept", "reject", None, "modify"),
        strict=True,
    ):
        if action is not None:
            db.add(
                EvidencePredictionReview(
                    prediction_id=candidate.id,
                    assignment_id=assignment.id,
                    reviewer_user_id=scope.user.id,
                    action=action,
                    revision=1,
                    metadata_={},
                )
            )
    db.commit()
    expected_ids = [candidate.id for candidate in candidates]
    run_id = scope.run.id
    db.refresh(scope.user)

    page, statements = _selected_statements(
        db,
        lambda: inference_service.list_candidates(
            db,
            run_id=run_id,
            user=scope.user,
            limit=2,
            offset=1,
        ),
    )

    assert [candidate.id for candidate in page] == expected_ids[1:3]
    assert [candidate.review_status for candidate in page] == ["rejected", "pending"]
    assert len(statements) == 2

    accepted = inference_service.list_candidates(
        db,
        run_id=run_id,
        user=scope.user,
        status="accepted",
        limit=1,
        offset=0,
    )
    assert [candidate.id for candidate in accepted] == expected_ids[:1]

    with pytest.raises(ValidationError, match="page size"):
        inference_service.list_candidates(
            db,
            run_id=run_id,
            user=scope.user,
            limit=inference_service.MAX_CANDIDATE_PAGE_SIZE + 1,
        )
