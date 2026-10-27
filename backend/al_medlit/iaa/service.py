from sqlalchemy.orm import Session, joinedload

from al_medlit.annotation.models import Annotation
from al_medlit.annotation.types import REGISTRY
from al_medlit.auth.models import User
from al_medlit.core.exceptions import ValidationError
from al_medlit.corpus.models import Document
from al_medlit.evidence.models import EvidenceBlockAnnotation, EvidenceReviewCoverage
from al_medlit.iaa import alignment, evidence_metrics, metrics
from al_medlit.iaa.schemas import EvidencePairIaa, IaaReport
from al_medlit.project.models import TaskAssignment
from al_medlit.submission.models import AnnotationSubmission


def _compute_evidence_iaa(
    db: Session,
    *,
    project_id: int,
    document_id: int | None,
    target_version_id: int | None,
    structure_version_id: int | None,
    filter_structure_version: bool,
    guideline_version_id: int | None,
    filter_guideline_version: bool,
) -> IaaReport:
    if document_id is None:
        raise ValidationError("document_id is required for evidence-block IAA")
    if target_version_id is None:
        raise ValidationError("target_version_id is required for evidence-block IAA")

    coverage_query = db.query(EvidenceReviewCoverage).filter(
        EvidenceReviewCoverage.project_id == project_id,
        EvidenceReviewCoverage.document_id == document_id,
        EvidenceReviewCoverage.target_version_id == target_version_id,
    )
    if filter_structure_version or structure_version_id is not None:
        coverage_query = coverage_query.filter(
            EvidenceReviewCoverage.structure_version_id == structure_version_id
        )
    if filter_guideline_version or guideline_version_id is not None:
        coverage_query = coverage_query.filter(
            EvidenceReviewCoverage.guideline_version_id == guideline_version_id
        )
    coverage_rows = coverage_query.all()
    assignments = db.query(TaskAssignment).filter(
        TaskAssignment.project_id == project_id,
        TaskAssignment.document_id == document_id,
        TaskAssignment.target_version_id == target_version_id,
    ).populate_existing().all()
    if assignments:
        finalized_scopes = {
            (row.assignee_user_id, row.structure_version_id, row.guideline_version_id)
            for row in assignments
            if row.status in {"submitted", "adjudication_ready", "adjudicated", "completed"}
        }
        coverage_rows = [row for row in coverage_rows if (
            row.reviewer_user_id, row.structure_version_id, row.guideline_version_id
        ) in finalized_scopes]
    reviewer_ids = sorted({row.reviewer_user_id for row in coverage_rows})
    usernames = {
        user.id: user.username
        for user in db.query(User).filter(User.id.in_(reviewer_ids)).all()
    }
    annotator_ids = [usernames.get(user_id, str(user_id)) for user_id in reviewer_ids]
    report = IaaReport(
        project_id=project_id,
        annotation_type="evidence_block",
        document_id=document_id,
        status="insufficient_annotators",
        annotator_ids=annotator_ids,
        item_count=0,
        percent_agreement=None,
        cohens_kappa=None,
        fleiss_kappa=None,
        span_detection_f1=None,
        evidence_metrics={
            "target_version_id": target_version_id,
            "structure_version_id": structure_version_id,
            "guideline_version_id": guideline_version_id,
            "pairs": [],
            "aggregate": {},
        },
    )
    if len(reviewer_ids) < 2:
        return report

    coverage_by_scope: dict[tuple[int, int, int | None], list[tuple[int, int]]] = {}
    for row in coverage_rows:
        coverage_by_scope.setdefault(
            (
                row.reviewer_user_id,
                row.structure_version_id,
                row.guideline_version_id,
            ),
            [],
        ).append((row.start_sentence_ordinal, row.end_sentence_ordinal))

    pair_reports = []
    reviewed_intervals_by_structure: dict[int, list[tuple[int, int]]] = {}
    for left_index, left_user_id in enumerate(reviewer_ids):
        for right_user_id in reviewer_ids[left_index + 1 :]:
            common_scopes = sorted(
                {
                    (scope[1], scope[2])
                    for scope in coverage_by_scope
                    if scope[0] == left_user_id
                }
                & {
                    (scope[1], scope[2])
                    for scope in coverage_by_scope
                    if scope[0] == right_user_id
                },
                key=lambda item: (item[0], item[1] or -1),
            )
            for current_structure_id, current_guideline_id in common_scopes:
                intersection = evidence_metrics.intersect_intervals(
                    coverage_by_scope[
                        (left_user_id, current_structure_id, current_guideline_id)
                    ],
                    coverage_by_scope[
                        (right_user_id, current_structure_id, current_guideline_id)
                    ],
                )
                reviewed_sentence_count = sum(end - start + 1 for start, end in intersection)
                if reviewed_sentence_count == 0:
                    continue
                reviewed_intervals_by_structure.setdefault(current_structure_id, []).extend(
                    intersection
                )
                annotations = (
                    db.query(Annotation)
                    .join(EvidenceBlockAnnotation)
                    .filter(
                        Annotation.project_id == project_id,
                        Annotation.document_id == document_id,
                        Annotation.annotation_type == "evidence_block",
                        Annotation.source == "human",
                        Annotation.status != "gold",
                        Annotation.annotator_user_id.in_([left_user_id, right_user_id]),
                        EvidenceBlockAnnotation.target_version_id == target_version_id,
                        EvidenceBlockAnnotation.structure_version_id
                        == current_structure_id,
                    )
                    .all()
                )
                if current_guideline_id is None:
                    annotations = [
                        annotation
                        for annotation in annotations
                        if annotation.guideline_version_id is None
                    ]
                else:
                    annotations = [
                        annotation
                        for annotation in annotations
                        if annotation.guideline_version_id == current_guideline_id
                    ]
                spans_by_user = {
                    left_user_id: evidence_metrics.clip_spans(
                        [
                            (
                                annotation.evidence_block.start_sentence_ordinal,
                                annotation.evidence_block.end_sentence_ordinal,
                            )
                            for annotation in annotations
                            if annotation.annotator_user_id == left_user_id
                        ],
                        intersection,
                    ),
                    right_user_id: evidence_metrics.clip_spans(
                        [
                            (
                                annotation.evidence_block.start_sentence_ordinal,
                                annotation.evidence_block.end_sentence_ordinal,
                            )
                            for annotation in annotations
                            if annotation.annotator_user_id == right_user_id
                        ],
                        intersection,
                    ),
                }
                metrics_row = evidence_metrics.pair_metrics(
                    spans_by_user[left_user_id],
                    spans_by_user[right_user_id],
                    reviewed_sentence_count=reviewed_sentence_count,
                )
                metrics_row.update(
                    {
                        "left_annotator_id": usernames.get(
                            left_user_id, str(left_user_id)
                        ),
                        "right_annotator_id": usernames.get(
                            right_user_id, str(right_user_id)
                        ),
                    }
                )
                pair_reports.append(metrics_row)

    if not pair_reports:
        report.status = "no_items" if len(reviewer_ids) >= 2 else report.status
        return report
    report.status = "ok"
    report.item_count = 0
    for intervals in reviewed_intervals_by_structure.values():
        last_end = -1
        for start, end in sorted(intervals):
            report.item_count += max(0, end - max(start, last_end + 1) + 1)
            last_end = max(last_end, end)
    report.evidence_metrics.pairs = [
        EvidencePairIaa.model_validate(pair_report) for pair_report in pair_reports
    ]
    report.evidence_metrics.aggregate = evidence_metrics.average_pair_metrics(pair_reports)
    return report


def compute_iaa(
    db: Session,
    *,
    project_id: int,
    annotation_type: str,
    document_id: int | None = None,
    target_version_id: int | None = None,
    structure_version_id: int | None = None,
    filter_structure_version: bool = False,
    guideline_version_id: int | None = None,
    filter_guideline_version: bool = False,
) -> IaaReport:
    """Compute IAA for one annotation type in one project, optionally scoped to one document."""
    if annotation_type == "evidence_block":
        return _compute_evidence_iaa(
            db,
            project_id=project_id,
            document_id=document_id,
            target_version_id=target_version_id,
            structure_version_id=structure_version_id,
            filter_structure_version=filter_structure_version,
            guideline_version_id=guideline_version_id,
            filter_guideline_version=filter_guideline_version,
        )

    filter_structure = filter_structure_version or structure_version_id is not None
    filter_guideline = filter_guideline_version or guideline_version_id is not None
    query = (
        db.query(Annotation)
        .join(Document, Document.id == Annotation.document_id)
        .filter(
            Annotation.project_id == project_id,
            Document.project_id == project_id,
            Annotation.annotation_type == annotation_type,
            Annotation.source == "human",
            Annotation.status.in_(("draft", "accepted")),
        )
    )
    if annotation_type == "relation":
        query = query.options(joinedload(Annotation.head), joinedload(Annotation.tail))
    if document_id is not None:
        query = query.filter(Annotation.document_id == document_id)
    if filter_structure:
        query = query.filter(
            Annotation.structure_version_id == structure_version_id
        )
    if filter_guideline:
        query = query.filter(
            Annotation.guideline_version_id == guideline_version_id
        )
    annotations = query.all()

    selected_scopes = {
        (
            annotation.document_id,
            annotation.structure_version_id,
            annotation.guideline_version_id,
        )
        for annotation in annotations
    }
    submissions = db.query(AnnotationSubmission).options(
        joinedload(AnnotationSubmission.assignment).joinedload(TaskAssignment.task)
    ).filter(
        AnnotationSubmission.project_id == project_id,
        AnnotationSubmission.kind == "submission",
    )
    if document_id is not None:
        submissions = submissions.filter(AnnotationSubmission.document_id == document_id)
    submitted_annotators = set()
    for submission in submissions:
        assignment = submission.assignment
        if assignment is not None:
            if assignment.task.annotation_type != annotation_type:
                continue
            if filter_structure and assignment.structure_version_id != structure_version_id:
                continue
            if filter_guideline and assignment.guideline_version_id != guideline_version_id:
                continue
            selected_scopes.add((
                submission.document_id,
                assignment.structure_version_id,
                assignment.guideline_version_id,
            ))
        elif filter_structure or filter_guideline:
            continue
        if submission.annotator_id:
            submitted_annotators.add(submission.annotator_id)

    # Never collapse multiple annotation rounds into one agreement score. A
    # structure id is document-specific, so project-wide IAA may legitimately
    # contain one different structure per document; what is ambiguous is more
    # than one structure for the same document, or more than one guideline
    # version across the selected records when that pin was omitted.
    if not filter_guideline:
        selected_guidelines = {scope[2] for scope in selected_scopes}
        if len(selected_guidelines) > 1:
            raise ValidationError(
                "IAA cannot combine annotation rounds; specify guideline_version_id"
            )
    if not filter_structure:
        structures_by_document: dict[int, set[int | None]] = {}
        for scope_document_id, scope_structure_id, _ in selected_scopes:
            structures_by_document.setdefault(scope_document_id, set()).add(scope_structure_id)
        if any(len(structures) > 1 for structures in structures_by_document.values()):
            raise ValidationError(
                "IAA cannot combine annotation rounds; specify document_id and "
                "structure_version_id"
            )

    try:
        annotators, items, spans_by_annotator = alignment.build_rating_matrix(
            annotations, submitted_annotators=submitted_annotators,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    report = IaaReport(
        project_id=project_id,
        annotation_type=annotation_type,
        document_id=document_id,
        status="ok",
        annotator_ids=annotators,
        item_count=len(items),
        percent_agreement=None,
        cohens_kappa=None,
        fleiss_kappa=None,
        span_detection_f1=None,
    )

    if len(annotators) < 2:
        report.status = "insufficient_annotators"
        return report
    if not items:
        report.status = "no_items"
        return report

    report.percent_agreement = metrics.percent_agreement(items, annotators)
    report.fleiss_kappa = metrics.fleiss_kappa(items, annotators)
    if len(annotators) == 2:
        report.cohens_kappa = metrics.cohens_kappa(items, annotators[0], annotators[1])

    spec = REGISTRY.get(annotation_type)
    if spec is not None and spec.requires_span:
        report.span_detection_f1 = metrics.pairwise_span_f1(spans_by_annotator, annotators)

    return report
