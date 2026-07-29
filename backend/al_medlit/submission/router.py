from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import (
    assert_document_resource_access,
    assert_project_member,
    assert_task_assigned,
    has_assignment_bypass,
    lock_document_resource_for_mutation,
)
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from al_medlit.core.storage import ObjectNotFoundError, ObjectStorage, get_object_storage
from al_medlit.submission import service
from al_medlit.submission.schemas import SubmissionCreate, SubmissionKind, SubmissionRead

router = APIRouter(tags=["submissions"])


@router.post(
    "/projects/{project_id}/documents/{document_id}/submissions",
    response_model=SubmissionRead,
)
def create_submission(
    project_id: int,
    document_id: int,
    payload: SubmissionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    lock_document_resource_for_mutation(
        db,
        current_user,
        project_id=project_id,
        document_id=document_id,
        lock_assignment=(payload.kind == "re_export" and payload.assignment_id is None),
    )
    payload.annotator_user_id = current_user.id
    payload.annotator_id = current_user.username
    return service.create_submission(
        db,
        storage,
        project_id=project_id,
        document_id=document_id,
        data=payload,
    )


@router.get("/projects/{project_id}/submissions", response_model=list[SubmissionRead])
def list_submissions(
    project_id: int,
    document_id: int | None = Query(None),
    annotator_user_id: int | None = Query(None),
    annotator_id: str | None = Query(None),
    kind: SubmissionKind | None = Query(None),
    scope: str | None = Query(None, pattern="^(mine|all)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    member = assert_project_member(db, current_user, project_id)
    requested_annotator_user_id = annotator_user_id
    restricted = not has_assignment_bypass(current_user, member)
    if document_id is not None:
        assert_document_resource_access(
            db,
            current_user,
            project_id=project_id,
            document_id=document_id,
        )
    if restricted and annotator_user_id not in (None, current_user.id):
        raise ForbiddenError("Cannot view another annotator's submissions")
    if restricted and annotator_id not in (None, current_user.username):
        raise ForbiddenError("Cannot view another annotator's submissions")
    if scope == "mine" or (scope is None and restricted):
        requested_annotator_user_id = current_user.id
        annotator_id = None
    elif scope == "all" and restricted:
        raise ForbiddenError("Insufficient workspace role to view all submissions")
    return service.list_submissions(
        db,
        project_id,
        document_id=document_id,
        annotator_user_id=requested_annotator_user_id,
        annotator_id=annotator_id,
        kind=kind,
        user=current_user,
    )


@router.get("/submissions/{submission_id}", response_model=SubmissionRead)
def get_submission(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    submission = service.get_submission(db, submission_id)
    assert_document_resource_access(
        db,
        current_user,
        project_id=submission.project_id,
        document_id=submission.document_id,
        owner_user_id=submission.annotator_user_id,
        enforce_owner=True,
    )
    return submission


@router.get("/submissions/{submission_id}/download")
def download_submission(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    submission = service.get_submission(db, submission_id)
    member = assert_document_resource_access(
        db,
        current_user,
        project_id=submission.project_id,
        document_id=submission.document_id,
        owner_user_id=submission.annotator_user_id,
        enforce_owner=True,
    )
    try:
        data = storage.get_bytes(submission.storage_key)
    except ObjectNotFoundError as exc:
        raise NotFoundError("Submission file is missing from storage") from exc
    if not service.snapshot_checksum_matches(data, submission.checksum_sha256):
        raise ConflictError("Submission file checksum does not match its metadata")
    if not has_assignment_bypass(current_user, member):
        if submission.assignment is not None:
            assert_task_assigned(
                db,
                current_user,
                member,
                project_id=submission.project_id,
                document_id=submission.document_id,
                annotation_type=submission.assignment.task.annotation_type,
                target_version_id=submission.assignment.target_version_id,
                structure_version_id=submission.assignment.structure_version_id,
                guideline_version_id=submission.assignment.guideline_version_id,
            )
        annotation_types = service.snapshot_annotation_types(data)
        if annotation_types is None:
            raise ForbiddenError("Unable to verify submission task access")
        for annotation_type in annotation_types:
            assert_task_assigned(
                db,
                current_user,
                member,
                project_id=submission.project_id,
                document_id=submission.document_id,
                annotation_type=annotation_type,
            )
    return Response(
        content=data,
        media_type=submission.content_type,
        headers={"Content-Disposition": f'attachment; filename="{submission.file_name}"'},
    )


@router.delete("/submissions/{submission_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_submission(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    submission = service.get_submission(db, submission_id)
    lock_document_resource_for_mutation(
        db,
        current_user,
        project_id=submission.project_id,
        document_id=submission.document_id,
        min_role="manager",
        lock_assignment=False,
    )
    service.delete_submission(db, storage, submission_id)
