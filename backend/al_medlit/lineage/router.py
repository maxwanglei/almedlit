from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_project_member
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import NotFoundError
from al_medlit.core.storage import ObjectStorage, get_object_storage
from al_medlit.lineage import service
from al_medlit.lineage.models import AnnotationSet, CorpusSnapshot, LineageArtifact
from al_medlit.lineage.schemas import (
    AnnotationSetCreate,
    AnnotationSetRead,
    CorpusSnapshotCreate,
    CorpusSnapshotRead,
    LineageGraphRead,
)
from al_medlit.workspace.capability_dependencies import enforce_capability

router = APIRouter(tags=["lineage"])


@router.post(
    "/projects/{project_id}/corpus-snapshots",
    response_model=CorpusSnapshotRead,
)
def create_corpus_snapshot(
    project_id: int,
    payload: CorpusSnapshotCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    assert_project_member(db, current_user, project_id, min_role="manager")
    enforce_capability(db, project_id=project_id, key="lineage")
    return service.freeze_corpus_snapshot(
        db,
        storage,
        project_id=project_id,
        data=payload,
        actor_user_id=current_user.id,
    )


@router.get(
    "/projects/{project_id}/corpus-snapshots",
    response_model=list[CorpusSnapshotRead],
)
def list_corpus_snapshots(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_project_member(db, current_user, project_id, min_role="trainer")
    enforce_capability(db, project_id=project_id, key="lineage")
    return (
        db.query(CorpusSnapshot)
        .filter(CorpusSnapshot.project_id == project_id)
        .order_by(CorpusSnapshot.created_at.desc())
        .all()
    )


@router.get("/corpus-snapshots/{snapshot_id}", response_model=CorpusSnapshotRead)
def get_corpus_snapshot(
    snapshot_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    snapshot = db.get(CorpusSnapshot, snapshot_id)
    if snapshot is None:
        raise NotFoundError("Corpus snapshot not found")
    assert_project_member(db, current_user, snapshot.project_id, min_role="trainer")
    enforce_capability(db, project_id=snapshot.project_id, key="lineage")
    return snapshot


@router.post(
    "/projects/{project_id}/annotation-sets",
    response_model=AnnotationSetRead,
)
def create_annotation_set(
    project_id: int,
    payload: AnnotationSetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    assert_project_member(db, current_user, project_id, min_role="manager")
    enforce_capability(db, project_id=project_id, key="lineage")
    return service.freeze_annotation_set(
        db,
        storage,
        project_id=project_id,
        data=payload,
        actor_user_id=current_user.id,
    )


@router.get(
    "/projects/{project_id}/annotation-sets",
    response_model=list[AnnotationSetRead],
)
def list_annotation_sets(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_project_member(db, current_user, project_id, min_role="trainer")
    enforce_capability(db, project_id=project_id, key="lineage")
    return (
        db.query(AnnotationSet)
        .filter(AnnotationSet.project_id == project_id)
        .order_by(AnnotationSet.created_at.desc())
        .all()
    )


@router.get("/annotation-sets/{annotation_set_id}", response_model=AnnotationSetRead)
def get_annotation_set(
    annotation_set_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    annotation_set = db.get(AnnotationSet, annotation_set_id)
    if annotation_set is None:
        raise NotFoundError("Annotation set not found")
    assert_project_member(db, current_user, annotation_set.project_id, min_role="trainer")
    enforce_capability(db, project_id=annotation_set.project_id, key="lineage")
    return annotation_set


@router.get(
    "/lineage/artifacts/{artifact_id}/graph",
    response_model=LineageGraphRead,
)
def get_lineage_graph(
    artifact_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    artifact = db.get(LineageArtifact, artifact_id)
    if artifact is None:
        raise NotFoundError("Lineage artifact not found")
    assert_project_member(db, current_user, artifact.project_id, min_role="trainer")
    enforce_capability(db, project_id=artifact.project_id, key="lineage")
    artifacts, edges = service.lineage_graph(db, artifact_id)
    return {
        "artifacts": artifacts,
        "edges": [
            {
                "id": edge.id,
                "upstream_artifact_id": edge.upstream_artifact_id,
                "downstream_artifact_id": edge.downstream_artifact_id,
                "relationship_type": edge.relationship_type,
                "metadata": edge.metadata_,
            }
            for edge in edges
        ],
    }
