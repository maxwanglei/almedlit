"""Integration coverage for the canonical learning workflow."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from al_medlit.auth.models import User
from al_medlit.auth.security import create_access_token
from al_medlit.core.exceptions import ConflictError, ForbiddenError, ValidationError
from al_medlit.corpus.models import Document
from al_medlit.lineage.models import (
    ImmutableRecordError,
)
from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.model_artifacts.models import (
    ArtifactStorageReservation,
    BaseModelAsset,
    BaseModelAssetState,
)
from al_medlit.model_artifacts.schemas import ArtifactPackageCreate
from al_medlit.project.models import Project
from al_medlit.training.runtime_profiles import RUNTIME_PROFILES, RuntimeReadinessReport
from al_medlit.workflow import models, schemas, service
from al_medlit.workspace import capability_service
from al_medlit.workspace import service as workspace_service


def _user(db, username: str) -> User:
    user = User(
        username=username,
        password_hash="test-only",
        display_name=username,
        is_active=True,
    )
    db.add(user)
    db.flush()
    return user


def _headers(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(str(user.id))}"}


def _runtime_report(runtime_class: str, image_digest: str) -> dict:
    descriptor = RUNTIME_PROFILES[runtime_class]
    return RuntimeReadinessReport(
        runtime_profile=runtime_class,
        generated_at=datetime.now(UTC),
        python_version="3.12.0",
        worker_image_digest=image_digest.removeprefix("sha256:"),
        dependency_versions={name: "test" for name in descriptor.required_imports},
        missing_dependencies=[],
        device=descriptor.required_device,
        device_available=True,
        device_memory_bytes=16 * 1024 * 1024 * 1024,
        scratch_path="/tmp/al-medlit-test",
        scratch_available_bytes=descriptor.minimum_scratch_bytes,
        storage_access_verified=True,
        ready=True,
    ).model_dump(mode="json")


def test_registered_model_names_are_nonblank_and_normalized():
    assert schemas.RegisteredModelCreate(project_id=1, name="  classifier  ").name == ("classifier")
    with pytest.raises(ValueError, match="Model name cannot be blank"):
        schemas.RegisteredModelCreate(project_id=1, name="   ")


@pytest.fixture
def workflow_scope(db):
    manager = _user(db, "workflow-manager")
    trainer = _user(db, "workflow-trainer")
    annotator = _user(db, "workflow-annotator")
    outsider = _user(db, "workflow-outsider")
    workspace = workspace_service.create_team_workspace(db, manager, name="Workflow Workspace")
    capability_service.set_capability(
        db,
        workspace.id,
        preset="full",
        actor_user_id=manager.id,
    )
    workspace_service.add_member(db, workspace.id, trainer.id, role="trainer")
    workspace_service.add_member(db, workspace.id, annotator.id, role="annotator")
    project = Project(name="Workflow Project", workspace_id=workspace.id)
    db.add(project)
    db.commit()
    return SimpleNamespace(
        project=project,
        workspace=workspace,
        manager=manager,
        trainer=trainer,
        annotator=annotator,
        outsider=outsider,
    )


def test_project_corpus_snapshot_requires_trainer_role(client, db, workflow_scope):
    dataset = models.Dataset(
        project_id=workflow_scope.project.id,
        name="Authorized corpus snapshot",
        source_type="project_corpus",
        created_by_user_id=workflow_scope.manager.id,
    )
    document = Document(
        project_id=workflow_scope.project.id,
        external_id="authorized-document",
        title="Authorization source",
        text="Only trainers and higher roles may snapshot the project corpus.",
        source="test",
        metadata_={},
    )
    db.add_all([dataset, document])
    db.commit()
    path = (
        f"/api/projects/{workflow_scope.project.id}/datasets/{dataset.id}/versions/project-corpus"
    )

    annotator = client.post(path, headers=_headers(workflow_scope.annotator))
    assert annotator.status_code == 403

    trainer = client.post(path, headers=_headers(workflow_scope.trainer))
    assert trainer.status_code == 201, trainer.text
    assert trainer.json()["created_by_user_id"] == workflow_scope.trainer.id

    workflow_scope.project.settings = {
        "modules": ["models", "activity"],
    }
    db.commit()
    blocked_by_capability = client.post(path, headers=_headers(workflow_scope.trainer))
    assert blocked_by_capability.status_code == 403
    assert "data" in blocked_by_capability.json()["detail"]


def test_storage_policy_is_bound_to_real_backend_prefix_and_encryption(
    db,
    workflow_scope,
    monkeypatch,
):
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "storage_backend", "minio")
    monkeypatch.setattr(settings, "storage_encryption_mode", "none")
    monkeypatch.setattr(settings, "storage_kms_key_id", "")
    expected_prefix = artifact_service.workspace_blob_prefix(workflow_scope.workspace.id)

    with pytest.raises(ValidationError, match="must equal"):
        service.create_storage_policy(
            db,
            schemas.StoragePolicyCreate(
                project_id=workflow_scope.project.id,
                name="Nested project prefix",
                backend="minio",
                artifact_prefix=f"{expected_prefix}/project-{workflow_scope.project.id}",
            ),
            workflow_scope.manager,
        )
    with pytest.raises(ValidationError, match="configured object store"):
        service.create_storage_policy(
            db,
            schemas.StoragePolicyCreate(
                project_id=workflow_scope.project.id,
                name="Wrong backend",
                backend="local",
                artifact_prefix=expected_prefix,
            ),
            workflow_scope.manager,
        )
    with pytest.raises(ValidationError, match="does not match"):
        service.create_storage_policy(
            db,
            schemas.StoragePolicyCreate(
                project_id=workflow_scope.project.id,
                name="False encryption claim",
                backend="minio",
                artifact_prefix=expected_prefix,
                encryption={"mode": "sse-s3"},
            ),
            workflow_scope.manager,
        )
    with pytest.raises(ValidationError, match="unsupported fields"):
        service.create_storage_policy(
            db,
            schemas.StoragePolicyCreate(
                project_id=workflow_scope.project.id,
                name="Legacy encryption claim",
                backend="minio",
                artifact_prefix=expected_prefix,
                encryption={"enabled": True, "algorithm": "AES-256-GCM"},
            ),
            workflow_scope.manager,
        )
    with pytest.raises(ValidationError, match="cache policies"):
        service.create_storage_policy(
            db,
            schemas.StoragePolicyCreate(
                project_id=workflow_scope.project.id,
                name="Unsupported cache",
                backend="minio",
                artifact_prefix=expected_prefix,
                cache_policy={"enabled": True},
            ),
            workflow_scope.manager,
        )

    monkeypatch.setattr(settings, "storage_backend", "local")
    monkeypatch.setattr(settings, "deployment_profile", "lab")
    with pytest.raises(ValidationError, match="development-only"):
        service.create_storage_policy(
            db,
            schemas.StoragePolicyCreate(
                project_id=workflow_scope.project.id,
                name="Shared local files",
                backend="local",
                artifact_prefix=expected_prefix,
            ),
            workflow_scope.manager,
        )
    monkeypatch.setattr(settings, "storage_backend", "minio")
    monkeypatch.setattr(settings, "deployment_profile", "laptop")
    policy = service.create_storage_policy(
        db,
        schemas.StoragePolicyCreate(
            project_id=workflow_scope.project.id,
            name="Executable object store",
            backend="minio",
            artifact_prefix=expected_prefix,
        ),
        workflow_scope.manager,
    )
    assert policy.artifact_prefix == expected_prefix
    assert policy.encryption == {"mode": "none"}


def test_storage_policy_records_configured_minio_kms_encryption(
    db,
    workflow_scope,
    monkeypatch,
):
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "storage_backend", "minio")
    monkeypatch.setattr(settings, "storage_encryption_mode", "sse-kms")
    monkeypatch.setattr(settings, "storage_kms_key_id", "workspace-models")
    policy = service.create_storage_policy(
        db,
        schemas.StoragePolicyCreate(
            project_id=workflow_scope.project.id,
            name="KMS object store",
            backend="minio",
            artifact_prefix=artifact_service.workspace_blob_prefix(workflow_scope.workspace.id),
        ),
        workflow_scope.manager,
    )

    assert policy.encryption == {
        "mode": "sse-kms",
        "key_id": "workspace-models",
    }


def test_available_environment_requires_complete_ready_attestation(
    db,
    workflow_scope,
):
    image_digest = "sha256:" + "c" * 64
    environment = service.create_execution_environment(
        db,
        schemas.ExecutionEnvironmentCreate(
            project_id=workflow_scope.project.id,
            name="Incomplete worker",
            environment_class="classical-cpu",
            image_digest=image_digest,
        ),
        workflow_scope.manager,
    )
    incomplete = _runtime_report("classical-cpu", image_digest)
    incomplete["dependency_versions"].pop("skops")
    with pytest.raises(ValidationError, match="dependencies do not match"):
        service.verify_execution_environment(
            db,
            workflow_scope.project.id,
            environment.id,
            schemas.EnvironmentVerification(
                status="available",
                verification_report=incomplete,
            ),
        )

    no_memory = _runtime_report("classical-cpu", image_digest)
    no_memory["device_memory_bytes"] = 0
    with pytest.raises(ValidationError, match="measure available device memory"):
        service.verify_execution_environment(
            db,
            workflow_scope.project.id,
            environment.id,
            schemas.EnvironmentVerification(
                status="available",
                verification_report=no_memory,
            ),
        )


def test_runtime_and_storage_provisioning_requires_workspace_admin(
    client,
    workflow_scope,
):
    project_id = workflow_scope.project.id
    image_digest = "sha256:" + "d" * 64
    environment_payload = {
        "project_id": project_id,
        "name": "Admin-managed CPU worker",
        "environment_class": "classical-cpu",
        "image_digest": image_digest,
        "package_manifest": {},
        "hardware_constraints": {},
    }

    trainer_create = client.post(
        "/api/environments",
        json=environment_payload,
        headers=_headers(workflow_scope.trainer),
    )
    assert trainer_create.status_code == 403

    admin_create = client.post(
        "/api/environments",
        json=environment_payload,
        headers=_headers(workflow_scope.manager),
    )
    assert admin_create.status_code == 201, admin_create.text
    environment_id = admin_create.json()["id"]

    verification_path = f"/api/environments/{environment_id}/verification?project_id={project_id}"
    verification_payload = {
        "status": "available",
        "verification_report": _runtime_report("classical-cpu", image_digest),
    }
    trainer_verify = client.post(
        verification_path,
        json=verification_payload,
        headers=_headers(workflow_scope.trainer),
    )
    assert trainer_verify.status_code == 403

    admin_verify = client.post(
        verification_path,
        json=verification_payload,
        headers=_headers(workflow_scope.manager),
    )
    assert admin_verify.status_code == 200, admin_verify.text

    storage_payload = {
        "project_id": project_id,
        "name": "Admin-managed artifact storage",
        "backend": "minio",
        "artifact_prefix": artifact_service.workspace_blob_prefix(workflow_scope.workspace.id),
        "retention_class": "indefinite",
        "encryption": {"mode": "none"},
        "cache_policy": {},
        "is_default": True,
    }
    trainer_storage = client.post(
        "/api/storage/policies",
        json=storage_payload,
        headers=_headers(workflow_scope.trainer),
    )
    assert trainer_storage.status_code == 403

    admin_storage = client.post(
        "/api/storage/policies",
        json=storage_payload,
        headers=_headers(workflow_scope.manager),
    )
    assert admin_storage.status_code == 201, admin_storage.text


@pytest.fixture
def workflow_data(db, workflow_scope):
    project_id = workflow_scope.project.id
    actor = workflow_scope.manager
    task = service.create_task_definition(
        db,
        schemas.TaskDefinitionCreate(
            project_id=project_id,
            key="sentiment",
            name="Sentiment",
        ),
        actor,
    )
    task_version = service.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=project_id,
            task_definition_id=task.id,
            task_kind="classification",
            input_schema={"text": "string"},
            output_schema={"label": ["negative", "positive"]},
            metrics=["f1"],
            trainer_compatibility=[
                "linear-classification",
                "tfidf_logistic_regression",
                "sklearn_tfidf",
            ],
        ),
        actor,
    )
    dataset = service.create_dataset(
        db,
        schemas.DatasetCreate(
            project_id=project_id,
            name="Public reviews",
            source_type="public_registry",
        ),
        actor,
    )
    dataset_version = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project_id,
            dataset_id=dataset.id,
            source_uri="hf://example/reviews",
            source_revision="0123456789abcdef",
            source_format="jsonl",
            data_schema={"text": "string"},
            provenance={"registry": "huggingface"},
            license_info={"name": "apache-2.0"},
            items=[
                schemas.DatasetItemCreate(
                    stable_key="train-1",
                    group_key="patient-1",
                    payload={"text": "Helpful"},
                ),
                schemas.DatasetItemCreate(
                    stable_key="pool-1",
                    group_key="patient-2",
                    payload={"text": "Unclear"},
                ),
                schemas.DatasetItemCreate(
                    stable_key="test-1",
                    group_key="patient-3",
                    payload={"text": "Protected"},
                ),
            ],
        ),
        actor,
    )
    items = {
        item.stable_key: item
        for item in service.list_dataset_items(db, project_id, dataset_version.id)
    }
    split_map = service.create_split_map(
        db,
        schemas.SplitMapCreate(
            project_id=project_id,
            dataset_version_id=dataset_version.id,
            name="fixed-v1",
            strategy="grouped",
            assignments={
                "train-1": "train",
                "pool-1": "pool",
                "test-1": "test",
            },
            protected_splits=["test"],
        ),
        actor,
    )
    return SimpleNamespace(
        task=task,
        task_version=task_version,
        dataset=dataset,
        dataset_version=dataset_version,
        items=items,
        split_map=split_map,
    )


def test_versions_are_immutable_and_protected_items_cannot_be_selected(
    db, workflow_scope, workflow_data
):
    selection_run = service.create_selection_run(
        db,
        schemas.SelectionRunCreate(
            project_id=workflow_scope.project.id,
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            split_map_id=workflow_data.split_map.id,
            strategy="uncertainty",
        ),
        workflow_scope.trainer,
    )
    with pytest.raises(ValidationError, match="Protected evaluation items"):
        service.create_selection_set(
            db,
            schemas.SelectionSetCreate(
                project_id=workflow_scope.project.id,
                selection_run_id=selection_run.id,
                items=[
                    schemas.SelectionItem(
                        dataset_item_id=workflow_data.items["test-1"].id,
                        rank=1,
                        score=0.9,
                    )
                ],
            ),
            workflow_scope.trainer,
        )

    workflow_data.dataset_version.source_revision = "rewritten"
    with pytest.raises(ImmutableRecordError, match="immutable"):
        db.commit()
    db.rollback()
    assert db.get(models.DatasetVersion, workflow_data.dataset_version.id).source_revision == (
        "0123456789abcdef"
    )


def test_round_work_items_are_assignment_scoped_and_do_not_expose_dataset_pool(
    db,
    client,
    workflow_scope,
    workflow_data,
):
    project_id = workflow_scope.project.id
    unassigned = _user(db, "workflow-unassigned-annotator")
    workspace_service.add_member(
        db,
        workflow_scope.workspace.id,
        unassigned.id,
        role="annotator",
    )
    selection_run = service.create_selection_run(
        db,
        schemas.SelectionRunCreate(
            project_id=project_id,
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            split_map_id=workflow_data.split_map.id,
            strategy="random",
        ),
        workflow_scope.trainer,
    )
    selection_set = service.create_selection_set(
        db,
        schemas.SelectionSetCreate(
            project_id=project_id,
            selection_run_id=selection_run.id,
            items=[
                schemas.SelectionItem(
                    dataset_item_id=workflow_data.items["pool-1"].id,
                    rank=1,
                    reason={"signal": "random"},
                )
            ],
        ),
        workflow_scope.trainer,
    )
    annotation_round = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=project_id,
            name="Assigned pool item",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            selection_set_version_id=selection_set.id,
            assistance_policy="blind",
            annotator_user_ids=[workflow_scope.annotator.id],
        ),
        workflow_scope.manager,
    )
    service.transition_annotation_round(
        db,
        project_id,
        annotation_round.id,
        "open",
    )

    dataset_path = (
        f"/api/datasets/items?project_id={project_id}"
        f"&dataset_version_id={workflow_data.dataset_version.id}"
    )
    assert client.get(dataset_path, headers=_headers(workflow_scope.annotator)).status_code == 403

    work_path = f"/api/rounds/{annotation_round.id}/work-items?project_id={project_id}"
    unassigned_response = client.get(work_path, headers=_headers(unassigned))
    assert unassigned_response.status_code == 403
    assert "not assigned" in unassigned_response.json()["detail"].lower()
    assert (
        client.get(
            f"/api/rounds/{annotation_round.id}/items?project_id={project_id}",
            headers=_headers(unassigned),
        ).status_code
        == 403
    )

    assigned_response = client.get(
        work_path,
        headers=_headers(workflow_scope.annotator),
    )
    assert assigned_response.status_code == 200
    round_item_payload = assigned_response.json()[0]["round_item"]
    assert set(round_item_payload) == {
        "id",
        "created_at",
        "updated_at",
        "project_id",
        "annotation_round_id",
        "dataset_item_id",
    }
    assert not {
        "selection_rank",
        "selection_score",
        "selection_probability",
        "selection_reason",
        "metadata",
    }.intersection(round_item_payload)
    assert [
        (
            item["round_item"]["dataset_item_id"],
            item["dataset_item"]["stable_key"],
            item["dataset_item"]["payload"],
        )
        for item in assigned_response.json()
    ] == [
        (
            workflow_data.items["pool-1"].id,
            "pool-1",
            {"text": "Unclear"},
        )
    ]
    serialized_work = str(assigned_response.json())
    assert "train-1" not in serialized_work
    assert "test-1" not in serialized_work
    assert "Protected" not in serialized_work
    assert (
        client.get(
            f"/api/rounds/{annotation_round.id}/items?project_id={project_id}",
            headers=_headers(workflow_scope.annotator),
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/datasets/label-sets",
            params={
                "project_id": project_id,
                "dataset_version_id": workflow_data.dataset_version.id,
            },
            headers=_headers(workflow_scope.annotator),
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/datasets/split-maps",
            params={
                "project_id": project_id,
                "dataset_version_id": workflow_data.dataset_version.id,
            },
            headers=_headers(workflow_scope.annotator),
        ).status_code
        == 403
    )

    manager_response = client.get(
        work_path,
        headers=_headers(workflow_scope.manager),
    )
    assert manager_response.status_code == 200
    assert [item["dataset_item"]["stable_key"] for item in manager_response.json()] == ["pool-1"]
    manager_items_response = client.get(
        f"/api/rounds/{annotation_round.id}/items?project_id={project_id}",
        headers=_headers(workflow_scope.manager),
    )
    assert manager_items_response.status_code == 200
    assert manager_items_response.json()[0]["selection_reason"] == {"signal": "random"}
    dataset_response = client.get(
        dataset_path,
        headers=_headers(workflow_scope.manager),
    )
    assert dataset_response.status_code == 200
    assert {item["stable_key"] for item in dataset_response.json()} == {
        "train-1",
        "pool-1",
        "test-1",
    }


def test_same_item_can_be_reannotated_with_feedback_without_mixing_rounds(
    db, client, workflow_scope, workflow_data
):
    project_id = workflow_scope.project.id
    selection_run = service.create_selection_run(
        db,
        schemas.SelectionRunCreate(
            project_id=project_id,
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            split_map_id=workflow_data.split_map.id,
            strategy="uncertainty",
        ),
        workflow_scope.trainer,
    )
    selection_set = service.create_selection_set(
        db,
        schemas.SelectionSetCreate(
            project_id=project_id,
            selection_run_id=selection_run.id,
            items=[
                schemas.SelectionItem(
                    dataset_item_id=workflow_data.items["pool-1"].id,
                    rank=1,
                    score=0.8,
                    reason={"signal": "entropy"},
                )
            ],
        ),
        workflow_scope.trainer,
    )
    llm_feedback = schemas.FeedbackRunCreate(
        project_id=project_id,
        dataset_version_id=workflow_data.dataset_version.id,
        task_version_id=workflow_data.task_version.id,
        producer_type="external_llm",
        provider="approved-local",
        external_model_id="reviewer",
        exact_revision="r1",
        prompt_template_hash="a" * 64,
        configuration={"temperature": 0, "top_p": 1},
        data_egress_policy={"mode": "local_only"},
    )
    with pytest.raises(ForbiddenError, match="Manager role"):
        service.create_feedback_run(db, llm_feedback, workflow_scope.trainer)
    feedback_run = service.create_feedback_run(db, llm_feedback, workflow_scope.manager)
    feedback_set = service.create_feedback_set(
        db,
        schemas.FeedbackSetCreate(
            project_id=project_id,
            feedback_run_id=feedback_run.id,
            candidates=[
                schemas.FeedbackCandidateCreate(
                    dataset_item_id=workflow_data.items["pool-1"].id,
                    output={"label": "positive"},
                    score=0.91,
                )
            ],
        ),
        workflow_scope.trainer,
    )
    candidate = service.list_feedback_candidates(db, project_id, feedback_set.id)[0]
    hidden = client.get(
        "/api/feedback-runs/candidates",
        params={
            "project_id": project_id,
            "feedback_set_version_id": feedback_set.id,
        },
        headers=_headers(workflow_scope.annotator),
    )
    assert hidden.status_code == 403
    manager_view = client.get(
        "/api/feedback-runs/candidates",
        params={
            "project_id": project_id,
            "feedback_set_version_id": feedback_set.id,
        },
        headers=_headers(workflow_scope.manager),
    )
    assert manager_view.status_code == 200, manager_view.text
    assert manager_view.json()[0]["id"] == candidate.id

    with pytest.raises(ValidationError, match="explicit annotators"):
        service.create_annotation_round(
            db,
            schemas.AnnotationRoundCreate(
                project_id=project_id,
                name="Implicitly open round",
                dataset_version_id=workflow_data.dataset_version.id,
                task_version_id=workflow_data.task_version.id,
                selection_set_version_id=selection_set.id,
                feedback_set_version_id=feedback_set.id,
            ),
            workflow_scope.manager,
        )

    round_ids = []
    round_item_ids = []
    for index in range(2):
        annotation_round = service.create_annotation_round(
            db,
            schemas.AnnotationRoundCreate(
                project_id=project_id,
                name=f"Review {index + 1}",
                dataset_version_id=workflow_data.dataset_version.id,
                task_version_id=workflow_data.task_version.id,
                parent_round_id=round_ids[-1] if round_ids else None,
                selection_set_version_id=selection_set.id,
                feedback_set_version_id=feedback_set.id,
                assistance_policy="reveal_after_first_pass",
                annotator_user_ids=[workflow_scope.annotator.id],
                reason="Re-check model uncertainty",
            ),
            workflow_scope.manager,
        )
        round_item = service.list_round_items(db, project_id, annotation_round.id)[0]
        service.transition_annotation_round(db, project_id, annotation_round.id, "open")
        round_ids.append(annotation_round.id)
        round_item_ids.append(round_item.id)

    assert round_item_ids[0] != round_item_ids[1]
    assert {db.get(models.RoundItem, item_id).dataset_item_id for item_id in round_item_ids} == {
        workflow_data.items["pool-1"].id
    }

    with pytest.raises(ForbiddenError, match="Manager role"):
        service.create_annotation_decision(
            db,
            schemas.AnnotationDecisionCreate(
                project_id=project_id,
                round_item_id=round_item_ids[1],
                output={"label": "negative"},
                decision_kind="adjudication",
            ),
            workflow_scope.annotator,
        )

    with pytest.raises(ConflictError, match="independent initial annotation"):
        service.create_feedback_exposure(
            db,
            schemas.FeedbackExposureCreate(
                project_id=project_id,
                round_item_id=round_item_ids[1],
                feedback_candidate_id=candidate.id,
                exposure_mode="reveal_after_first_pass",
            ),
            workflow_scope.annotator,
        )

    initial = service.create_annotation_decision(
        db,
        schemas.AnnotationDecisionCreate(
            project_id=project_id,
            round_item_id=round_item_ids[1],
            output={"label": "negative"},
            is_initial_checkpoint=True,
        ),
        workflow_scope.annotator,
    )
    reveal = client.post(
        (f"/api/rounds/{round_ids[1]}/items/{round_item_ids[1]}/feedback-reveal"),
        json={
            "project_id": project_id,
            "candidate_key": "primary",
            "context": {"surface": "annotator-workspace"},
        },
        headers=_headers(workflow_scope.annotator),
    )
    assert reveal.status_code == 201
    reveal_payload = reveal.json()
    assert reveal_payload["exposure"]["user_id"] == workflow_scope.annotator.id
    assert reveal_payload["exposure"]["feedback_candidate_id"] == candidate.id
    assert reveal_payload["candidate"]["output"] == {"label": "positive"}
    assert initial.round_item_id == round_item_ids[1]
    submission = service.create_round_submission(
        db,
        schemas.RoundSubmissionCreate(
            project_id=project_id,
            annotation_round_id=round_ids[1],
            decision_ids=[initial.id],
        ),
        workflow_scope.annotator,
    )
    service.transition_annotation_round(db, project_id, round_ids[1], "closed")
    human_labels = service.create_round_label_set(
        db,
        round_ids[1],
        schemas.RoundLabelSetCreate(
            project_id=project_id,
            name="Round 2 finalized labels",
            source_kind="human",
            submission_ids=[submission.id],
        ),
        workflow_scope.manager,
    )
    assert human_labels.labels == {"pool-1": {"label": "negative"}}
    assert human_labels.source_annotation_round_id == round_ids[1]
    assert human_labels.source_submission_ids == [submission.id]
    assert human_labels.source_decision_ids == [initial.id]

    with pytest.raises(ValidationError, match="derived from finalized"):
        service.create_label_set_version(
            db,
            schemas.LabelSetVersionCreate(
                project_id=project_id,
                dataset_version_id=workflow_data.dataset_version.id,
                task_version_id=workflow_data.task_version.id,
                name="forged-human-labels",
                source_kind="human",
                labels={"pool-1": {"label": "positive"}},
            ),
            workflow_scope.trainer,
        )


def test_round_history_is_user_scoped_and_manager_auditable(
    db, client, workflow_scope, workflow_data
):
    project_id = workflow_scope.project.id
    reviewer = _user(db, "workflow-round-reviewer")
    unassigned = _user(db, "workflow-round-unassigned")
    workspace_service.add_member(
        db,
        workflow_scope.project.workspace_id,
        reviewer.id,
        role="annotator",
    )
    workspace_service.add_member(
        db,
        workflow_scope.project.workspace_id,
        unassigned.id,
        role="annotator",
    )
    annotation_round = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=project_id,
            name="Private annotation history",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=[workflow_scope.annotator.id, reviewer.id],
        ),
        workflow_scope.manager,
    )
    service.transition_annotation_round(db, project_id, annotation_round.id, "open")
    round_item = service.list_round_items(db, project_id, annotation_round.id)[0]

    first = service.create_annotation_decision(
        db,
        schemas.AnnotationDecisionCreate(
            project_id=project_id,
            round_item_id=round_item.id,
            output={"label": "negative"},
            is_initial_checkpoint=True,
        ),
        workflow_scope.annotator,
    )
    revised = service.create_annotation_decision(
        db,
        schemas.AnnotationDecisionCreate(
            project_id=project_id,
            round_item_id=round_item.id,
            supersedes_decision_id=first.id,
            output={"label": "positive"},
        ),
        workflow_scope.annotator,
    )
    annotator_submission = service.create_round_submission(
        db,
        schemas.RoundSubmissionCreate(
            project_id=project_id,
            annotation_round_id=annotation_round.id,
            decision_ids=[revised.id],
        ),
        workflow_scope.annotator,
    )
    reviewer_decision = service.create_annotation_decision(
        db,
        schemas.AnnotationDecisionCreate(
            project_id=project_id,
            round_item_id=round_item.id,
            output={"label": "negative"},
        ),
        reviewer,
    )
    reviewer_submission = service.create_round_submission(
        db,
        schemas.RoundSubmissionCreate(
            project_id=project_id,
            annotation_round_id=annotation_round.id,
            decision_ids=[reviewer_decision.id],
        ),
        reviewer,
    )

    annotator_decisions = client.get(
        f"/api/rounds/{annotation_round.id}/decisions",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.annotator),
    )
    assert annotator_decisions.status_code == 200, annotator_decisions.text
    assert [entry["id"] for entry in annotator_decisions.json()] == [
        first.id,
        revised.id,
    ]
    assert annotator_decisions.json()[1]["supersedes_decision_id"] == first.id

    annotator_submissions = client.get(
        f"/api/rounds/{annotation_round.id}/submissions",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.annotator),
    )
    assert annotator_submissions.status_code == 200, annotator_submissions.text
    assert [entry["id"] for entry in annotator_submissions.json()] == [annotator_submission.id]

    other_annotator = client.get(
        f"/api/rounds/{annotation_round.id}/decisions",
        params={"project_id": project_id, "annotator_user_id": reviewer.id},
        headers=_headers(workflow_scope.annotator),
    )
    assert other_annotator.status_code == 403

    not_assigned = client.get(
        f"/api/rounds/{annotation_round.id}/submissions",
        params={"project_id": project_id},
        headers=_headers(unassigned),
    )
    assert not_assigned.status_code == 403

    manager_decisions = client.get(
        f"/api/rounds/{annotation_round.id}/decisions",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.manager),
    )
    assert manager_decisions.status_code == 200, manager_decisions.text
    assert [entry["id"] for entry in manager_decisions.json()] == [
        first.id,
        revised.id,
        reviewer_decision.id,
    ]

    filtered_decisions = client.get(
        f"/api/rounds/{annotation_round.id}/decisions",
        params={"project_id": project_id, "annotator_user_id": reviewer.id},
        headers=_headers(workflow_scope.manager),
    )
    assert filtered_decisions.status_code == 200, filtered_decisions.text
    assert [entry["id"] for entry in filtered_decisions.json()] == [reviewer_decision.id]

    manager_submissions = client.get(
        f"/api/rounds/{annotation_round.id}/submissions",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.manager),
    )
    assert manager_submissions.status_code == 200, manager_submissions.text
    assert {entry["id"] for entry in manager_submissions.json()} == {
        annotator_submission.id,
        reviewer_submission.id,
    }


def test_open_round_history_is_available_to_workspace_annotators(
    db, client, workflow_scope, workflow_data
):
    project_id = workflow_scope.project.id
    annotation_round = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=project_id,
            name="Open annotation history",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            open_to_all_annotators=True,
        ),
        workflow_scope.manager,
    )
    service.transition_annotation_round(db, project_id, annotation_round.id, "open")
    round_item = service.list_round_items(db, project_id, annotation_round.id)[0]
    decision = service.create_annotation_decision(
        db,
        schemas.AnnotationDecisionCreate(
            project_id=project_id,
            round_item_id=round_item.id,
            output={"label": "positive"},
        ),
        workflow_scope.trainer,
    )
    submission = service.create_round_submission(
        db,
        schemas.RoundSubmissionCreate(
            project_id=project_id,
            annotation_round_id=annotation_round.id,
            decision_ids=[decision.id],
        ),
        workflow_scope.trainer,
    )

    decisions = client.get(
        f"/api/rounds/{annotation_round.id}/decisions",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.trainer),
    )
    submissions = client.get(
        f"/api/rounds/{annotation_round.id}/submissions",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.trainer),
    )
    assert decisions.status_code == 200, decisions.text
    assert [entry["id"] for entry in decisions.json()] == [decision.id]
    assert submissions.status_code == 200, submissions.text
    assert [entry["id"] for entry in submissions.json()] == [submission.id]


def test_direct_round_resolver_derives_project_and_enforces_assignment(
    db,
    client,
    workflow_scope,
    workflow_data,
):
    project_id = workflow_scope.project.id
    reviewer = _user(db, "workflow-direct-round-reviewer")
    workspace_service.add_member(
        db,
        workflow_scope.workspace.id,
        reviewer.id,
        role="annotator",
    )
    assigned = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=project_id,
            name="Direct assigned round",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=[workflow_scope.annotator.id],
        ),
        workflow_scope.manager,
    )
    open_to_all = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=project_id,
            name="Direct open round",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            open_to_all_annotators=True,
        ),
        workflow_scope.manager,
    )
    restricted = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=project_id,
            name="Direct restricted round",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=[workflow_scope.annotator.id],
        ),
        workflow_scope.manager,
    )
    for annotation_round in (assigned, open_to_all, restricted):
        service.transition_annotation_round(
            db,
            project_id,
            annotation_round.id,
            "open",
        )
    other_workspace = workspace_service.create_team_workspace(
        db,
        workflow_scope.outsider,
        name="Other round workspace",
    )
    db.commit()

    assigned_response = client.get(
        f"/api/rounds/{assigned.id}/work-context",
        headers=_headers(workflow_scope.annotator),
    )
    assert assigned_response.status_code == 200, assigned_response.text
    assigned_context = assigned_response.json()
    assert assigned_context["round"]["id"] == assigned.id
    assert assigned_context["round"]["project_id"] == project_id
    assert assigned_context["project"] == {
        "id": project_id,
        "name": workflow_scope.project.name,
    }
    assert assigned_context["task_version"]["id"] == workflow_data.task_version.id
    assert assigned_context["task"] == {
        "id": workflow_data.task.id,
        "key": "sentiment",
        "name": "Sentiment",
    }
    assert assigned_context["cycle"] is None
    assert assigned_context["guideline"] is None
    assert assigned_context["round"]["feedback_available"] is False
    assert not {
        "annotator_user_ids",
        "cycle_id",
        "feedback_set_version_id",
        "guideline_revision_id",
        "open_to_all_annotators",
        "parent_round_id",
        "reason",
        "reannotation_mode",
        "selection_set_version_id",
    }.intersection(assigned_context["round"])
    full_round = client.get(
        f"/api/rounds/{assigned.id}",
        headers=_headers(workflow_scope.annotator),
    )
    assert full_round.status_code == 403

    open_response = client.get(
        f"/api/rounds/{open_to_all.id}/work-context",
        headers=_headers(reviewer),
    )
    assert open_response.status_code == 200, open_response.text
    assert open_response.json()["round"]["project_id"] == project_id

    unassigned_response = client.get(
        f"/api/rounds/{restricted.id}/work-context",
        headers=_headers(reviewer),
    )
    assert unassigned_response.status_code == 403
    assert "not assigned" in unassigned_response.json()["detail"]

    cross_workspace = client.get(
        f"/api/rounds/{assigned.id}/work-context",
        headers=_headers(workflow_scope.outsider),
    )
    assert cross_workspace.status_code == 403

    mismatched_workspace = client.get(
        f"/api/rounds/{assigned.id}/work-context",
        params={"workspace_id": other_workspace.id},
        headers=_headers(workflow_scope.annotator),
    )
    assert mismatched_workspace.status_code == 404

    missing = client.get(
        "/api/rounds/999999/work-context",
        headers=_headers(workflow_scope.annotator),
    )
    assert missing.status_code == 404

    manager_round = client.get(
        f"/api/rounds/{assigned.id}",
        headers=_headers(workflow_scope.manager),
    )
    assert manager_round.status_code == 200, manager_round.text
    assert manager_round.json()["annotator_user_ids"] == [workflow_scope.annotator.id]

    annotator_queue = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/my-work/rounds",
        headers=_headers(workflow_scope.annotator),
    )
    assert annotator_queue.status_code == 200, annotator_queue.text
    annotator_queue_ids = {item["round"]["id"] for item in annotator_queue.json()}
    assert annotator_queue_ids == {
        assigned.id,
        open_to_all.id,
        restricted.id,
    }
    assert all(
        item["project"]
        == {
            "id": project_id,
            "name": workflow_scope.project.name,
        }
        for item in annotator_queue.json()
    )
    assert all(
        not {
            "annotator_user_ids",
            "feedback_set_version_id",
            "selection_set_version_id",
        }.intersection(item["round"])
        for item in annotator_queue.json()
    )

    reviewer_queue = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/my-work/rounds",
        headers=_headers(reviewer),
    )
    assert reviewer_queue.status_code == 200, reviewer_queue.text
    assert {item["round"]["id"] for item in reviewer_queue.json()} == {open_to_all.id}

    manager_queue = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/my-work/rounds",
        headers=_headers(workflow_scope.manager),
    )
    assert manager_queue.status_code == 200, manager_queue.text
    assert {item["round"]["id"] for item in manager_queue.json()} == {open_to_all.id}

    cross_workspace_queue = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/my-work/rounds",
        headers=_headers(workflow_scope.outsider),
    )
    assert cross_workspace_queue.status_code == 403


def test_public_dataset_training_launch_is_independent_of_annotation(
    db, client, workflow_scope, workflow_data
):
    project_id = workflow_scope.project.id
    labels = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project_id,
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            name="source-labels",
            source_kind="imported",
            labels={"train-1": {"label": "positive"}},
        ),
        workflow_scope.trainer,
    )
    training_dataset = service.create_training_dataset_version(
        db,
        schemas.TrainingDatasetVersionCreate(
            project_id=project_id,
            name="public-reviews-train-v1",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            label_set_version_ids=[labels.id],
            split_map_id=workflow_data.split_map.id,
        ),
        workflow_scope.trainer,
    )
    registered = service.create_registered_model(
        db,
        schemas.RegisteredModelCreate(
            project_id=project_id,
            name="review-sentiment-linear",
        ),
        workflow_scope.trainer,
    )
    bind_url = f"/api/training-recipes/trusted/tfidf_logistic_regression?project_id={project_id}"
    bound = client.post(bind_url, headers=_headers(workflow_scope.trainer))
    repeated_binding = client.post(bind_url, headers=_headers(workflow_scope.trainer))
    assert bound.status_code == 200, bound.text
    assert repeated_binding.status_code == 200, repeated_binding.text
    assert repeated_binding.json()["id"] == bound.json()["id"]
    recipe_version = db.get(
        models.TrainingRecipeVersion,
        bound.json()["id"],
    )
    assert recipe_version is not None
    assert recipe_version.trainer_plugin_key == "sklearn_tfidf"
    image_digest = "sha256:" + "b" * 64
    environment = service.create_execution_environment(
        db,
        schemas.ExecutionEnvironmentCreate(
            project_id=project_id,
            name="CPU worker",
            environment_class="classical-cpu",
            image_digest=image_digest,
        ),
        workflow_scope.manager,
    )
    service.verify_execution_environment(
        db,
        project_id,
        environment.id,
        schemas.EnvironmentVerification(
            status="available",
            verification_report=_runtime_report("classical-cpu", image_digest),
        ),
    )
    storage = service.create_storage_policy(
        db,
        schemas.StoragePolicyCreate(
            project_id=project_id,
            name="Workspace artifacts",
            backend="minio",
            artifact_prefix=artifact_service.workspace_blob_prefix(workflow_scope.workspace.id),
            encryption={"mode": "none"},
            is_default=True,
        ),
        workflow_scope.manager,
    )
    launch = schemas.TrainingRunCreate(
        project_id=project_id,
        registered_model_id=registered.id,
        task_version_id=workflow_data.task_version.id,
        training_dataset_version_id=training_dataset.id,
        recipe_version_id=recipe_version.id,
        environment_id=environment.id,
        storage_policy_id=storage.id,
        idempotency_key="public-linear-001",
        evaluation_plan={"splits": ["validation", "test"]},
        config={
            "fields": {
                "input_field": "text",
                "target_field": "label",
            }
        },
    )
    first = service.create_training_run(db, launch, workflow_scope.trainer)
    repeated = service.create_training_run(db, launch, workflow_scope.trainer)
    assert repeated.id == first.id
    assert first.status == "queued"
    assert first.runtime_snapshot["environment_class"] == "classical-cpu"

    run_count = db.query(models.TrainingRun).count()
    reservation_count = db.query(ArtifactStorageReservation).count()
    with pytest.raises(
        ValidationError,
        match="Parent model version launches are not supported",
    ):
        service.create_training_run(
            db,
            launch.model_copy(
                update={
                    "idempotency_key": "public-linear-parent",
                    "parent_model_version_id": 999999,
                }
            ),
            workflow_scope.trainer,
        )
    assert db.query(models.TrainingRun).count() == run_count
    assert db.query(ArtifactStorageReservation).count() == reservation_count
    with pytest.raises(ValidationError, match="credentials or secrets"):
        service.create_training_run(
            db,
            launch.model_copy(
                update={
                    "idempotency_key": "public-linear-secret",
                    "config": {
                        **launch.config,
                        "api_key": "must-not-be-persisted",
                    },
                }
            ),
            workflow_scope.trainer,
        )
    assert db.query(models.TrainingRun).count() == run_count
    assert db.query(ArtifactStorageReservation).count() == reservation_count

    undeclared_recipe = service.create_training_recipe(
        db,
        schemas.TrainingRecipeCreate(
            project_id=project_id,
            key="undeclared-linear",
            name="Undeclared linear trainer",
        ),
        workflow_scope.manager,
    )
    undeclared_recipe_version = service.create_training_recipe_version(
        db,
        schemas.TrainingRecipeVersionCreate(
            project_id=project_id,
            training_recipe_id=undeclared_recipe.id,
            trainer_plugin_key="sklearn_tfidf",
            trainer_plugin_version="1",
            compatible_task_kinds=["classification"],
            environment_class="classical-cpu",
        ),
        workflow_scope.manager,
    )
    with pytest.raises(
        ValidationError,
        match="registered trusted worker contract",
    ):
        service.create_training_run(
            db,
            launch.model_copy(
                update={
                    "recipe_version_id": undeclared_recipe_version.id,
                    "idempotency_key": "public-linear-undeclared",
                }
            ),
            workflow_scope.trainer,
        )

    reservation = db.get(
        ArtifactStorageReservation,
        first.artifact_reservation_id,
    )
    assert reservation is not None
    assert reservation.status == "active"
    assert reservation.owner_type == "training_run"
    quota_snapshot = artifact_quota.workspace_artifact_quota_snapshot(
        db,
        workspace_id=workflow_scope.project.workspace_id,
    )
    assert quota_snapshot.reserved_bytes == reservation.reserved_bytes

    artifact_quota.set_workspace_artifact_quota(
        db,
        workspace_id=workflow_scope.project.workspace_id,
        limit_bytes=(quota_snapshot.used_bytes + quota_snapshot.reserved_bytes + 10),
        reservation_ttl_seconds=None,
        actor_user_id=workflow_scope.manager.id,
    )
    db.commit()
    with pytest.raises(ConflictError, match="10 bytes available"):
        service.create_training_run(
            db,
            launch.model_copy(
                update={
                    "idempotency_key": "public-linear-over-quota",
                    "artifact_reservation_bytes": 11,
                }
            ),
            workflow_scope.trainer,
        )
    db.rollback()

    with pytest.raises(ConflictError, match="idempotency key"):
        service.create_training_run(
            db,
            launch.model_copy(update={"seed": 7}),
            workflow_scope.trainer,
        )

    forbidden = client.get(
        "/api/datasets",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.outsider),
    )
    assert forbidden.status_code == 403
    annotator_create = client.post(
        "/api/datasets",
        json={
            "project_id": project_id,
            "name": "Not allowed",
            "source_type": "upload",
        },
        headers=_headers(workflow_scope.annotator),
    )
    assert annotator_create.status_code == 403

    trainer_create = client.post(
        "/api/datasets",
        json={
            "project_id": project_id,
            "name": "Trainer upload",
            "source_type": "upload",
        },
        headers=_headers(workflow_scope.trainer),
    )
    assert trainer_create.status_code == 201, trainer_create.text
    assert trainer_create.json()["name"] == "Trainer upload"
    cancelled = service.transition_training_run(
        db,
        project_id,
        first.id,
        schemas.TrainingRunTransition(status="cancelled"),
    )
    assert cancelled.status == "cancelled"
    db.refresh(reservation)
    assert reservation.status == "released"
    assert reservation.release_reason == "Training run ended as cancelled"
    assert (
        artifact_quota.workspace_artifact_quota_snapshot(
            db,
            workspace_id=workflow_scope.project.workspace_id,
        ).reserved_bytes
        == 0
    )


def test_workspace_training_aggregates_are_enriched_isolated_and_paginated(
    db,
    client,
    object_storage,
    workflow_scope,
    workflow_data,
):
    project_id = workflow_scope.project.id
    labels = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project_id,
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            name="aggregate-source-labels",
            source_kind="imported",
            labels={"train-1": {"label": "positive"}},
        ),
        workflow_scope.trainer,
    )
    training_dataset = service.create_training_dataset_version(
        db,
        schemas.TrainingDatasetVersionCreate(
            project_id=project_id,
            name="aggregate-training-data",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            label_set_version_ids=[labels.id],
            split_map_id=workflow_data.split_map.id,
        ),
        workflow_scope.trainer,
    )
    registered = service.create_registered_model(
        db,
        schemas.RegisteredModelCreate(
            project_id=project_id,
            name="aggregate-linear-model",
        ),
        workflow_scope.trainer,
    )
    recipe_version = service.ensure_trusted_training_recipe_version(
        db,
        project_id,
        "tfidf_logistic_regression",
        workflow_scope.trainer,
    )
    image_digest = "sha256:" + "d" * 64
    environment = service.create_execution_environment(
        db,
        schemas.ExecutionEnvironmentCreate(
            project_id=project_id,
            name="Aggregate CPU",
            environment_class="classical-cpu",
            image_digest=image_digest,
        ),
        workflow_scope.manager,
    )
    service.verify_execution_environment(
        db,
        project_id,
        environment.id,
        schemas.EnvironmentVerification(
            status="available",
            verification_report=_runtime_report("classical-cpu", image_digest),
        ),
    )
    storage = service.create_storage_policy(
        db,
        schemas.StoragePolicyCreate(
            project_id=project_id,
            name="Aggregate artifacts",
            backend="minio",
            artifact_prefix=artifact_service.workspace_blob_prefix(workflow_scope.workspace.id),
            encryption={"mode": "none"},
            is_default=True,
        ),
        workflow_scope.manager,
    )
    model_version = service.create_model_version(
        db,
        schemas.ModelVersionCreate(
            project_id=project_id,
            registered_model_id=registered.id,
            task_version_id=workflow_data.task_version.id,
            training_dataset_version_id=training_dataset.id,
            family="conventional_ml",
            framework="scikit-learn",
            base_model={},
            training_method="full",
            recipe_key="tfidf_logistic_regression",
            recipe_version=recipe_version.trainer_plugin_version,
            runtime_digest="a" * 64,
            code_digest="b" * 64,
        ),
        workflow_scope.trainer,
    )
    training_run = service.create_training_run(
        db,
        schemas.TrainingRunCreate(
            project_id=project_id,
            registered_model_id=registered.id,
            task_version_id=workflow_data.task_version.id,
            training_dataset_version_id=training_dataset.id,
            recipe_version_id=recipe_version.id,
            environment_id=environment.id,
            storage_policy_id=storage.id,
            idempotency_key="workspace-aggregate-run",
            config={
                "fields": {
                    "input_field": "text",
                    "target_field": "label",
                }
            },
        ),
        workflow_scope.trainer,
    )
    base_package = artifact_service.publish_artifact_package(
        db,
        object_storage,
        project_id=project_id,
        data=ArtifactPackageCreate(
            package_kind="base_model",
            package_format="safetensors",
            display_name="Pinned biomedical encoder",
            model_family="deep_learning",
            model_type="transformer_encoder",
            readiness="ready",
            deployable=True,
            license_info={"name": "apache-2.0"},
        ),
        files=[
            artifact_service.PackageFileUpload(
                relative_path="model.safetensors",
                source=b"pinned-biomedical-encoder",
                role="model_weights",
            )
        ],
        actor_user_id=workflow_scope.manager.id,
    )
    base_asset = BaseModelAsset(
        project_id=project_id,
        workspace_id=workflow_scope.workspace.id,
        package_id=base_package.id,
        provider="hugging_face",
        source_model_id="research/biomedical-encoder",
        exact_revision="abc123-pinned-revision",
        display_name="Pinned biomedical encoder",
        model_family="deep_learning",
        model_type="transformer_encoder",
        license_name="apache-2.0",
        access_mode="execution_only",
        metadata_={},
        created_by_user_id=workflow_scope.manager.id,
    )
    db.add(base_asset)
    db.flush()
    db.add(
        BaseModelAssetState(
            base_model_asset_id=base_asset.id,
            readiness="ready",
            updated_by_user_id=workflow_scope.manager.id,
        )
    )
    training_run.config = {
        **training_run.config,
        "base_model_asset_id": base_asset.id,
    }
    training_run.output_model_version_id = model_version.id
    standalone = Project(
        name="Standalone training",
        description="Public data only",
        workspace_id=workflow_scope.workspace.id,
        settings={
            "modules": [
                "data",
                "models",
                "train",
                "activity",
            ]
        },
    )
    db.add(standalone)
    db.commit()

    annotator = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/training-contexts",
        headers=_headers(workflow_scope.annotator),
    )
    assert annotator.status_code == 403

    first_page = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/training-contexts",
        params={"limit": 1},
        headers=_headers(workflow_scope.trainer),
    )
    assert first_page.status_code == 200, first_page.text
    assert first_page.json()["items"] == [
        {
            "project_id": standalone.id,
            "project_name": "Standalone training",
            "project_description": "Public data only",
            "training_only": True,
            "effective_modules": ["data", "train", "models", "activity"],
            "task_version_count": 0,
            "training_dataset_version_count": 0,
            "environment_count": 0,
            "available_environment_count": 0,
            "storage_policy_count": 0,
        }
    ]
    assert first_page.json()["next_cursor"] == standalone.id
    second_page = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/training-contexts",
        params={"limit": 1, "cursor": standalone.id},
        headers=_headers(workflow_scope.trainer),
    )
    assert second_page.status_code == 200, second_page.text
    assert second_page.json()["items"][0]["project_id"] == project_id
    assert second_page.json()["items"][0]["training_only"] is False
    assert second_page.json()["next_cursor"] is None

    runs = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/training-runs",
        params={
            "project_id": project_id,
            "status": "queued",
            "creator_user_id": workflow_scope.trainer.id,
            "task_version_id": workflow_data.task_version.id,
            "family": "conventional_ml",
        },
        headers=_headers(workflow_scope.trainer),
    )
    assert runs.status_code == 200, runs.text
    assert runs.json()["next_cursor"] is None
    assert len(runs.json()["items"]) == 1
    run = runs.json()["items"][0]
    assert run["id"] == training_run.id
    assert run["project_name"] == workflow_scope.project.name
    assert run["model_name"] == registered.name
    assert run["family"] == "conventional_ml"
    assert run["framework"] == "scikit-learn"
    assert run["base_model"] == {
        "asset_id": base_asset.id,
        "display_name": "Pinned biomedical encoder",
        "provider": "hugging_face",
        "source_model_id": "research/biomedical-encoder",
        "source_identity": "hugging_face / research/biomedical-encoder",
        "exact_revision": "abc123-pinned-revision",
    }
    assert run["training_method"] == "full"
    assert run["task_name"] == "Sentiment"
    assert run["training_dataset_name"] == "aggregate-training-data"
    assert run["runtime_name"] == "Aggregate CPU"
    assert run["storage_policy_name"] == "Aggregate artifacts"
    assert run["creator_username"] == workflow_scope.trainer.username

    registry = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/models",
        params={
            "project_id": project_id,
            "status": "active",
            "creator_user_id": workflow_scope.trainer.id,
            "task_version_id": workflow_data.task_version.id,
            "family": "conventional_ml",
        },
        headers=_headers(workflow_scope.trainer),
    )
    assert registry.status_code == 200, registry.text
    assert registry.json()["next_cursor"] is None
    assert len(registry.json()["items"]) == 1
    model = registry.json()["items"][0]
    assert model["id"] == registered.id
    assert model["project_name"] == workflow_scope.project.name
    assert model["latest_version"]["id"] == model_version.id
    assert model["latest_version"]["training_run_id"] == training_run.id
    assert model["latest_version"]["source_dataset_version_id"] == workflow_data.dataset_version.id
    assert (
        model["latest_version"]["source_dataset_version_number"]
        == workflow_data.dataset_version.version_number
    )
    assert model["latest_version"]["runtime_id"] == environment.id
    assert model["latest_version"]["runtime_name"] == "Aggregate CPU"
    assert model["latest_version"]["storage_policy_id"] == storage.id
    assert model["latest_version"]["storage_policy_name"] == "Aggregate artifacts"
    assert model["latest_version"]["creator_username"] == workflow_scope.trainer.username
    assert model["task_name"] == "Sentiment"
    assert model["training_dataset_name"] == "aggregate-training-data"
    assert model["creator_username"] == workflow_scope.trainer.username

    versions_response = client.get(
        "/api/models/versions",
        params={
            "project_id": project_id,
            "registered_model_id": registered.id,
        },
        headers=_headers(workflow_scope.trainer),
    )
    assert versions_response.status_code == 200, versions_response.text
    version_payload = versions_response.json()[0]
    assert version_payload["source_dataset_version_id"] == workflow_data.dataset_version.id
    assert (
        version_payload["source_dataset_version_number"]
        == workflow_data.dataset_version.version_number
    )
    assert version_payload["runtime_name"] == "Aggregate CPU"
    assert version_payload["storage_policy_name"] == "Aggregate artifacts"
    assert version_payload["creator_username"] == workflow_scope.trainer.username

    history_model = service.create_registered_model(
        db,
        schemas.RegisteredModelCreate(
            project_id=project_id,
            name="aggregate-version-filter-model",
        ),
        workflow_scope.trainer,
    )
    old_transformer_version = service.create_model_version(
        db,
        schemas.ModelVersionCreate(
            project_id=project_id,
            registered_model_id=history_model.id,
            task_version_id=workflow_data.task_version.id,
            training_dataset_version_id=training_dataset.id,
            family="transformer",
            framework="transformers",
            base_model={"name": "old-transformer"},
            training_method="full_finetune",
            recipe_key="transformer_classification",
            recipe_version="1",
            runtime_digest="c" * 64,
            code_digest="d" * 64,
        ),
        workflow_scope.trainer,
    )
    latest_linear_version = service.create_model_version(
        db,
        schemas.ModelVersionCreate(
            project_id=project_id,
            registered_model_id=history_model.id,
            task_version_id=workflow_data.task_version.id,
            training_dataset_version_id=training_dataset.id,
            parent_version_id=old_transformer_version.id,
            family="linear",
            framework="scikit-learn",
            base_model={"name": "tfidf"},
            training_method="full_fit",
            recipe_key="tfidf_logistic_regression",
            recipe_version="1",
            runtime_digest="e" * 64,
            code_digest="f" * 64,
        ),
        workflow_scope.trainer,
    )
    historical_family = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/models",
        params={"project_id": project_id, "family": "transformer"},
        headers=_headers(workflow_scope.trainer),
    )
    assert historical_family.status_code == 200, historical_family.text
    assert history_model.id not in {item["id"] for item in historical_family.json()["items"]}
    latest_family = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/models",
        params={"project_id": project_id, "family": "linear"},
        headers=_headers(workflow_scope.trainer),
    )
    assert latest_family.status_code == 200, latest_family.text
    history_payload = next(
        item for item in latest_family.json()["items"] if item["id"] == history_model.id
    )
    assert history_payload["latest_version"]["id"] == latest_linear_version.id
    assert history_payload["latest_version"]["family"] == "linear"

    excluded_project = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/models",
        params={"project_id": standalone.id},
        headers=_headers(workflow_scope.trainer),
    )
    assert excluded_project.status_code == 200
    assert excluded_project.json() == {"items": [], "next_cursor": None}

    preserved_runtime = {
        **training_run.runtime_snapshot,
        "worker_observation": "keep-runtime",
    }
    preserved_storage = {
        **training_run.storage_snapshot,
        "worker_observation": "keep-storage",
    }
    training_run.status = "running"
    training_run.output_model_version_id = model_version.id
    training_run.runtime_snapshot = preserved_runtime
    training_run.storage_snapshot = preserved_storage
    training_run.failure_code = "prior-worker-note"
    training_run.failure_reason = "Preserve existing execution diagnostics"
    db.commit()

    forged_cancel = client.post(
        f"/api/training-runs/{training_run.id}/transition",
        params={"project_id": project_id},
        json={
            "status": "cancelled",
            "output_model_version_id": 999999,
            "runtime_snapshot": {"forged": True},
            "storage_snapshot": {"forged": True},
            "failure_code": "forged",
            "failure_reason": "forged",
        },
        headers=_headers(workflow_scope.trainer),
    )
    assert forged_cancel.status_code == 422
    db.refresh(training_run)
    assert training_run.status == "running"
    assert training_run.output_model_version_id == model_version.id
    assert training_run.runtime_snapshot == preserved_runtime
    assert training_run.storage_snapshot == preserved_storage
    assert training_run.failure_code == "prior-worker-note"
    assert training_run.failure_reason == "Preserve existing execution diagnostics"

    other_trainer = _user(db, "workflow-other-trainer")
    workspace_service.add_member(
        db,
        workflow_scope.workspace.id,
        other_trainer.id,
        role="trainer",
    )
    db.commit()
    forbidden_cancel = client.post(
        f"/api/training-runs/{training_run.id}/transition",
        params={"project_id": project_id},
        json={"status": "cancelled"},
        headers=_headers(other_trainer),
    )
    assert forbidden_cancel.status_code == 403
    assert "only their own" in forbidden_cancel.json()["detail"]
    other_manager = _user(db, "workflow-other-manager")
    workspace_service.add_member(
        db,
        workflow_scope.workspace.id,
        other_manager.id,
        role="manager",
    )
    db.commit()
    manager_cancel = client.post(
        f"/api/training-runs/{training_run.id}/transition",
        params={"project_id": project_id},
        json={"status": "cancelled"},
        headers=_headers(other_manager),
    )
    assert manager_cancel.status_code == 403
    assert "only their own" in manager_cancel.json()["detail"]
    administrator_cancel = client.post(
        f"/api/training-runs/{training_run.id}/transition",
        params={"project_id": project_id},
        json={"status": "cancelled"},
        headers=_headers(workflow_scope.manager),
    )
    assert administrator_cancel.status_code == 403
    assert "only their own" in administrator_cancel.json()["detail"]
    creator_cancel = client.post(
        f"/api/training-runs/{training_run.id}/transition",
        params={"project_id": project_id},
        json={"status": "cancelled"},
        headers=_headers(workflow_scope.trainer),
    )
    assert creator_cancel.status_code == 200, creator_cancel.text
    cancelled_payload = creator_cancel.json()
    assert cancelled_payload["status"] == "cancelled"
    assert cancelled_payload["output_model_version_id"] == model_version.id
    assert cancelled_payload["runtime_snapshot"] == preserved_runtime
    assert cancelled_payload["storage_snapshot"] == preserved_storage
    assert cancelled_payload["failure_code"] == "prior-worker-note"
    assert cancelled_payload["failure_reason"] == "Preserve existing execution diagnostics"


def test_model_version_reads_ignore_cross_project_relationships(
    db,
    client,
    workflow_scope,
    workflow_data,
):
    project_id = workflow_scope.project.id
    registered = service.create_registered_model(
        db,
        schemas.RegisteredModelCreate(
            project_id=project_id,
            name="cross-project-defense",
        ),
        workflow_scope.trainer,
    )
    foreign_project = Project(
        name="Foreign model project",
        workspace_id=workflow_scope.workspace.id,
        settings={
            "modules": [
                "data",
                "models",
                "activity",
            ]
        },
    )
    db.add(foreign_project)
    db.flush()
    forged_version = models.ModelVersion(
        project_id=foreign_project.id,
        registered_model_id=registered.id,
        version_number=1,
        task_version_id=workflow_data.task_version.id,
        training_dataset_version_id=None,
        family="foreign-family",
        framework="foreign-framework",
        base_model={"secret": "cross-project"},
        training_method="foreign-method",
        recipe_key="foreign-recipe",
        recipe_version="1",
        parameters={},
        metrics={},
        runtime_digest="c" * 64,
        code_digest="d" * 64,
        seed=42,
        content_hash="e" * 64,
        created_by_user_id=workflow_scope.manager.id,
    )
    db.add(forged_version)
    db.commit()

    project_versions = client.get(
        "/api/models/versions",
        params={
            "project_id": project_id,
            "registered_model_id": registered.id,
        },
        headers=_headers(workflow_scope.trainer),
    )
    assert project_versions.status_code == 200, project_versions.text
    assert project_versions.json() == []

    workspace_models = client.get(
        f"/api/workspaces/{workflow_scope.workspace.id}/models",
        headers=_headers(workflow_scope.trainer),
    )
    assert workspace_models.status_code == 200, workspace_models.text
    target = next(item for item in workspace_models.json()["items"] if item["id"] == registered.id)
    assert target["project_id"] == project_id
    assert target["latest_version"] is None
    assert "foreign-family" not in str(target)
    assert "cross-project" not in str(target["latest_version"])


def test_workspace_training_aggregates_enforce_tenancy_and_capability(
    db,
    client,
    workflow_scope,
):
    path = f"/api/workspaces/{workflow_scope.workspace.id}/training-contexts"
    outsider = client.get(path, headers=_headers(workflow_scope.outsider))
    assert outsider.status_code == 403

    capability_service.set_capability(
        db,
        workflow_scope.workspace.id,
        preset="annotate",
        actor_user_id=workflow_scope.manager.id,
    )
    db.commit()
    unavailable = client.get(path, headers=_headers(workflow_scope.manager))
    assert unavailable.status_code == 403
    assert "Capability 'training'" in unavailable.json()["detail"]


def test_training_and_model_reads_require_trainer_role(
    client,
    workflow_scope,
):
    project_id = workflow_scope.project.id
    reads = [
        ("/api/datasets/training-versions", {"project_id": project_id}),
        (
            "/api/datasets/training-versions/999/labels",
            {"project_id": project_id},
        ),
        ("/api/models", {"project_id": project_id}),
        (
            "/api/models/versions",
            {"project_id": project_id, "registered_model_id": 999},
        ),
        (
            "/api/models/999/versions/999/evaluations",
            {"project_id": project_id},
        ),
        ("/api/training-recipes", {"project_id": project_id}),
        (
            "/api/training-recipes/versions",
            {"project_id": project_id, "training_recipe_id": 999},
        ),
        ("/api/environments", {"project_id": project_id}),
        ("/api/storage/policies", {"project_id": project_id}),
        ("/api/training-runs", {"project_id": project_id}),
        ("/api/training-runs/999", {"project_id": project_id}),
        (
            "/api/training-runs/999/evaluations",
            {"project_id": project_id},
        ),
        ("/api/evaluations/999", {"project_id": project_id}),
    ]
    for path, params in reads:
        response = client.get(
            path,
            params=params,
            headers=_headers(workflow_scope.annotator),
        )
        assert response.status_code == 403, (path, response.text)


def test_project_collections_and_planned_modules_are_not_annotator_enumerable(
    client,
    workflow_scope,
    workflow_data,
):
    project_id = workflow_scope.project.id
    trainer_reads = [
        (f"/api/projects/{project_id}/modules", {}),
        ("/api/tasks", {"project_id": project_id}),
        ("/api/tasks/versions", {"project_id": project_id}),
        ("/api/datasets", {"project_id": project_id}),
        (
            "/api/datasets/versions",
            {
                "project_id": project_id,
                "dataset_id": workflow_data.dataset.id,
            },
        ),
    ]
    manager_reads = [
        ("/api/rounds", {"project_id": project_id}),
        ("/api/cycles", {"project_id": project_id}),
        ("/api/selection-runs", {"project_id": project_id}),
        ("/api/feedback-runs", {"project_id": project_id}),
        ("/api/feedback-runs/sets", {"project_id": project_id}),
        ("/api/feedback-runs/events", {"project_id": project_id}),
        ("/api/feedback-runs/review-cases", {"project_id": project_id}),
        ("/api/workflow-guidelines", {"project_id": project_id}),
        (
            "/api/workflow-guidelines/revisions",
            {"project_id": project_id, "guideline_id": 999999},
        ),
        ("/api/workflow-guidelines/proposals", {"project_id": project_id}),
        (
            "/api/workflow-guidelines/impact-evaluations",
            {"project_id": project_id},
        ),
    ]

    for path, params in [*trainer_reads, *manager_reads]:
        response = client.get(
            path,
            params=params,
            headers=_headers(workflow_scope.annotator),
        )
        assert response.status_code == 403, (path, response.text)

    for path, params in trainer_reads:
        response = client.get(
            path,
            params=params,
            headers=_headers(workflow_scope.trainer),
        )
        assert response.status_code == 200, (path, response.text)

    for path, params in manager_reads:
        response = client.get(
            path,
            params=params,
            headers=_headers(workflow_scope.trainer),
        )
        assert response.status_code == 403, (path, response.text)


def test_server_side_training_composition_is_grouped_deterministic_and_compact(
    db,
    client,
    workflow_scope,
    workflow_data,
):
    project_id = workflow_scope.project.id
    task_version = service.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=project_id,
            task_definition_id=workflow_data.task.id,
            task_kind="classification",
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
            output_schema={
                "type": "string",
                "enum": ["negative", "positive"],
            },
            metrics=["macro_f1"],
            trainer_compatibility=["tfidf_logistic_regression"],
        ),
        workflow_scope.manager,
    )
    dataset_version = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project_id,
            dataset_id=workflow_data.dataset.id,
            source_uri="upload://compose-source",
            source_revision="server-side-compose-v1",
            source_format="jsonl",
            data_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "label": {"type": "string"},
                },
            },
            provenance={"test": "server-side-compose"},
            license_info={"name": "apache-2.0"},
            items=[
                schemas.DatasetItemCreate(
                    stable_key="a-1",
                    group_key="group-a",
                    payload={"text": "A one", "label": "positive"},
                ),
                schemas.DatasetItemCreate(
                    stable_key="a-2",
                    group_key="group-a",
                    payload={"text": "A two", "label": "positive"},
                ),
                schemas.DatasetItemCreate(
                    stable_key="b-1",
                    group_key="group-b",
                    payload={"text": "B", "label": "negative"},
                ),
                schemas.DatasetItemCreate(
                    stable_key="c-1",
                    group_key="group-c",
                    payload={"text": "C", "label": "positive"},
                ),
                schemas.DatasetItemCreate(
                    stable_key="d-1",
                    group_key="group-d",
                    payload={"text": "D", "label": "negative"},
                ),
            ],
        ),
        workflow_scope.manager,
    )
    payload = {
        "project_id": project_id,
        "name": "Server composed reviews",
        "dataset_version_id": dataset_version.id,
        "task_version_id": task_version.id,
        "input_field": "text",
        "label_field": "label",
        "train_percent": 50,
        "validation_percent": 25,
        "seed": 17,
    }
    composed = client.post(
        "/api/datasets/training-versions/compose",
        json=payload,
        headers=_headers(workflow_scope.trainer),
    )
    assert composed.status_code == 201, composed.text
    body = composed.json()
    assert set(body) == {
        "training_dataset_version",
        "label_set_version_id",
        "split_map_id",
        "split_counts",
        "group_count",
    }
    assert body["group_count"] == 4
    assert all(body["split_counts"][split] > 0 for split in ("train", "validation", "test"))

    def nested_keys(value):
        if isinstance(value, dict):
            return set(value).union(*(nested_keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(nested_keys(item) for item in value))
        return set()

    assert "labels" not in nested_keys(body)
    assert "assignments" not in nested_keys(body)
    label_set = db.get(models.LabelSetVersion, body["label_set_version_id"])
    split_map = db.get(models.SplitMap, body["split_map_id"])
    training_dataset = db.get(
        models.TrainingDatasetVersion,
        body["training_dataset_version"]["id"],
    )
    assert label_set is not None
    assert label_set.source_kind == "imported"
    assert label_set.label_count == 5
    assert split_map is not None
    assert split_map.protected_splits == ["test"]
    assert split_map.assignments["a-1"] == split_map.assignments["a-2"]
    assert training_dataset is not None
    assert training_dataset.preprocessing == {
        "input_field": "text",
        "target_field": "label",
    }
    assert training_dataset.label_set_version_ids == [label_set.id]

    repeated = client.post(
        "/api/datasets/training-versions/compose",
        json={**payload, "name": "Server composed reviews copy"},
        headers=_headers(workflow_scope.trainer),
    )
    assert repeated.status_code == 201, repeated.text
    repeated_split = db.get(
        models.SplitMap,
        repeated.json()["split_map_id"],
    )
    assert repeated_split is not None
    assert repeated_split.assignments == split_map.assignments

    forbidden = client.post(
        "/api/datasets/training-versions/compose",
        json={**payload, "name": "Annotator attempt"},
        headers=_headers(workflow_scope.annotator),
    )
    assert forbidden.status_code == 403
    invalid_source = client.post(
        "/api/datasets/training-versions/compose",
        json={
            **payload,
            "name": "Ambiguous labels",
            "label_set_version_id": label_set.id,
        },
        headers=_headers(workflow_scope.trainer),
    )
    assert invalid_source.status_code == 422
    invalid_percentages = client.post(
        "/api/datasets/training-versions/compose",
        json={
            **payload,
            "name": "No test remainder",
            "train_percent": 80,
            "validation_percent": 20,
        },
        headers=_headers(workflow_scope.trainer),
    )
    assert invalid_percentages.status_code == 422


def test_training_dataset_binding_validates_task_input_schema(db, workflow_scope, workflow_data):
    project_id = workflow_scope.project.id
    strict_task = service.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=project_id,
            task_definition_id=workflow_data.task.id,
            task_kind="classification",
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "required": ["label"],
                "properties": {
                    "label": {"enum": ["negative", "positive"]},
                },
            },
            metrics=["f1"],
            trainer_compatibility=["linear-classification"],
        ),
        workflow_scope.manager,
    )
    dataset_version = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project_id,
            dataset_id=workflow_data.dataset.id,
            source_uri="hf://example/reviews",
            source_revision="input-contract-regression",
            source_format="jsonl",
            data_schema={"text": "string"},
            provenance={"registry": "huggingface"},
            license_info={"name": "apache-2.0"},
            items=[
                schemas.DatasetItemCreate(
                    stable_key="bad-train",
                    group_key="input-contract-train",
                    payload={"text": 42},
                ),
                schemas.DatasetItemCreate(
                    stable_key="good-test",
                    group_key="input-contract-test",
                    payload={"text": "Protected holdout"},
                ),
            ],
        ),
        workflow_scope.manager,
    )
    split_map = service.create_split_map(
        db,
        schemas.SplitMapCreate(
            project_id=project_id,
            dataset_version_id=dataset_version.id,
            name="input-contract-splits",
            strategy="grouped",
            assignments={"bad-train": "train", "good-test": "test"},
            protected_splits=["test"],
        ),
        workflow_scope.manager,
    )
    labels = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project_id,
            dataset_version_id=dataset_version.id,
            task_version_id=strict_task.id,
            name="input-contract-labels",
            source_kind="imported",
            labels={"bad-train": {"label": "positive"}},
        ),
        workflow_scope.trainer,
    )

    with pytest.raises(
        ValidationError,
        match=r"Dataset item 'bad-train'.*task input schema",
    ):
        service.create_training_dataset_version(
            db,
            schemas.TrainingDatasetVersionCreate(
                project_id=project_id,
                name="invalid-task-inputs",
                dataset_version_id=dataset_version.id,
                task_version_id=strict_task.id,
                label_set_version_ids=[labels.id],
                split_map_id=split_map.id,
            ),
            workflow_scope.trainer,
        )


def test_guideline_activation_requires_a_closed_passing_pilot(db, workflow_scope, workflow_data):
    project_id = workflow_scope.project.id
    selection_run = service.create_selection_run(
        db,
        schemas.SelectionRunCreate(
            project_id=project_id,
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            split_map_id=workflow_data.split_map.id,
            strategy="random",
        ),
        workflow_scope.trainer,
    )
    selection_set = service.create_selection_set(
        db,
        schemas.SelectionSetCreate(
            project_id=project_id,
            selection_run_id=selection_run.id,
            items=[
                schemas.SelectionItem(
                    dataset_item_id=workflow_data.items["pool-1"].id,
                    rank=1,
                )
            ],
        ),
        workflow_scope.trainer,
    )
    guideline = service.create_guideline(
        db,
        schemas.GuidelineCreate(
            project_id=project_id,
            task_definition_id=workflow_data.task.id,
            name="Sentiment decisions",
        ),
        workflow_scope.manager,
    )
    revision = service.create_guideline_revision(
        db,
        schemas.GuidelineRevisionCreate(
            project_id=project_id,
            guideline_id=guideline.id,
            task_version_id=workflow_data.task_version.id,
            markdown="Choose the label supported by the complete sentence.",
        ),
        workflow_scope.manager,
    )
    service.transition_guideline_revision(
        db,
        project_id,
        revision.id,
        schemas.GuidelineRevisionTransition(status="pilot"),
        workflow_scope.manager,
    )
    with pytest.raises(ConflictError, match="passing pilot"):
        service.transition_guideline_revision(
            db,
            project_id,
            revision.id,
            schemas.GuidelineRevisionTransition(status="active"),
            workflow_scope.manager,
        )

    pilot_round = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=project_id,
            name="Guideline pilot",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            guideline_revision_id=revision.id,
            selection_set_version_id=selection_set.id,
            assistance_policy="blind",
            annotator_user_ids=[workflow_scope.manager.id, workflow_scope.annotator.id],
        ),
        workflow_scope.manager,
    )
    impact = service.create_guideline_impact(
        db,
        schemas.GuidelineImpactCreate(
            project_id=project_id,
            guideline_revision_id=revision.id,
            pilot_round_id=pilot_round.id,
        ),
        workflow_scope.manager,
    )
    with pytest.raises(ConflictError, match="pilot round closes"):
        service.complete_guideline_impact(
            db,
            project_id,
            impact.id,
            schemas.GuidelineImpactComplete(minimum_agreement=0.8),
            workflow_scope.manager,
        )

    service.transition_annotation_round(db, project_id, pilot_round.id, "open")
    pilot_item = service.list_round_items(db, project_id, pilot_round.id)[0]
    for actor in (workflow_scope.manager, workflow_scope.annotator):
        decision = service.create_annotation_decision(
            db,
            schemas.AnnotationDecisionCreate(
                project_id=project_id,
                round_item_id=pilot_item.id,
                output={"label": "positive"},
            ),
            actor,
        )
        service.create_round_submission(
            db,
            schemas.RoundSubmissionCreate(
                project_id=project_id,
                annotation_round_id=pilot_round.id,
                decision_ids=[decision.id],
            ),
            actor,
        )
    service.transition_annotation_round(db, project_id, pilot_round.id, "closed")
    service.complete_guideline_impact(
        db,
        project_id,
        impact.id,
        schemas.GuidelineImpactComplete(minimum_agreement=0.8),
        workflow_scope.manager,
    )
    db.refresh(impact)
    assert impact.passed is True
    assert impact.metrics["coverage"] == 1.0
    assert impact.metrics["exact_agreement"] == 1.0
    active = service.transition_guideline_revision(
        db,
        project_id,
        revision.id,
        schemas.GuidelineRevisionTransition(status="active"),
        workflow_scope.manager,
    )
    assert active.status == "active"
    assert active.approved_by_user_id == workflow_scope.manager.id


def test_training_label_composition_is_explicit_and_deterministic(
    db, client, workflow_scope, workflow_data
):
    project_id = workflow_scope.project.id
    imported = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project_id,
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            name="source-labels",
            source_kind="imported",
            labels={"train-1": "negative", "pool-1": "negative"},
        ),
        workflow_scope.trainer,
    )
    corrected = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project_id,
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            name="human-corrections",
            source_kind="imported",
            labels={"pool-1": "positive"},
        ),
        workflow_scope.trainer,
    )
    excluded = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project_id,
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            name="exclusions",
            source_kind="derived",
            composition_policy="exclude",
            labels={"train-1": True},
        ),
        workflow_scope.trainer,
    )
    training_dataset = service.create_training_dataset_version(
        db,
        schemas.TrainingDatasetVersionCreate(
            project_id=project_id,
            name="composed-v1",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            label_set_version_ids=[imported.id, corrected.id, excluded.id],
            split_map_id=workflow_data.split_map.id,
            composition=[
                {"label_set_version_id": imported.id, "policy": "inherit"},
                {"label_set_version_id": corrected.id, "policy": "replace"},
                {"label_set_version_id": excluded.id, "policy": "exclude"},
            ],
        ),
        workflow_scope.trainer,
    )

    composed = service.compose_training_dataset_labels(
        db,
        project_id,
        training_dataset.id,
    )
    assert composed.labels == {"pool-1": "positive"}
    assert composed.label_count == 1
    response = client.get(
        f"/api/datasets/training-versions/{training_dataset.id}/labels",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.trainer),
    )
    assert response.status_code == 200, response.text
    assert response.json()["labels"] == {"pool-1": "positive"}

    with pytest.raises(ValidationError, match="every declared label set"):
        service.create_training_dataset_version(
            db,
            schemas.TrainingDatasetVersionCreate(
                project_id=project_id,
                name="invalid-composition",
                dataset_version_id=workflow_data.dataset_version.id,
                task_version_id=workflow_data.task_version.id,
                label_set_version_ids=[imported.id, corrected.id],
                split_map_id=workflow_data.split_map.id,
                composition=[
                    {"label_set_version_id": imported.id, "policy": "replace"},
                ],
            ),
            workflow_scope.trainer,
        )


def test_project_modules_are_independent_and_authorized(
    client,
    workflow_scope,
):
    project_id = workflow_scope.project.id
    initial = client.get(
        f"/api/projects/{project_id}/modules",
        headers=_headers(workflow_scope.manager),
    )
    assert initial.status_code == 200, initial.text
    assert "train" in initial.json()["effective"]

    configured = client.patch(
        f"/api/projects/{project_id}/modules",
        json={"selected": ["data", "annotate"]},
        headers=_headers(workflow_scope.manager),
    )
    assert configured.status_code == 200, configured.text
    assert configured.json()["effective"] == ["data", "annotate"]

    data_read = client.get(
        "/api/datasets",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.trainer),
    )
    assert data_read.status_code == 200, data_read.text
    training_read = client.get(
        "/api/training-runs",
        params={"project_id": project_id},
        headers=_headers(workflow_scope.trainer),
    )
    assert training_read.status_code == 403
    assert "required project modules" in training_read.json()["detail"]


def test_learning_cycle_requires_and_records_a_feedback_source(
    client,
    workflow_scope,
    workflow_data,
):
    payload = {
        "project_id": workflow_scope.project.id,
        "name": "Rule-assisted review",
        "task_version_id": workflow_data.task_version.id,
        "source_dataset_version_id": workflow_data.dataset_version.id,
    }
    missing_source = client.post(
        "/api/cycles",
        json=payload,
        headers=_headers(workflow_scope.manager),
    )
    assert missing_source.status_code == 422

    created = client.post(
        "/api/cycles",
        json={
            **payload,
            "feedback_sources": [
                {
                    "producer_type": "rule",
                    "name": "Negation rules v1",
                    "configuration": {"rule_set_revision": "sha256:rules-v1"},
                }
            ],
        },
        headers=_headers(workflow_scope.manager),
    )
    assert created.status_code == 201, created.text
    assert created.json()["feedback_sources"] == [
        {
            "producer_type": "rule",
            "name": "Negation rules v1",
            "provider": None,
            "external_model_id": None,
            "exact_revision": None,
            "configuration": {"rule_set_revision": "sha256:rules-v1"},
            "data_egress_policy": {},
        }
    ]
    matching_run = client.post(
        "/api/feedback-runs",
        json={
            "project_id": workflow_scope.project.id,
            "dataset_version_id": workflow_data.dataset_version.id,
            "task_version_id": workflow_data.task_version.id,
            "cycle_id": created.json()["id"],
            "producer_type": "rule",
        },
        headers=_headers(workflow_scope.manager),
    )
    assert matching_run.status_code == 201, matching_run.text

    undeclared_run = client.post(
        "/api/feedback-runs",
        json={
            "project_id": workflow_scope.project.id,
            "dataset_version_id": workflow_data.dataset_version.id,
            "task_version_id": workflow_data.task_version.id,
            "cycle_id": created.json()["id"],
            "producer_type": "dictionary",
        },
        headers=_headers(workflow_scope.manager),
    )
    assert undeclared_run.status_code == 422


def test_assisted_rounds_require_feedback_and_empty_rounds_cannot_close(
    client,
    workflow_scope,
    workflow_data,
):
    base = {
        "project_id": workflow_scope.project.id,
        "name": "Governed annotation",
        "dataset_version_id": workflow_data.dataset_version.id,
        "task_version_id": workflow_data.task_version.id,
        "reannotation_mode": "full_dataset",
        "open_to_all_annotators": True,
    }
    missing_feedback = client.post(
        "/api/rounds",
        json={**base, "assistance_policy": "reveal_after_first_pass"},
        headers=_headers(workflow_scope.manager),
    )
    assert missing_feedback.status_code == 422

    blind = client.post(
        "/api/rounds",
        json={**base, "name": "Blind governed annotation", "assistance_policy": "blind"},
        headers=_headers(workflow_scope.manager),
    )
    assert blind.status_code == 201, blind.text
    opened = client.post(
        f"/api/rounds/{blind.json()['id']}/transition",
        params={"project_id": workflow_scope.project.id},
        json={"status": "open"},
        headers=_headers(workflow_scope.manager),
    )
    assert opened.status_code == 200, opened.text
    closed = client.post(
        f"/api/rounds/{blind.json()['id']}/transition",
        params={"project_id": workflow_scope.project.id},
        json={"status": "closed"},
        headers=_headers(workflow_scope.manager),
    )
    assert closed.status_code == 422
    assert "cover every item" in closed.json()["detail"]


def test_protected_test_feedback_cannot_feed_guideline_mining(db, workflow_scope, workflow_data):
    project_id = workflow_scope.project.id
    annotation_round = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=project_id,
            name="Full audit",
            dataset_version_id=workflow_data.dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=[workflow_scope.manager.id],
        ),
        workflow_scope.manager,
    )
    test_round_item = next(
        item
        for item in service.list_round_items(db, project_id, annotation_round.id)
        if item.dataset_item_id == workflow_data.items["test-1"].id
    )
    feedback_event = service.create_feedback_event(
        db,
        schemas.FeedbackEventCreate(
            project_id=project_id,
            event_type="evaluation_failure",
            annotation_round_id=annotation_round.id,
            round_item_id=test_round_item.id,
            payload={"metric": "f1"},
        ),
        workflow_scope.manager,
    )
    guideline = service.create_guideline(
        db,
        schemas.GuidelineCreate(
            project_id=project_id,
            task_definition_id=workflow_data.task.id,
            name="Protected mining check",
        ),
        workflow_scope.manager,
    )
    with pytest.raises(ValidationError, match="Protected evaluation items"):
        service.create_guideline_proposal(
            db,
            schemas.GuidelineProposalCreate(
                project_id=project_id,
                guideline_id=guideline.id,
                feedback_event_ids=[feedback_event.id],
                proposed_change={"add": "Do not leak holdouts"},
                rationale="Should be rejected",
            ),
            workflow_scope.manager,
        )


def test_feedback_event_lineage_is_normalized_and_protected_groups_are_blocked(
    db, workflow_scope, workflow_data
):
    project_id = workflow_scope.project.id
    dataset_version = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project_id,
            dataset_id=workflow_data.dataset.id,
            source_uri="hf://example/reviews",
            source_revision="feedback-lineage-regression",
            source_format="jsonl",
            data_schema={"text": "string"},
            provenance={"registry": "huggingface"},
            license_info={"name": "apache-2.0"},
            items=[
                schemas.DatasetItemCreate(
                    stable_key="shared-pool",
                    group_key="shared-protected-group",
                    payload={"text": "Candidate"},
                ),
                schemas.DatasetItemCreate(
                    stable_key="shared-test",
                    group_key="shared-protected-group",
                    payload={"text": "Protected sibling"},
                ),
            ],
        ),
        workflow_scope.manager,
    )
    items = {
        item.stable_key: item
        for item in service.list_dataset_items(db, project_id, dataset_version.id)
    }
    service.create_split_map(
        db,
        schemas.SplitMapCreate(
            project_id=project_id,
            dataset_version_id=dataset_version.id,
            name="feedback-lineage-splits",
            strategy="grouped",
            assignments={"shared-pool": "test", "shared-test": "test"},
            protected_splits=["test"],
        ),
        workflow_scope.manager,
    )
    cycle = service.create_learning_cycle(
        db,
        schemas.LearningCycleCreate(
            project_id=project_id,
            name="Feedback lineage",
            task_version_id=workflow_data.task_version.id,
            source_dataset_version_id=dataset_version.id,
            feedback_sources=[
                schemas.LearningFeedbackSource(
                    producer_type="human_disagreement",
                    name="Annotator disagreement",
                )
            ],
        ),
        workflow_scope.manager,
    )
    other_cycle = service.create_learning_cycle(
        db,
        schemas.LearningCycleCreate(
            project_id=project_id,
            name="Different lineage",
            task_version_id=workflow_data.task_version.id,
            source_dataset_version_id=dataset_version.id,
            feedback_sources=[
                schemas.LearningFeedbackSource(
                    producer_type="human_disagreement",
                    name="Annotator disagreement",
                )
            ],
        ),
        workflow_scope.manager,
    )
    feedback_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=project_id,
            dataset_version_id=dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            producer_type="human_disagreement",
            cycle_id=cycle.id,
        ),
        workflow_scope.manager,
    )
    feedback_set = service.create_feedback_set(
        db,
        schemas.FeedbackSetCreate(
            project_id=project_id,
            feedback_run_id=feedback_run.id,
            candidates=[
                schemas.FeedbackCandidateCreate(
                    dataset_item_id=items["shared-pool"].id,
                    output={"label": "positive"},
                ),
                schemas.FeedbackCandidateCreate(
                    dataset_item_id=items["shared-test"].id,
                    output={"label": "negative"},
                ),
            ],
        ),
        workflow_scope.manager,
    )
    candidates = {
        candidate.dataset_item_id: candidate
        for candidate in service.list_feedback_candidates(
            db,
            project_id,
            feedback_set.id,
        )
    }
    other_feedback_set = service.create_feedback_set(
        db,
        schemas.FeedbackSetCreate(
            project_id=project_id,
            feedback_run_id=feedback_run.id,
            candidates=[
                schemas.FeedbackCandidateCreate(
                    dataset_item_id=items["shared-pool"].id,
                    output={"label": "negative"},
                ),
            ],
        ),
        workflow_scope.manager,
    )
    other_pool_candidate = service.list_feedback_candidates(
        db,
        project_id,
        other_feedback_set.id,
    )[0]
    annotation_round = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=project_id,
            name="Feedback lineage round",
            dataset_version_id=dataset_version.id,
            task_version_id=workflow_data.task_version.id,
            cycle_id=cycle.id,
            feedback_set_version_id=feedback_set.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=[workflow_scope.annotator.id],
        ),
        workflow_scope.manager,
    )
    round_items = {
        item.dataset_item_id: item
        for item in service.list_round_items(db, project_id, annotation_round.id)
    }
    pool_candidate = candidates[items["shared-pool"].id]
    test_candidate = candidates[items["shared-test"].id]
    pool_round_item = round_items[items["shared-pool"].id]

    normalized = service.create_feedback_event(
        db,
        schemas.FeedbackEventCreate(
            project_id=project_id,
            event_type="model_disagreement",
            round_item_id=pool_round_item.id,
            feedback_candidate_id=pool_candidate.id,
        ),
        workflow_scope.manager,
    )
    assert normalized.annotation_round_id == annotation_round.id
    assert normalized.cycle_id == cycle.id

    with pytest.raises(ValidationError, match="same dataset item"):
        service.create_feedback_event(
            db,
            schemas.FeedbackEventCreate(
                project_id=project_id,
                event_type="model_disagreement",
                round_item_id=pool_round_item.id,
                feedback_candidate_id=test_candidate.id,
            ),
            workflow_scope.manager,
        )

    with pytest.raises(ValidationError, match="not pinned"):
        service.create_feedback_event(
            db,
            schemas.FeedbackEventCreate(
                project_id=project_id,
                event_type="model_disagreement",
                round_item_id=pool_round_item.id,
                feedback_candidate_id=other_pool_candidate.id,
            ),
            workflow_scope.manager,
        )

    with pytest.raises(ValidationError, match="cycle does not match"):
        service.create_feedback_event(
            db,
            schemas.FeedbackEventCreate(
                project_id=project_id,
                event_type="model_disagreement",
                cycle_id=other_cycle.id,
                round_item_id=pool_round_item.id,
                feedback_candidate_id=pool_candidate.id,
            ),
            workflow_scope.manager,
        )

    group_leak_event = service.create_feedback_event(
        db,
        schemas.FeedbackEventCreate(
            project_id=project_id,
            event_type="evaluation_failure",
            feedback_candidate_id=pool_candidate.id,
        ),
        workflow_scope.manager,
    )
    assert group_leak_event.cycle_id == cycle.id
    assert group_leak_event.annotation_round_id is None

    guideline = service.create_guideline(
        db,
        schemas.GuidelineCreate(
            project_id=project_id,
            task_definition_id=workflow_data.task.id,
            name="Protected group mining check",
        ),
        workflow_scope.manager,
    )
    with pytest.raises(ValidationError, match="Protected evaluation items"):
        service.create_guideline_proposal(
            db,
            schemas.GuidelineProposalCreate(
                project_id=project_id,
                guideline_id=guideline.id,
                feedback_event_ids=[group_leak_event.id],
                proposed_change={"add": "Do not mine protected groups"},
                rationale="Should be rejected",
            ),
            workflow_scope.manager,
        )
