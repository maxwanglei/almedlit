"""Dependency-free local inference used by CI and smoke deployments."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from al_medlit.core.archive import (
    ArchiveExtractionError,
    ArchiveExtractionLimits,
    extract_zip_bounded,
    write_deterministic_zip,
)
from al_medlit.core.config import settings
from al_medlit.core.exceptions import ConflictError, ValidationError
from al_medlit.core.storage import ObjectStorage
from al_medlit.corpus.models import Document, DocumentSentence
from al_medlit.evidence.models import EvidenceTargetVersion
from al_medlit.inference import service
from al_medlit.inference.decoder import (
    DecoderConfig,
    SentenceDecodingInput,
    aggregate_window_logits,
    decode_evidence_blocks,
)
from al_medlit.inference.models import EvidenceCandidatePrediction, InferenceRun, InferenceWindow
from al_medlit.lineage.service import add_lineage_edge, register_stored_artifact
from al_medlit.project.models import Project
from al_medlit.training.compute.base import ComputeBackend, ComputeBackendError, JobBundle
from al_medlit.training.compute.slurm import (
    OutputTransferLimits,
    SSHSlurmComputeBackend,
)
from al_medlit.training.compute_profiles import build_compute_backend
from al_medlit.training.runner import (
    NEURAL_MODEL_TYPES,
    PEFT_MODEL_TYPES,
    RESULT_SCHEMA_VERSION,
    _resolve_torch_device,
    sha256_file,
)


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def _synthetic_logits(ordinal: int) -> tuple[float, float, float]:
    # A deterministic two-sentence evidence span exercises B/I/O decoding and
    # overlapping-window averaging without requiring torch or network access.
    if ordinal == 0:
        return (0.0, 6.0, 0.0)
    if ordinal == 1:
        return (0.0, 0.0, 6.0)
    return (6.0, 0.0, 0.0)


def _extract_checkpoint(
    archive_path: Path,
    destination: Path,
    *,
    limits: ArchiveExtractionLimits | None = None,
) -> Path:
    try:
        extracted = extract_zip_bounded(archive_path, destination, limits=limits)
    except ArchiveExtractionError as exc:
        raise ConflictError(f"Unsafe checkpoint archive: {exc}") from exc
    return extracted / "checkpoint"


def _materialize_artifact_package(storage, package, destination: Path, *, label: str) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()
    for package_file in package.files:
        path = (resolved_root / package_file.relative_path).resolve()
        if not path.is_relative_to(resolved_root):
            raise ConflictError(f"{label} package contains an unsafe path")
        downloaded = storage.download_file(package_file.blob.storage_key, path)
        if (
            downloaded.checksum_sha256 != package_file.checksum_sha256
            or downloaded.size_bytes != package_file.size_bytes
        ):
            raise ConflictError(f"{label} package checksum verification failed")
    return resolved_root


def _materialize_base_model_package(storage, asset, destination: Path) -> Path:
    return _materialize_artifact_package(
        storage,
        asset.package,
        destination,
        label="Base-model",
    )


def _uses_direct_checkpoint_package(checkpoint) -> bool:
    package = checkpoint.package
    if package is None or package.package_format == "legacy_zip":
        return False
    if checkpoint.readiness != "ready" and not checkpoint.manifest.get("synthetic_mode"):
        return False
    retention = package.retention
    if (
        package.readiness != "ready"
        or retention is None
        or retention.archived_at is not None
        or retention.purged_at is not None
    ):
        raise ConflictError("Checkpoint package is archived or unavailable")
    return True


def execute_local_inference(
    db: Session,
    storage: ObjectStorage,
    *,
    run_id: int,
    work_root: str | Path | None = None,
) -> InferenceRun:
    run = service.get_inference_run(db, run_id)
    if run.status == "succeeded":
        return run
    if run.status in {"failed", "cancelled"}:
        raise ConflictError(f"Inference cannot execute from status '{run.status}'")
    if run.compute_profile.backend != "local":
        raise ValidationError("This execution path is only for local compute profiles")
    synthetic_mode = bool(run.checkpoint.manifest.get("synthetic_mode"))
    model_type = str(run.checkpoint.manifest.get("model_type", "evidence_block_sentence_tagger"))

    from al_medlit.training.model_types.evidence_conventional.model import (
        CONVENTIONAL_MODEL_TYPES,
    )

    conventional = model_type in CONVENTIONAL_MODEL_TYPES
    neural = model_type in NEURAL_MODEL_TYPES

    runtime_root = (
        Path(work_root)
        if work_root is not None
        else Path(settings.local_attempt_root)
    )
    run_root = runtime_root / f"inference-run-{run.id}"
    checkpoint_path = run_root / "checkpoint.zip"
    checkpoint_root = None
    if _uses_direct_checkpoint_package(run.checkpoint):
        package = run.checkpoint.package
        checkpoint_root = _materialize_artifact_package(
            storage,
            package,
            run_root / f"checkpoint-package-{package.manifest_digest}",
            label="Checkpoint",
        )
    else:
        downloaded = storage.download_file(
            run.checkpoint.artifact.storage_key,
            checkpoint_path,
        )
        if (
            downloaded.checksum_sha256 != run.checkpoint.artifact.content_hash
            or downloaded.size_bytes != run.checkpoint.artifact.size_bytes
        ):
            raise ConflictError("Checkpoint object does not match its immutable checksum")

    predictor = None
    peft_tokenizer = None
    torch = None
    if synthetic_mode:
        token_counter = _token_count
    elif conventional:
        from al_medlit.training.model_types.evidence_conventional.model import (
            MissingConventionalDependencyError,
            actual_token_count,
            load_conventional_model,
        )

        checkpoint_root = checkpoint_root or _extract_checkpoint(
            checkpoint_path,
            run_root / "extracted",
        )
        try:
            predictor = load_conventional_model(checkpoint_root)
        except (MissingConventionalDependencyError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc

        def token_counter(text: str) -> int:
            return actual_token_count(predictor, text)

    elif neural:
        from al_medlit.training.model_types.evidence_neural import load_neural_bundle
        from al_medlit.training.model_types.evidence_neural.model import (
            MissingNeuralDependencyError,
        )

        checkpoint_root = checkpoint_root or _extract_checkpoint(
            checkpoint_path,
            run_root / "extracted",
        )
        try:
            predictor = load_neural_bundle(checkpoint_root)
        except (MissingNeuralDependencyError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc

        def token_counter(text: str) -> int:
            return max(1, len(predictor.vocabulary.tokenize(text)))

    elif model_type in PEFT_MODEL_TYPES:
        from al_medlit.training.model_types.evidence_peft import (
            ImmutableBaseModelReference,
        )
        from al_medlit.training.model_types.evidence_peft.training import (
            load_peft_runtime,
        )

        checkpoint_root = checkpoint_root or _extract_checkpoint(
            checkpoint_path,
            run_root / "extracted",
        )
        base_asset = service.resolve_peft_base_model_asset(db, run.checkpoint)
        base_root = _materialize_base_model_package(
            storage,
            base_asset,
            run_root / "base-model",
        )
        reference = ImmutableBaseModelReference(
            asset_id=base_asset.id,
            package_id=base_asset.package_id,
            manifest_digest=base_asset.package.manifest_digest,
            exact_revision=base_asset.exact_revision,
        )
        try:
            predictor, peft_tokenizer = load_peft_runtime(
                checkpoint_root,
                base_model_root=base_root,
                base_reference=reference,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc

        def token_counter(text: str) -> int:
            return max(
                1,
                len(peft_tokenizer.encode(text, add_special_tokens=False)),
            )
    else:
        try:
            import torch as torch_module

            from al_medlit.training.model_types.evidence_block_sentence_tagger.model import (
                MissingMLDependencyError,
                load_sentence_tagger,
            )
        except ImportError as exc:  # pragma: no cover - optional ML environment
            raise ValidationError(
                "Install the optional 'ml' dependencies for real inference"
            ) from exc
        checkpoint_root = checkpoint_root or _extract_checkpoint(
            checkpoint_path,
            run_root / "extracted",
        )
        try:
            predictor = load_sentence_tagger(checkpoint_root)
        except MissingMLDependencyError as exc:  # pragma: no cover
            raise ValidationError(str(exc)) from exc
        torch = torch_module
        _logical_device, runtime_device = _resolve_torch_device(torch, predictor.config.device)
        predictor.model.to(torch.device(runtime_device))
        predictor.model.eval()

        def token_counter(text: str) -> int:
            return max(
                1,
                len(predictor.tokenizer.encode(text, add_special_tokens=False)),
            )

    if run.started_at is None:
        run.started_at = datetime.now(UTC)
    run.status = "running"
    run.external_job_id = run.external_job_id or f"local:inference-{run.id}"
    db.commit()

    windows = service.materialize_inference_windows(
        db,
        run_id=run.id,
        token_counter=token_counter,
    )
    windows_by_scope: dict[tuple[int, int, int], list[InferenceWindow]] = defaultdict(list)
    for window in windows:
        windows_by_scope[
            (window.document_id, window.structure_version_id, window.target_version_id)
        ].append(window)

    decoded_by_scope = {}
    diagnostics_scopes = []
    for (document_id, structure_version_id, target_version_id), scope_windows in sorted(
        windows_by_scope.items()
    ):
        sentences = (
            db.query(DocumentSentence)
            .filter(DocumentSentence.structure_version_id == structure_version_id)
            .order_by(DocumentSentence.ordinal)
            .all()
        )
        window_logits = []
        target_version = db.get(EvidenceTargetVersion, target_version_id)
        for window in scope_windows:
            selected_sentences = [
                sentence
                for sentence in sentences
                if window.start_sentence_ordinal <= sentence.ordinal <= window.end_sentence_ordinal
            ]
            if synthetic_mode:
                window_logits.append(
                    {
                        sentence.ordinal: _synthetic_logits(sentence.ordinal)
                        for sentence in selected_sentences
                    }
                )
            elif conventional:
                from al_medlit.training.model_types.evidence_conventional.model import (
                    predict_window,
                )

                values = predict_window(
                    predictor,
                    target_text=target_version.text,
                    sentences=[
                        sentence.structure_version.document.text[
                            sentence.start_offset : sentence.end_offset
                        ]
                        for sentence in selected_sentences
                    ],
                )
                if predictor.produces_sentence_scores:
                    inside = False
                    labels = []
                    for score in values:
                        positive = float(score) >= predictor.config.sentence_score_threshold
                        labels.append(
                            "B" if positive and not inside else ("I" if positive else "O")
                        )
                        inside = positive
                else:
                    labels = [str(value) for value in values]
                one_hot = {
                    "O": (1.0, 0.0, 0.0),
                    "B": (0.0, 1.0, 0.0),
                    "I": (0.0, 0.0, 1.0),
                }
                window_logits.append(
                    {
                        sentence.ordinal: one_hot[labels[index]]
                        for index, sentence in enumerate(selected_sentences)
                    }
                )
            elif neural:
                from al_medlit.training.model_types.evidence_neural import (
                    EvidenceTextDocument,
                    predict_neural_model,
                )

                prediction = predict_neural_model(
                    predictor,
                    (
                        EvidenceTextDocument(
                            document_id=str(document_id),
                            target_id=target_version_id,
                            target_text=target_version.text,
                            sentences=tuple(
                                sentence.structure_version.document.text[
                                    sentence.start_offset : sentence.end_offset
                                ]
                                for sentence in selected_sentences
                            ),
                            sentence_ordinals=tuple(
                                sentence.ordinal for sentence in selected_sentences
                            ),
                        ),
                    ),
                    device=predictor.config.device,
                )[0]
                window_logits.append(
                    {
                        ordinal: prediction.probabilities[index]
                        for index, ordinal in enumerate(prediction.sentence_ordinals)
                    }
                )
            elif model_type in PEFT_MODEL_TYPES:
                from al_medlit.training.model_types.evidence_peft.training import (
                    predict_peft_examples,
                    prepare_peft_examples,
                )

                row = {
                    "document_id": str(document_id),
                    "target": {"id": target_version_id, "text": target_version.text},
                    "sentences": [
                        {
                            "ordinal": sentence.ordinal,
                            "text": sentence.structure_version.document.text[
                                sentence.start_offset : sentence.end_offset
                            ],
                        }
                        for sentence in selected_sentences
                    ],
                }
                examples = prepare_peft_examples(
                    (row,),
                    peft_tokenizer,
                    predictor.config,
                    include_unreviewed=True,
                    supervised=False,
                )
                predicted, _valid_count = predict_peft_examples(
                    predictor,
                    peft_tokenizer,
                    examples,
                    config=predictor.config,
                )
                one_hot = {
                    "O": (1.0, 0.0, 0.0),
                    "B": (0.0, 1.0, 0.0),
                    "I": (0.0, 0.0, 1.0),
                }
                labels_by_ordinal = {
                    ordinal: label
                    for example, labels in zip(examples, predicted, strict=True)
                    for ordinal, label in zip(
                        example.sentence_ordinals,
                        labels,
                        strict=True,
                    )
                }
                window_logits.append(
                    {
                        sentence.ordinal: one_hot[labels_by_ordinal[sentence.ordinal]]
                        for sentence in selected_sentences
                    }
                )
            else:
                from al_medlit.training.model_types.evidence_block_sentence_tagger.dataset import (
                    encode_sentence_markers,
                )

                encoded = encode_sentence_markers(
                    predictor.tokenizer,
                    target_text=target_version.text,
                    sentences=[
                        sentence.structure_version.document.text[
                            sentence.start_offset : sentence.end_offset
                        ]
                        for sentence in selected_sentences
                    ],
                    target_marker_token=predictor.config.target_marker_token,
                    sentence_marker_token=predictor.config.sentence_marker_token,
                    target_conditioning=predictor.config.target_conditioning,
                    max_length=predictor.config.max_length,
                )
                device = next(predictor.model.parameters()).device
                with torch.no_grad():
                    output = predictor.model(
                        input_ids=torch.tensor(
                            [encoded.input_ids], dtype=torch.long, device=device
                        ),
                        attention_mask=torch.tensor(
                            [encoded.attention_mask], dtype=torch.long, device=device
                        ),
                        sentence_marker_positions=torch.tensor(
                            [encoded.sentence_marker_positions],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                values = output["logits"][0].detach().cpu().tolist()
                window_logits.append(
                    {
                        sentence.ordinal: tuple(values[index])
                        for index, sentence in enumerate(selected_sentences)
                    }
                )
        aggregated = aggregate_window_logits(window_logits, method="mean")
        result = decode_evidence_blocks(
            [
                SentenceDecodingInput(
                    id=sentence.id,
                    ordinal=sentence.ordinal,
                    start_char=sentence.start_offset,
                    end_char=sentence.end_offset,
                    section_path=tuple(sentence.section.path or []),
                )
                for sentence in sentences
            ],
            aggregated,
            DecoderConfig(
                block_threshold=float(run.decoder_config["block_threshold"]),
                allow_cross_section=bool(run.decoder_config["allow_cross_section"]),
                merge_adjacent=bool(run.decoder_config["merge_adjacent"]),
            ),
        )
        decoded_by_scope[(document_id, structure_version_id, target_version_id)] = (
            result,
            [window.id for window in scope_windows],
        )
        diagnostics_scopes.append(
            {
                "document_id": document_id,
                "structure_version_id": structure_version_id,
                "target_version_id": target_version_id,
                "window_ids": [window.id for window in scope_windows],
                "sentences": [
                    {
                        "ordinal": sentence.ordinal,
                        "probabilities": list(sentence.probabilities),
                        "contribution_count": sentence.contribution_count,
                        "raw_label": sentence.raw_label,
                        "decoded_label": sentence.decoded_label,
                    }
                    for sentence in result.sentences
                ],
            }
        )

    diagnostics_artifact_id = run.diagnostics_artifact_id
    if diagnostics_artifact_id is None:
        diagnostics_artifact_id = service.store_inference_diagnostics(
            db,
            storage,
            run_id=run.id,
            diagnostics={
                "schema_version": "inference-diagnostics-v1",
                "synthetic_mode": synthetic_mode,
                "checkpoint_id": run.checkpoint_id,
                "checkpoint_checksum_sha256": run.checkpoint.artifact.content_hash,
                "decoder_config": run.decoder_config,
                "scopes": diagnostics_scopes,
            },
            actor_user_id=run.created_by_user_id,
        )

    for (document_id, structure_version_id, target_version_id), (
        result,
        window_ids,
    ) in decoded_by_scope.items():
        service.persist_decoder_result(
            db,
            run_id=run.id,
            document_id=document_id,
            structure_version_id=structure_version_id,
            target_version_id=target_version_id,
            result=result,
            source_window_ids=window_ids,
            diagnostics_artifact_id=diagnostics_artifact_id,
        )

    candidate_count = (
        db.query(EvidenceCandidatePrediction)
        .filter(EvidenceCandidatePrediction.run_id == run.id)
        .count()
    )
    run = service.get_inference_run(db, run.id)
    run.status = "succeeded"
    run.completed_at = datetime.now(UTC)
    run.metrics = {
        **run.metrics,
        "synthetic_mode": synthetic_mode,
        "window_count": len(windows),
        "candidate_count": candidate_count,
        "checkpoint_checksum_sha256": run.checkpoint.artifact.content_hash,
        "decoder_version": "evidence-block-decoder-v1",
    }
    db.commit()
    db.refresh(run)
    return run


def execute_local_synthetic_inference(
    db: Session,
    storage: ObjectStorage,
    *,
    run_id: int,
    work_root: str | Path | None = None,
) -> InferenceRun:
    """Backward-compatible alias retained for existing integrations."""

    return execute_local_inference(
        db,
        storage,
        run_id=run_id,
        work_root=work_root,
    )


def _canonical_json(value: object) -> bytes:
    import json

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _deterministic_zip(destination: Path, root: Path, relative_paths: list[str]) -> None:
    write_deterministic_zip(destination, root, relative_paths)


def build_inference_bundle(
    db: Session,
    storage: ObjectStorage,
    *,
    run_id: int,
    backend: SSHSlurmComputeBackend,
    work_root: str | Path | None = None,
) -> JobBundle:
    run = service.get_inference_run(db, run_id)
    project = db.get(Project, run.project_id)
    if project is None:
        raise ValidationError("Inference project not found")
    root = (
        Path(work_root)
        if work_root is not None
        else Path(settings.local_attempt_root)
    )
    directory = root / f"inference-run-{run.id}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_identity = run.checkpoint.artifact.content_hash
    checkpoint_input_files: list[str] = []
    checkpoint_metadata = {
        "checksum_sha256": checkpoint_identity,
        "checkpoint_id": run.checkpoint_id,
        "training_mode": run.checkpoint.training_mode,
        "manifest": run.checkpoint.manifest,
    }
    if _uses_direct_checkpoint_package(run.checkpoint):
        checkpoint_package = run.checkpoint.package
        checkpoint_prefix = "checkpoint-package"
        _materialize_artifact_package(
            storage,
            checkpoint_package,
            directory / checkpoint_prefix,
            label="Checkpoint",
        )
        package_files = [
            {
                "path": f"{checkpoint_prefix}/{item.relative_path}",
                "relative_path": item.relative_path,
                "checksum_sha256": item.checksum_sha256,
                "size_bytes": item.size_bytes,
            }
            for item in checkpoint_package.files
        ]
        checkpoint_input_files.extend(item["path"] for item in package_files)
        checkpoint_metadata["package"] = {
            "path": checkpoint_prefix,
            "manifest_sha256": checkpoint_package.manifest_digest,
            "files": package_files,
        }
    else:
        checkpoint_path = directory / "checkpoint.zip"
        downloaded = storage.download_file(
            run.checkpoint.artifact.storage_key,
            checkpoint_path,
        )
        if (
            downloaded.checksum_sha256 != checkpoint_identity
            or downloaded.size_bytes != run.checkpoint.artifact.size_bytes
        ):
            raise ConflictError("Checkpoint object does not match its immutable checksum")
        checkpoint_metadata.update(
            {
                "path": checkpoint_path.name,
                "size_bytes": downloaded.size_bytes,
            }
        )
        checkpoint_input_files.append(checkpoint_path.name)

    base_model = None
    base_asset = None
    if run.checkpoint.model_type in PEFT_MODEL_TYPES:
        base_asset = service.resolve_peft_base_model_asset(db, run.checkpoint)
        materialized_root = _materialize_base_model_package(
            storage,
            base_asset,
            directory / "base-model-package",
        )
        base_package_files = [
            {
                "path": f"base-model-package/{item.relative_path}",
                "relative_path": item.relative_path,
                "checksum_sha256": item.checksum_sha256,
                "size_bytes": item.size_bytes,
            }
            for item in base_asset.package.files
        ]
        base_model = {
            "base_model_asset_id": base_asset.id,
            "base_model_package_id": base_asset.package_id,
            "base_model_manifest_sha256": base_asset.package.manifest_digest,
            "base_model_exact_revision": base_asset.exact_revision,
            "package": {
                "path": materialized_root.relative_to(directory).as_posix(),
                "manifest_sha256": base_asset.package.manifest_digest,
                "files": base_package_files,
            },
        }

    documents = []
    for snapshot_document in run.corpus_snapshot.documents:
        document = db.get(Document, snapshot_document.document_id)
        sentences = (
            db.query(DocumentSentence)
            .filter(DocumentSentence.structure_version_id == snapshot_document.structure_version_id)
            .order_by(DocumentSentence.ordinal)
            .all()
        )
        documents.append(
            {
                "document_id": document.id,
                "structure_version_id": snapshot_document.structure_version_id,
                "source_hash": snapshot_document.source_hash,
                "sentences": [
                    {
                        "id": sentence.id,
                        "ordinal": sentence.ordinal,
                        "paragraph_ordinal": sentence.paragraph.ordinal,
                        "section_path": list(sentence.section.path or []),
                        "text": document.text[sentence.start_offset : sentence.end_offset],
                        "start_char": sentence.start_offset,
                        "end_char": sentence.end_offset,
                    }
                    for sentence in sentences
                ],
            }
        )
    targets = []
    for target_id in run.target_version_ids:
        version = db.get(EvidenceTargetVersion, target_id)
        targets.append(
            {
                "id": version.id,
                "key": version.target.key,
                "name": version.target.name,
                "text": version.text,
            }
        )
    corpus_path = directory / "inference-input.json"
    corpus_path.write_bytes(
        _canonical_json(
            {
                "schema_version": "inference-input-v1",
                "corpus_snapshot_id": run.corpus_snapshot_id,
                "documents": documents,
                "targets": targets,
            }
        )
    )
    job_manifest = {
        "schema_version": "al-medlit-job-v1",
        "kind": "inference",
        "job_key": f"inference:{run.id}",
        "run_id": run.id,
        "workspace_id": project.workspace_id,
        "window_config": run.window_config,
        "decoder_config": run.decoder_config,
        "checkpoint": checkpoint_metadata,
        "base_model": base_model,
        "corpus": {
            "path": corpus_path.name,
            "checksum_sha256": sha256_file(corpus_path),
            "size_bytes": corpus_path.stat().st_size,
        },
        "output_directory": "outputs",
    }
    (directory / "job.json").write_bytes(_canonical_json(job_manifest))
    bundle = JobBundle(
        job_key=f"inference-{run.id}",
        command=(
            "python",
            "-m",
            "al_medlit.training.runner",
            "infer",
            "job.json",
        ),
        local_bundle_path=directory,
    )
    (directory / "job.sbatch").write_text(
        backend.render_sbatch_script(bundle),
        encoding="utf-8",
    )
    bundle_files = ["job.json", "inference-input.json", "job.sbatch", *checkpoint_input_files]
    if base_model is not None:
        bundle_files.extend(item["path"] for item in base_model["package"]["files"])
    archive_path = directory / "job-bundle.zip"
    _deterministic_zip(archive_path, directory, bundle_files)
    key = f"projects/{run.project_id}/inference/runs/{run.id}/job-bundle.zip"
    stored = storage.put_file(key, archive_path, content_type="application/zip")
    artifact = register_stored_artifact(
        db,
        project_id=run.project_id,
        artifact_type="inference_job_bundle",
        stored=stored,
        manifest={
            "schema_version": "inference-job-bundle-v1",
            "run_id": run.id,
            "checkpoint_checksum_sha256": checkpoint_identity,
            "corpus_input_checksum_sha256": sha256_file(corpus_path),
            "files": bundle_files,
        },
        created_by_user_id=run.created_by_user_id,
        schema_version="inference-job-bundle-v1",
    )
    add_lineage_edge(
        db,
        upstream_artifact_id=run.checkpoint.artifact_id,
        downstream_artifact_id=artifact.id,
        relationship_type="bundled_for_inference",
    )
    if base_asset is not None:
        add_lineage_edge(
            db,
            upstream_artifact_id=base_asset.package.lineage_artifact_id,
            downstream_artifact_id=artifact.id,
            relationship_type="used_base_model",
        )
    add_lineage_edge(
        db,
        upstream_artifact_id=run.corpus_snapshot.artifact_id,
        downstream_artifact_id=artifact.id,
        relationship_type="bundled_for_inference",
    )
    run.metrics = {**run.metrics, "bundle_artifact_id": artifact.id}
    db.commit()
    return bundle


def _retrieve_remote_result(
    backend: SSHSlurmComputeBackend,
    *,
    run_id: int,
    output_root: Path,
) -> dict:
    import json

    output_root.mkdir(parents=True, exist_ok=True)
    job_key = f"inference-{run_id}"
    try:
        backend.collect_outputs(
            job_key=job_key,
            output_root=output_root,
            limits=OutputTransferLimits(
                max_files=64,
                max_file_bytes=2 * 1024 * 1024 * 1024,
                max_total_bytes=4 * 1024 * 1024 * 1024,
            ),
        )
    except ComputeBackendError as exc:
        raise ConflictError(str(exc)) from exc
    manifest_path = output_root / "artifact-manifest.json"
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        artifact_manifest.get("schema_version") != RESULT_SCHEMA_VERSION
        or artifact_manifest.get("kind") != "inference"
        or artifact_manifest.get("job_key") != f"inference:{run_id}"
        or artifact_manifest.get("status") != "succeeded"
    ):
        raise ConflictError("Remote inference result does not match this run")
    return artifact_manifest


def finalize_remote_inference(
    db: Session,
    storage: ObjectStorage,
    *,
    run_id: int,
    backend: SSHSlurmComputeBackend,
    work_root: str | Path | None = None,
) -> InferenceRun:
    import json

    run = service.get_inference_run(db, run_id)
    if run.status != "succeeded":
        raise ConflictError("Only a succeeded inference run can be finalized")
    if run.diagnostics_artifact_id is not None:
        return run
    root = (
        Path(work_root)
        if work_root is not None
        else Path(settings.local_attempt_root)
    )
    output_root = root / f"inference-run-{run.id}" / "outputs"
    _retrieve_remote_result(backend, run_id=run.id, output_root=output_root)
    inference_result = json.loads(
        (output_root / "inference-result.json").read_text(encoding="utf-8")
    )
    if (
        inference_result.get("schema_version") != "inference-window-logits-v1"
        or inference_result.get("checkpoint_checksum_sha256")
        != run.checkpoint.artifact.content_hash
    ):
        raise ConflictError("Remote logits do not match the selected checkpoint")

    rows_by_scope = defaultdict(list)
    for row in inference_result["windows"]:
        scope = (
            int(row["document_id"]),
            int(row["structure_version_id"]),
            int(row["target_version_id"]),
        )
        rows_by_scope[scope].append(row)
    decoded_by_scope = {}
    for (document_id, structure_version_id, target_version_id), rows in rows_by_scope.items():
        window_models = []
        window_logits = []
        for row in rows:
            model = (
                db.query(InferenceWindow)
                .filter(
                    InferenceWindow.run_id == run.id,
                    InferenceWindow.stable_key == row["stable_key"],
                )
                .first()
            )
            if model is None:
                model = InferenceWindow(
                    run_id=run.id,
                    document_id=document_id,
                    structure_version_id=structure_version_id,
                    target_version_id=target_version_id,
                    stable_key=row["stable_key"],
                    start_sentence_ordinal=row["start_sentence_ordinal"],
                    end_sentence_ordinal=row["end_sentence_ordinal"],
                    token_count=row["token_count"],
                    status="pending",
                )
                db.add(model)
                db.flush()
            window_models.append(model)
            window_logits.append(
                {int(ordinal): values for ordinal, values in row["logits"].items()}
            )
        aggregated = aggregate_window_logits(window_logits, method="mean")
        sentences = (
            db.query(DocumentSentence)
            .filter(DocumentSentence.structure_version_id == structure_version_id)
            .order_by(DocumentSentence.ordinal)
            .all()
        )
        result = decode_evidence_blocks(
            [
                SentenceDecodingInput(
                    id=sentence.id,
                    ordinal=sentence.ordinal,
                    start_char=sentence.start_offset,
                    end_char=sentence.end_offset,
                    section_path=tuple(sentence.section.path or []),
                )
                for sentence in sentences
            ],
            aggregated,
            DecoderConfig(
                block_threshold=float(run.decoder_config["block_threshold"]),
                allow_cross_section=bool(run.decoder_config["allow_cross_section"]),
                merge_adjacent=bool(run.decoder_config["merge_adjacent"]),
            ),
        )
        decoded_by_scope[(document_id, structure_version_id, target_version_id)] = (
            result,
            [model.id for model in window_models],
        )
    db.commit()
    diagnostics_artifact_id = service.store_inference_diagnostics(
        db,
        storage,
        run_id=run.id,
        diagnostics=inference_result,
        actor_user_id=run.created_by_user_id,
    )
    for scope, (result, window_ids) in decoded_by_scope.items():
        service.persist_decoder_result(
            db,
            run_id=run.id,
            document_id=scope[0],
            structure_version_id=scope[1],
            target_version_id=scope[2],
            result=result,
            source_window_ids=window_ids,
            diagnostics_artifact_id=diagnostics_artifact_id,
        )
    stored_log = storage.put_file(
        f"projects/{run.project_id}/inference/runs/{run.id}/infer.log",
        output_root / "infer.log",
        content_type="text/plain",
    )
    log_artifact = register_stored_artifact(
        db,
        project_id=run.project_id,
        artifact_type="inference_log",
        stored=stored_log,
        manifest={"schema_version": "inference-log-v1", "run_id": run.id},
        created_by_user_id=run.created_by_user_id,
        schema_version="inference-log-v1",
    )
    bundle_artifact_id = run.metrics.get("bundle_artifact_id")
    if bundle_artifact_id:
        add_lineage_edge(
            db,
            upstream_artifact_id=bundle_artifact_id,
            downstream_artifact_id=log_artifact.id,
            relationship_type="produced",
        )
    run.metrics = {
        **run.metrics,
        "window_count": sum(len(rows) for rows in rows_by_scope.values()),
        "candidate_count": (
            db.query(EvidenceCandidatePrediction)
            .filter(EvidenceCandidatePrediction.run_id == run.id)
            .count()
        ),
        "inference_log_artifact_id": log_artifact.id,
        "checkpoint_checksum_sha256": run.checkpoint.artifact.content_hash,
    }
    db.commit()
    db.refresh(run)
    return run


def execute_inference_run(
    db: Session,
    storage: ObjectStorage,
    *,
    run_id: int,
    backend: ComputeBackend | None = None,
    work_root: str | Path | None = None,
) -> InferenceRun:
    run = service.get_inference_run(db, run_id)
    if run.compute_profile.backend == "local":
        return execute_local_inference(
            db,
            storage,
            run_id=run.id,
            work_root=work_root,
        )
    selected_backend = backend or build_compute_backend(run.compute_profile)
    if not isinstance(selected_backend, SSHSlurmComputeBackend):
        raise ValidationError("SSH/Slurm inference requires the SSHSlurm compute backend")
    if run.status == "queued":
        bundle = build_inference_bundle(
            db,
            storage,
            run_id=run.id,
            backend=selected_backend,
            work_root=work_root,
        )
        run = service.submit_inference_run(
            db,
            run_id=run.id,
            bundle=bundle,
            backend=selected_backend,
        )
    elif run.status in service.OPEN_RUN_STATUSES:
        run = service.reconcile_inference_run(
            db,
            run_id=run.id,
            backend=selected_backend,
        )
    if run.status == "succeeded":
        return finalize_remote_inference(
            db,
            storage,
            run_id=run.id,
            backend=selected_backend,
            work_root=work_root,
        )
    return run
