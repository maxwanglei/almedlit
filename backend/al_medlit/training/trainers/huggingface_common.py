"""Shared fail-closed helpers for optional Hugging Face worker plugins."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.recipe_registry import training_recipes
from al_medlit.training.trainers.contracts import (
    TrainerPreflight,
    TrainingInput,
    TrainingPlan,
)

FORBIDDEN_ARTIFACT_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".joblib", ".pickle", ".pkl", ".pt", ".pth"}
)
PACKAGE_IMPORTS = {
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "peft": "peft",
    "safetensors": "safetensors",
    "torch": "torch",
    "transformers": "transformers",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value), encoding="ascii")


def validate_recipe(
    recipe_keys: Sequence[str],
    recipe_key: str,
    config: Mapping,
) -> tuple[dict, str]:
    if recipe_key not in recipe_keys:
        raise ValidationError(f"Unsupported trainer recipe '{recipe_key}'")
    validated = training_recipes.validate(recipe_key, dict(config))
    if not validated.valid or validated.normalized_config is None:
        raise ValidationError("; ".join(validated.errors))
    return validated.normalized_config, validated.recipe_version


def validate_task_kind(recipe_key: str, training_input: TrainingInput) -> str:
    descriptor = training_recipes.get(recipe_key)
    supported = {kind.value for kind in descriptor.supported_task_kinds}
    task_kind = training_input.task_kind
    if task_kind is None:
        if len(supported) != 1:
            choices = ", ".join(sorted(supported))
            raise ValidationError(
                f"Training input task_kind is required; recipe supports: {choices}"
            )
        task_kind = next(iter(supported))
    if task_kind not in supported:
        raise ValidationError(
            f"Recipe '{recipe_key}' does not support task kind '{task_kind}'"
        )
    return task_kind


def validate_base_model_provenance(
    normalized_config: Mapping,
    training_input: TrainingInput,
) -> dict:
    if training_input.base_model_fingerprint is None:
        raise ValidationError("Transformer training requires base_model_fingerprint")
    return {
        "asset_id": normalized_config["base_model_asset_id"],
        "fingerprint": training_input.base_model_fingerprint,
        "local_only": True,
    }


def require_local_base_model(training_input: TrainingInput) -> Path:
    if training_input.base_model_path is None:
        raise ValidationError("Worker execution requires a local base_model_path")
    path = Path(training_input.base_model_path).expanduser().resolve()
    if not path.is_dir():
        raise ValidationError("base_model_path must be an existing local directory")
    return path


def preflight_for_recipe(
    recipe_keys: Sequence[str],
    recipe_key: str | None,
) -> TrainerPreflight:
    selected_key = recipe_key or recipe_keys[0]
    if selected_key not in recipe_keys:
        raise ValidationError(f"Unsupported trainer recipe '{selected_key}'")
    descriptor = training_recipes.get(selected_key)
    checks: list[dict] = []
    for package in descriptor.environment.packages:
        module = PACKAGE_IMPORTS.get(package, package.replace("-", "_"))
        present = importlib.util.find_spec(module) is not None
        checks.append(
            {
                "key": f"package:{package}",
                "status": "pass" if present else "fail",
                "message": (
                    f"{package} is installed"
                    if present
                    else (
                        f"{package} is required in the "
                        f"{descriptor.environment.runtime_class.value} worker"
                    )
                ),
            }
        )

    runtime_class = descriptor.environment.runtime_class.value
    if runtime_class in {"peft-accelerator", "qlora-cuda"}:
        accelerator_ready = False
        device_label = "CUDA" if runtime_class == "qlora-cuda" else "accelerator"
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                cuda_ready = bool(torch.cuda.is_available())
                mps_backend = getattr(torch.backends, "mps", None)
                mps_ready = bool(mps_backend and mps_backend.is_available())
                npu_backend = getattr(torch, "npu", None)
                npu_ready = bool(npu_backend and npu_backend.is_available())
                accelerator_ready = (
                    cuda_ready
                    if runtime_class == "qlora-cuda"
                    else cuda_ready or mps_ready or npu_ready
                )
            except Exception:
                accelerator_ready = False
        checks.append(
            {
                "key": (
                    "device:cuda"
                    if runtime_class == "qlora-cuda"
                    else "device:accelerator"
                ),
                "status": "pass" if accelerator_ready else "fail",
                "message": (
                    f"{device_label} is available"
                    if accelerator_ready
                    else (
                        "QLoRA requires a verified CUDA worker"
                        if runtime_class == "qlora-cuda"
                        else "Full and PEFT SFT require a verified accelerator worker"
                    )
                ),
            }
        )
    return TrainerPreflight(
        ready=all(check["status"] == "pass" for check in checks),
        runtime_class=descriptor.environment.runtime_class.value,
        checks=tuple(checks),
    )


def require_ready(report: TrainerPreflight) -> None:
    if report.ready:
        return
    messages = [check["message"] for check in report.checks if check["status"] == "fail"]
    raise ValidationError("Worker preflight failed: " + "; ".join(messages))


def base_plan(
    *,
    trainer_key: str,
    trainer_version: str,
    recipe_key: str,
    recipe_version: str,
    normalized_config: dict,
    training_input: TrainingInput,
    task_kind: str,
    seed: int,
    row_mapping: dict,
    output_contract: dict,
) -> TrainingPlan:
    descriptor = training_recipes.get(recipe_key)
    manifest = {
        "schema_version": "al-medlit-training-plan-v1",
        "recipe": {"key": recipe_key, "version": recipe_version},
        "trainer": {"key": trainer_key, "version": trainer_version},
        "runtime_class": descriptor.environment.runtime_class.value,
        "task": {
            "kind": task_kind,
            "schema": training_input.task_schema,
            "row_mapping": row_mapping,
        },
        "base_model": validate_base_model_provenance(
            normalized_config,
            training_input,
        ),
        "dataset_fingerprint": training_input.dataset_fingerprint,
        "split_fingerprint": training_input.split_fingerprint,
        "seed": seed,
        "row_counts": {
            "train": len(training_input.rows),
            "validation": len(training_input.validation_rows),
        },
        "output_contract": output_contract,
        "artifact_policy": {
            "weights": "safetensors_only",
            "configuration": "json",
            "forbidden_extensions": sorted(FORBIDDEN_ARTIFACT_SUFFIXES),
            "remote_code": False,
        },
    }
    manifest["plan_sha256"] = canonical_sha256(manifest)
    return TrainingPlan(
        manifest=manifest,
        normalized_config=normalized_config,
    )


def prepare_destination(destination: Path) -> Path:
    if destination.exists():
        raise ValidationError("Training destination must not already exist")
    destination.mkdir(parents=True)
    return destination


def artifact_inventory(destination: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(destination).as_posix()
        if path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            raise ValidationError(f"Unsafe training artifact was produced: {relative}")
        if relative != "manifest.json":
            inventory[relative] = file_sha256(path)
    return inventory


def finalize_output_manifest(
    destination: Path,
    *,
    plan: TrainingPlan,
    validation_metrics: Mapping[str, float | None],
) -> dict:
    manifest = {
        "schema_version": "al-medlit-training-package-v1",
        "plan": plan.manifest,
        "normalized_config_sha256": canonical_sha256(plan.normalized_config),
        "validation_metrics": dict(validation_metrics),
        "files": artifact_inventory(destination),
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def relative_artifact_paths(destination: Path) -> tuple[str, ...]:
    return tuple(
        path.relative_to(destination).as_posix()
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    )


def temporary_trainer_directory() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="al-medlit-trainer-")


def _move_to_device(value, device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def execute_torch_training(
    *,
    torch,
    model,
    train_rows: Sequence[dict],
    validation_rows: Sequence[dict],
    collator,
    config: Mapping,
    seed: int,
    dispatched: bool = False,
) -> dict[str, float | None]:
    if not train_rows:
        raise ValidationError("Training input must contain at least one row")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif getattr(torch, "npu", None) and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
        device = torch.device("npu")
    else:
        device = torch.device("cpu")
    if not dispatched:
        model.to(device)
    else:
        try:
            device = next(model.parameters()).device
        except StopIteration as exc:
            raise ValidationError("Loaded model has no trainable parameters") from exc

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        ListDataset(train_rows),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    validation_loader = (
        torch.utils.data.DataLoader(
            ListDataset(validation_rows),
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collator,
        )
        if validation_rows
        else None
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValidationError("Loaded model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"])
    accumulation = config["gradient_accumulation_steps"]
    precision = config.get("mixed_precision", "none")
    autocast_enabled = precision != "none" and device.type in {"cpu", "cuda", "mps", "npu"}
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = (
        torch.amp.GradScaler(device.type)
        if autocast_enabled and precision == "fp16"
        else None
    )
    remaining_batches = math.ceil(float(config["epochs"]) * len(loader))
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while remaining_batches > 0:
        epoch_batches = min(len(loader), remaining_batches)
        for batch_index, batch in enumerate(islice(loader, epoch_batches)):
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(accumulation, epoch_batches - group_start)
            batch = _move_to_device(batch, device)
            context = (
                torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=True,
                )
                if autocast_enabled
                else nullcontext()
            )
            with context:
                loss = model(**batch).loss / group_size
            if not math.isfinite(float(loss.detach().cpu())):
                raise ValidationError("Training produced a non-finite loss")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * group_size)
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == epoch_batches:
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
        remaining_batches -= epoch_batches

    metrics: dict[str, float | None] = {
        "train_loss": sum(losses) / len(losses) if losses else None,
    }
    if validation_loader is not None:
        evaluation_losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for batch in validation_loader:
                batch = _move_to_device(batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    evaluation_loss = float(model(**batch).loss.detach().cpu())
                if not math.isfinite(evaluation_loss):
                    raise ValidationError("Validation produced a non-finite loss")
                evaluation_losses.append(evaluation_loss)
        metrics["validation_loss"] = (
            sum(evaluation_losses) / len(evaluation_losses)
            if evaluation_losses
            else None
        )
    return metrics


class ListDataset:
    """Minimal torch Dataset-compatible wrapper without importing torch."""

    def __init__(self, rows: Sequence[dict]) -> None:
        self._rows = tuple(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        return self._rows[index]
