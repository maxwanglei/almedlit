"""Adapter-only PEFT build, package, and load primitives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from al_medlit.training.model_types.evidence_peft.config import (
    EvidenceLoRAConfig,
    EvidenceQLoRAConfig,
    ImmutableBaseModelReference,
    validate_base_reference,
)
from al_medlit.training.model_types.evidence_peft.preflight import require_peft_preflight


class PeftAdapterPackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["evidence-peft-adapter-v1"] = "evidence-peft-adapter-v1"
    model_family: Literal["llm_finetune"] = "llm_finetune"
    model_type: Literal["evidence_lora", "evidence_qlora"]
    task_contract: dict = Field(default_factory=lambda: {"key": "evidence_blocks", "version": "1"})
    base_model: ImmutableBaseModelReference
    adapter_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_weights_size_bytes: int = Field(ge=0)
    safe_serialization: Literal[True] = True
    deployable: Literal[True] = True

    def package_reference(self) -> dict:
        return {
            "target_package_id": self.base_model.package_id,
            "relationship_type": "uses_base_model",
            "metadata": {
                "base_model_asset_id": self.base_model.asset_id,
                "base_model_manifest_digest": self.base_model.manifest_digest,
                "base_model_exact_revision": self.base_model.exact_revision,
            },
        }


@dataclass(slots=True)
class PeftAdapterBundle:
    model: object
    config: EvidenceLoRAConfig | EvidenceQLoRAConfig
    base_model: ImmutableBaseModelReference

    def save_pretrained(self, destination: str | Path) -> PeftAdapterPackageManifest:
        validate_base_reference(self.config, self.base_model)
        target = Path(destination)
        if target.exists() and target.is_symlink():
            raise ValueError("Adapter destination cannot be a symlink")
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target, safe_serialization=True)
        unsafe_files = tuple(
            path
            for path in target.rglob("*")
            if path.is_symlink()
            or path.suffix.lower() in {".bin", ".pkl", ".pickle", ".pt", ".pth"}
        )
        if unsafe_files:
            raise ValueError("PEFT package contains an unsafe executable or symlinked file")
        adapter_config_path = target / "adapter_config.json"
        adapter_weights_path = target / "adapter_model.safetensors"
        for path in (adapter_config_path, adapter_weights_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    "PEFT packaging must produce adapter_config.json and "
                    "adapter_model.safetensors as regular files"
                )
        if adapter_config_path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("PEFT adapter configuration exceeds the 4 MiB safety limit")
        raw_adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        if not isinstance(raw_adapter_config, dict):
            raise ValueError("PEFT adapter configuration must be a JSON object")
        # Avoid leaking a worker-local materialization path while retaining an
        # immutable platform reference that the loader can verify.
        raw_adapter_config["base_model_name_or_path"] = (
            f"al-medlit-base-asset:{self.base_model.asset_id}"
        )
        raw_adapter_config["revision"] = self.base_model.exact_revision
        raw_adapter_config["auto_mapping"] = None
        raw_adapter_config["peft_type"] = "LORA"
        adapter_config_path.write_text(
            json.dumps(raw_adapter_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = PeftAdapterPackageManifest(
            model_type=f"evidence_{self.config.model_kind}",
            base_model=self.base_model,
            adapter_config_sha256=_sha256(adapter_config_path),
            adapter_weights_sha256=_sha256(adapter_weights_path),
            adapter_weights_size_bytes=adapter_weights_path.stat().st_size,
        )
        (target / "al-medlit-adapter.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        (target / "al-medlit-training-config.json").write_text(
            self.config.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest


def create_peft_adapter(
    base_model: object,
    config: EvidenceLoRAConfig | EvidenceQLoRAConfig,
    base_reference: ImmutableBaseModelReference,
) -> PeftAdapterBundle:
    """Attach an adapter to an already materialized, exact base model.

    This function never resolves or downloads a base model. The orchestrator
    owns catalog authorization, checksum verification, and materialization.
    """

    validate_base_reference(config, base_reference)
    require_peft_preflight(config)
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover - guarded by preflight
        raise RuntimeError("PEFT dependency became unavailable after preflight") from exc
    if isinstance(config, EvidenceQLoRAConfig):
        if not bool(getattr(base_model, "is_loaded_in_4bit", False)):
            raise ValueError("QLoRA requires a base model materialized in 4-bit mode")
        base_model = prepare_model_for_kbit_training(base_model)
    adapter_config = LoraConfig(
        task_type=config.peft_task_type,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=config.bias,
        target_modules=list(config.target_modules),
        inference_mode=False,
    )
    return PeftAdapterBundle(
        model=get_peft_model(base_model, adapter_config),
        config=config,
        base_model=base_reference,
    )


def load_peft_adapter(
    checkpoint_directory: str | Path,
    *,
    base_model: object,
    base_reference: ImmutableBaseModelReference,
    trainable: bool = False,
) -> PeftAdapterBundle:
    root = Path(checkpoint_directory)
    manifest_path = root / "al-medlit-adapter.json"
    training_config_path = root / "al-medlit-training-config.json"
    required = (
        manifest_path,
        training_config_path,
        root / "adapter_config.json",
        root / "adapter_model.safetensors",
    )
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise ValueError("Adapter package is incomplete or contains symlinked files")
    manifest = PeftAdapterPackageManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.base_model != base_reference:
        raise ValueError("Materialized base model does not match adapter package lineage")
    if _sha256(required[2]) != manifest.adapter_config_sha256:
        raise ValueError("Adapter configuration checksum mismatch")
    if _sha256(required[3]) != manifest.adapter_weights_sha256:
        raise ValueError("Adapter weights checksum mismatch")
    raw_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    if raw_config.get("model_kind") == "lora":
        config = EvidenceLoRAConfig.model_validate(raw_config)
    elif raw_config.get("model_kind") == "qlora":
        config = EvidenceQLoRAConfig.model_validate(raw_config)
    else:
        raise ValueError("Unsupported adapter model_kind")
    validate_base_reference(config, base_reference)
    require_peft_preflight(config)
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - guarded by preflight
        raise RuntimeError("PEFT dependency became unavailable after preflight") from exc
    model = PeftModel.from_pretrained(
        base_model,
        root,
        is_trainable=trainable,
        local_files_only=True,
    )
    return PeftAdapterBundle(model=model, config=config, base_model=base_reference)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
