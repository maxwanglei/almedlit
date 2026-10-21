"""SFT output must satisfy the package policy used by training publication."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from test_model_artifacts import _neural_files, _neural_spec, _workspace_project
from test_trainer_plugins import _sft_config

from al_medlit.model_artifacts.schemas import (
    ArtifactPackageCreate,
    ArtifactPackageReferenceCreate,
)
from al_medlit.model_artifacts.service import publish_artifact_package
from al_medlit.training.recipe_registry import training_recipes
from al_medlit.training.run_execution import _package_contract, _safe_output_files
from al_medlit.training.trainers import (
    HuggingFaceCausalSftTrainer,
    HuggingFaceSeq2SeqSftTrainer,
    huggingface_sft,
)
from al_medlit.training.trainers.contracts import TrainerPreflight, TrainingInput


class _Tokenizer:
    eos_token_id = 2

    def __call__(self, text=None, **kwargs):
        return {"input_ids": [3, 4], "attention_mask": [1, 1]}

    def save_pretrained(self, destination: Path):
        destination.mkdir(parents=True)
        (destination / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _Model:
    def __init__(self, *, adapter):
        self.adapter = adapter

    def save_pretrained(self, destination: Path, *, safe_serialization):
        assert safe_serialization is True
        destination.mkdir(parents=True, exist_ok=True)
        config_name = "adapter_config.json" if self.adapter else "config.json"
        weights_name = "adapter_model.safetensors" if self.adapter else "model.safetensors"
        (destination / config_name).write_text('{"peft_type":"LORA"}', encoding="utf-8")
        (destination / weights_name).write_bytes(b"safe-weights-fixture")


@pytest.mark.parametrize("architecture", ["causal_lm", "seq2seq_lm"])
@pytest.mark.parametrize("parameterization", ["full", "lora", "qlora"])
@pytest.mark.parametrize("with_validation", [False, True])
def test_sft_outputs_publish_as_deployable_packages(
    db, object_storage, tmp_path, monkeypatch, architecture, parameterization, with_validation
):
    trainer = (
        HuggingFaceCausalSftTrainer()
        if architecture == "causal_lm"
        else HuggingFaceSeq2SeqSftTrainer()
    )
    recipe_key = f"{architecture}_sft_{parameterization}"
    descriptor = training_recipes.get(recipe_key)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    transformers = ModuleType("transformers")
    transformers.set_seed = lambda seed: None
    transformers.DataCollatorForSeq2Seq = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        trainer,
        "preflight",
        lambda _recipe: TrainerPreflight(
            ready=True, runtime_class=descriptor.environment.runtime_class.value, checks=()
        ),
    )
    monkeypatch.setattr(trainer, "_load_tokenizer", lambda *_args: _Tokenizer())
    monkeypatch.setattr(
        trainer,
        "_load_model",
        lambda **_kwargs: (_Model(adapter=parameterization != "full"), False),
    )
    monkeypatch.setattr(
        huggingface_sft,
        "execute_torch_training",
        lambda **kwargs: {"validation_loss": 0.25 if kwargs["validation_rows"] else None},
    )
    base_model_path = tmp_path / "base-model"
    base_model_path.mkdir()
    destination = tmp_path / "output"
    output = trainer.train(
        recipe_key=recipe_key,
        config=_sft_config(),
        training_input=TrainingInput(
            rows=({"prompt": "Question", "answer": "Answer"},),
            validation_rows=({"prompt": "Validation", "answer": "Answer"},)
            if with_validation
            else (),
            dataset_fingerprint="sft-dataset",
            split_fingerprint="sft-split",
            task_kind="instruction_tuning",
            base_model_fingerprint="sft-base",
            base_model_path=str(base_model_path),
        ),
        destination=destination,
        seed=42,
    )
    user, _workspace, project = _workspace_project(db, suffix="sft")
    base = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=_neural_spec(package_kind="base_model", model_family="llm_finetune"),
        files=_neural_files(),
        actor_user_id=user.id,
    )
    package_kind, package_format = _package_contract(descriptor)
    package = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=ArtifactPackageCreate(
            package_kind=package_kind,
            package_format=package_format,
            model_family=descriptor.model_family.value,
            model_type=descriptor.architecture_family,
            deployable=True,
            references=[
                ArtifactPackageReferenceCreate(
                    target_package_id=base.id, relationship_type="uses_base_model"
                )
            ],
        ),
        files=_safe_output_files(output, destination),
        actor_user_id=user.id,
    )
    db.commit()

    assert package.package_kind == (
        "trained_model" if parameterization == "full" else "peft_adapter"
    )
    assert package.deployable is True
    paths = {item.relative_path for item in package.files}
    if parameterization == "full":
        assert {"model/config.json", "model/model.safetensors"} <= paths
        weights_path = "model/model.safetensors"
    else:
        assert {"adapter_config.json", "adapter_model.safetensors"} <= paths
        weights_path = "adapter_model.safetensors"
    assert not (destination / "adapter").exists()
    assert "tokenizer/tokenizer_config.json" in paths
    assert "sft_contract.json" in paths
    assert output.manifest["files"][weights_path]
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == output.manifest
