import json
import zipfile
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from al_medlit.training.model_types.evidence_peft import (
    EvidenceLoRAConfig,
    EvidenceLoRAPlugin,
    EvidenceQLoRAConfig,
    EvidenceQLoRAPlugin,
    ImmutableBaseModelReference,
    PeftAdapterBundle,
    preflight_peft,
    register_builtin_peft_model_types,
)
from al_medlit.training.model_types.evidence_peft.config import validate_base_reference
from al_medlit.training.model_types.evidence_peft.training import (
    parse_evidence_completion,
    prepare_peft_examples,
    render_evidence_completion,
)
from al_medlit.training.registry import ModelTypeRegistry
from al_medlit.training.runner import _train_peft_checkpoint


def _reference(asset_id=7):
    return ImmutableBaseModelReference(
        asset_id=asset_id,
        package_id=13,
        manifest_digest="a" * 64,
        exact_revision="0123456789abcdef",
    )


def test_peft_config_selects_catalog_asset_and_rejects_unsafe_targets():
    config = EvidenceLoRAConfig(base_model_asset_id=7)
    assert config.base_model_asset_id == 7
    assert config.target_conditioning is True
    assert config.rank == 16
    assert config.reload_best_checkpoint is True
    validate_base_reference(config, _reference())

    with pytest.raises(ValidationError, match="safe module names"):
        EvidenceLoRAConfig(base_model_asset_id=7, target_modules=("../../module",))
    with pytest.raises(ValueError, match="does not match"):
        validate_base_reference(config, _reference(asset_id=8))
    with pytest.raises(ValidationError, match="exact immutable revision"):
        ImmutableBaseModelReference(
            asset_id=7,
            package_id=13,
            manifest_digest="a" * 64,
            exact_revision="latest",
        )


def test_qlora_preflight_fails_closed_without_bitsandbytes():
    config = EvidenceQLoRAConfig(base_model_asset_id=7)
    fake_torch = SimpleNamespace(
        __version__="2.5.0",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
        ),
        backends=SimpleNamespace(),
        version=SimpleNamespace(cuda="12.4"),
    )

    def loader(name):
        if name == "torch":
            return fake_torch
        if name == "bitsandbytes":
            raise ModuleNotFoundError(name)
        return SimpleNamespace()

    result = preflight_peft(config, module_loader=loader)
    assert result.available is False
    assert result.quantization_ready is False
    assert result.missing_dependencies == ("bitsandbytes",)


def test_qlora_preflight_requires_proven_cuda_4bit_integration():
    config = EvidenceQLoRAConfig(base_model_asset_id=7)
    fake_torch = SimpleNamespace(
        __version__="2.5.0",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
        ),
        backends=SimpleNamespace(),
        version=SimpleNamespace(cuda="12.4"),
    )
    modules = {
        "torch": fake_torch,
        "transformers": SimpleNamespace(BitsAndBytesConfig=object),
        "peft": SimpleNamespace(),
        "safetensors": SimpleNamespace(),
        "bitsandbytes": SimpleNamespace(
            __version__="0.45.0",
            nn=SimpleNamespace(Linear4bit=object),
        ),
    }
    result = preflight_peft(config, module_loader=modules.__getitem__)
    assert result.available is True
    assert result.quantization_ready is True
    assert result.details["cuda_runtime"] == "12.4"

    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        is_bf16_supported=lambda: False,
    )
    unavailable = preflight_peft(config, module_loader=modules.__getitem__)
    assert unavailable.available is False
    assert unavailable.quantization_ready is False


def test_adapter_package_is_safetensors_only_and_records_immutable_base(tmp_path):
    class FakeAdapter:
        def save_pretrained(self, destination, *, safe_serialization):
            assert safe_serialization is True
            (destination / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/worker/private/base"}),
                encoding="utf-8",
            )
            (destination / "adapter_model.safetensors").write_bytes(b"safe-adapter")

    config = EvidenceLoRAConfig(base_model_asset_id=7)
    bundle = PeftAdapterBundle(
        model=FakeAdapter(),
        config=config,
        base_model=_reference(),
    )
    manifest = bundle.save_pretrained(tmp_path)
    sanitized = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert sanitized["base_model_name_or_path"] == "al-medlit-base-asset:7"
    assert "/worker/private/base" not in (tmp_path / "adapter_config.json").read_text(
        encoding="utf-8"
    )
    assert manifest.base_model.package_id == 13
    assert manifest.package_reference() == {
        "target_package_id": 13,
        "relationship_type": "uses_base_model",
        "metadata": {
            "base_model_asset_id": 7,
            "base_model_manifest_digest": "a" * 64,
            "base_model_exact_revision": "0123456789abcdef",
        },
    }
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert not tuple(tmp_path.glob("*.bin"))


def test_peft_plugins_register_after_runner_and_asset_resolution_are_wired():
    registry = ModelTypeRegistry()
    registry.register_builtin_descriptors()
    register_builtin_peft_model_types(registry)

    assert isinstance(registry.get("evidence_lora"), EvidenceLoRAPlugin)
    assert isinstance(registry.get("evidence_qlora"), EvidenceQLoRAPlugin)
    assert registry.get_descriptor("evidence_lora").implementation_status == "implemented"
    assert registry.get_descriptor("evidence_qlora").implementation_status == "experimental"


def test_peft_preparation_uses_exact_tokenizer_and_strict_structured_output():
    class FakeTokenizer:
        eos_token_id = 2

        def __call__(self, text, **_kwargs):
            return {"input_ids": list(range(max(1, len(text.split()))))}

    rows = [
        {
            "document_id": "doc-1",
            "target": {"id": 7, "text": "Safety evidence"},
            "sentences": [
                {"ordinal": 0, "text": "No event.", "label": "O", "reviewed": True},
                {"ordinal": 1, "text": "Rash occurred.", "label": "B", "reviewed": True},
                {
                    "ordinal": 2,
                    "text": "Unreviewed sentence.",
                    "label": "IGNORE",
                    "reviewed": False,
                },
            ],
        }
    ]
    config = EvidenceLoRAConfig(
        base_model_asset_id=7,
        max_sequence_length=128,
    )
    examples = prepare_peft_examples(rows, FakeTokenizer(), config)

    assert len(examples) == 1
    assert examples[0].sentence_ordinals == (0, 1)
    assert examples[0].gold_labels == ("O", "B")
    completion = render_evidence_completion(examples[0].gold_labels)
    assert parse_evidence_completion(completion, expected_count=2) == ("O", "B")
    assert parse_evidence_completion("prefix " + completion, expected_count=2) is None
    assert parse_evidence_completion('{"labels":["O"]}', expected_count=2) is None


def test_durable_peft_runner_resolves_exact_base_and_packages_adapter(
    tmp_path,
    monkeypatch,
):
    dataset = tmp_path / "training.jsonl"
    rows = [
        {
            "schema_version": "training-windows-v1",
            "document_id": f"doc-{split}",
            "target": {"id": 7, "text": "Safety evidence"},
            "split": split,
            "sentences": [
                {"ordinal": 0, "text": "Rash occurred.", "label": "B", "reviewed": True}
            ],
        }
        for split in ("train", "validation", "test")
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    base_archive = tmp_path / "base.zip"
    with zipfile.ZipFile(base_archive, "w") as archive:
        archive.writestr("config.json", "{}")

    captured = {}

    def fake_train(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            bundle=object(),
            history=(
                {
                    "phase": "train",
                    "split": "train",
                    "epoch": 1,
                    "step": 2,
                    "values": {"token_loss": 0.25},
                },
            ),
            best_epoch=1,
            selection_value=0.75,
            selection_tiebreaker=0.5,
            evaluations={"validation": {"metrics": {}}, "test": {"metrics": {}}},
            device="cuda",
            adapter_parameter_count=16,
            trainable_parameter_count=16,
        )

    def fake_package(_self, _bundle, destination):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapter_config.json").write_text("{}", encoding="utf-8")
        (destination / "adapter_model.safetensors").write_bytes(b"adapter")
        return SimpleNamespace(model_dump=lambda **_kwargs: {"safe_serialization": True})

    monkeypatch.setattr(
        "al_medlit.training.model_types.evidence_peft.training.train_peft_adapter",
        fake_train,
    )
    monkeypatch.setattr(EvidenceLoRAPlugin, "package", fake_package)
    checkpoint = tmp_path / "checkpoint"
    result = _train_peft_checkpoint(
        checkpoint,
        dataset_path=dataset,
        target_version_id=None,
        model_type="evidence_lora",
        raw_config={"base_model_asset_id": 7, "target_conditioning": True},
        staged_model_archive=base_archive,
        staged_model_metadata={
            "base_model_asset_id": 7,
            "base_model_package_id": 13,
            "base_model_manifest_sha256": "a" * 64,
            "base_model_exact_revision": "0123456789abcdef",
        },
    )

    assert captured["base_reference"] == _reference()
    assert len(captured["training_rows"]) == 1
    assert len(captured["validation_rows"]) == 1
    assert len(captured["test_rows"]) == 1
    assert result["model_family"] == "llm_finetune"
    assert result["package_format"] == "peft_adapter_safetensors"
    assert result["_evaluations"]["test"] == {"metrics": {}}
    assert (checkpoint / "adapter_model.safetensors").is_file()
