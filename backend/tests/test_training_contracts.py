import pytest
from pydantic import ValidationError as PydanticValidationError

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.contracts import ModelFamily, PanelDescriptor, PanelKind
from al_medlit.training.model_types.evidence_block_sentence_tagger.plugin import (
    EvidenceBlockSentenceTaggerPlugin,
)
from al_medlit.training.registry import ModelTypeRegistry, model_types


def test_builtin_catalog_covers_all_planned_model_families_and_types():
    descriptors = {
        descriptor.model_kind: descriptor
        for descriptor in model_types.list_descriptors()
    }

    assert set(descriptors) == {
        "crf",
        "svm",
        "random_forest",
        "bilstm",
        "cnn",
        "transformer",
        "lora",
        "qlora",
    }
    assert descriptors["crf"].family == ModelFamily.CONVENTIONAL_ML
    assert descriptors["transformer"].family == ModelFamily.DEEP_LEARNING
    assert descriptors["qlora"].family == ModelFamily.LLM_FINETUNE
    assert descriptors["transformer"].implementation_status == "implemented"
    assert descriptors["transformer"].availability.synthetic_available is True
    assert descriptors["transformer"].availability.available is True
    assert all(descriptor.capabilities.resume is False for descriptor in descriptors.values())
    assert descriptors["transformer"].capabilities.supported_devices == (
        "cpu",
        "mps",
        "cuda",
        "ascend",
    )
    serialized = descriptors["transformer"].model_dump(mode="json")
    assert serialized["availability"]["available"] is True
    assert serialized["family"] == "deep_learning"
    assert descriptors["crf"].implementation_status == "implemented"
    assert descriptors["svm"].implementation_status == "implemented"
    assert descriptors["random_forest"].implementation_status == "implemented"


def test_descriptors_expose_only_supported_panel_primitives_and_safe_formats():
    supported = set(PanelKind)
    for descriptor in model_types.list_descriptors():
        assert descriptor.config_schema["type"] == "object"
        assert descriptor.capabilities.artifact_formats
        assert all(panel.kind in supported for panel in descriptor.panels)
        assert not any("pickle" in value for value in descriptor.capabilities.artifact_formats)

    with pytest.raises(PydanticValidationError, match="data keys only"):
        PanelDescriptor(
            key="unsafe",
            label="Unsafe",
            kind=PanelKind.TABLE,
            data_keys=("javascript:alert(1)",),
        )


def test_plugin_registration_replaces_catalog_descriptor_with_matching_contract():
    registry = ModelTypeRegistry()
    registry.register_builtin_descriptors()
    plugin = EvidenceBlockSentenceTaggerPlugin()

    registry.register(plugin)

    assert registry.get(plugin.key) is plugin
    assert registry.get_descriptor(plugin.key).model_kind == "transformer"


def test_registry_rejects_unknown_plugins_and_duplicate_descriptors():
    registry = ModelTypeRegistry()
    registry.register_builtin_descriptors()

    with pytest.raises(ValidationError, match="Unknown model type"):
        registry.get("does_not_exist")
    with pytest.raises(ValidationError, match="already registered"):
        registry.register_descriptor(model_types.get_descriptor("evidence_crf"))
