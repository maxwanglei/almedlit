"""PEFT adapter support for the Evidence task."""

from al_medlit.training.model_types.evidence_peft.adapter import (
    PeftAdapterBundle,
    PeftAdapterPackageManifest,
    create_peft_adapter,
    load_peft_adapter,
)
from al_medlit.training.model_types.evidence_peft.config import (
    EvidenceLoRAConfig,
    EvidenceQLoRAConfig,
    ImmutableBaseModelReference,
)
from al_medlit.training.model_types.evidence_peft.plugin import (
    EvidenceLoRAPlugin,
    EvidenceQLoRAPlugin,
    register_builtin_peft_model_types,
)
from al_medlit.training.model_types.evidence_peft.preflight import (
    PeftPreflightResult,
    build_qlora_quantization_config,
    preflight_peft,
    require_peft_preflight,
)

__all__ = [
    "EvidenceLoRAConfig",
    "EvidenceLoRAPlugin",
    "EvidenceQLoRAConfig",
    "EvidenceQLoRAPlugin",
    "ImmutableBaseModelReference",
    "PeftAdapterBundle",
    "PeftAdapterPackageManifest",
    "PeftPreflightResult",
    "create_peft_adapter",
    "build_qlora_quantization_config",
    "load_peft_adapter",
    "preflight_peft",
    "register_builtin_peft_model_types",
    "require_peft_preflight",
]
