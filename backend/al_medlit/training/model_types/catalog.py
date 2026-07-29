"""Family-neutral catalog of built-in and planned Evidence model types."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable

from al_medlit.training.contracts import (
    AvailabilityDescriptor,
    AvailabilityStatus,
    MetricDescriptor,
    ModelCapabilities,
    ModelFamily,
    ModelTypeDescriptor,
    PanelDescriptor,
    PanelKind,
)
from al_medlit.training.model_types.evidence_block_sentence_tagger.config import (
    EvidenceBlockSentenceTaggerConfig,
)
from al_medlit.training.model_types.evidence_conventional.config import (
    EvidenceCRFConfig,
    EvidenceRandomForestConfig,
    EvidenceSVMConfig,
)
from al_medlit.training.model_types.evidence_neural.config import (
    EvidenceBiLSTMConfig,
    EvidenceCNNConfig,
)
from al_medlit.training.model_types.evidence_peft.config import (
    EvidenceLoRAConfig,
    EvidenceQLoRAConfig,
)

TASK_TYPE = "evidence_block_sentence_tagging"


COMMON_EVIDENCE_METRICS = (
    # Keys intentionally match EvaluationResult.metrics exactly. The evaluator
    # is the sole owner of comparable metric names across every model family.
    MetricDescriptor(key="sentence_f1", label="Sentence F1"),
    MetricDescriptor(key="sentence_precision", label="Sentence precision"),
    MetricDescriptor(key="sentence_recall", label="Sentence recall"),
    MetricDescriptor(key="exact_block_f1", label="Exact-block F1"),
    MetricDescriptor(key="block_iou_f1_0_25", label="Block IoU F1 @ 0.25"),
    MetricDescriptor(key="block_iou_f1_0_50", label="Block IoU F1 @ 0.50"),
    MetricDescriptor(key="block_iou_f1_0_75", label="Block IoU F1 @ 0.75"),
    MetricDescriptor(
        key="mean_start_boundary_deviation",
        label="Mean start-boundary deviation",
        direction="minimize",
        value_type="number",
    ),
    MetricDescriptor(
        key="mean_end_boundary_deviation",
        label="Mean end-boundary deviation",
        direction="minimize",
        value_type="number",
    ),
    MetricDescriptor(key="document_presence_f1", label="Document-presence F1"),
    MetricDescriptor(
        key="macro_block_iou_f1_0_50",
        label="Macro block IoU F1 @ 0.50",
    ),
    MetricDescriptor(key="macro_exact_block_f1", label="Macro exact-block F1"),
)

COMMON_PANELS = (
    PanelDescriptor(
        key="task_performance",
        label="Task performance",
        kind=PanelKind.METRIC_CARDS,
        data_keys=tuple(metric.key for metric in COMMON_EVIDENCE_METRICS),
    ),
    PanelDescriptor(
        key="document_presence_confusion",
        label="Document-presence confusion matrix",
        kind=PanelKind.CONFUSION_MATRIX,
        data_keys=("confusion_matrix.document_presence",),
    ),
    PanelDescriptor(
        key="per_target_results",
        label="Per-target results",
        kind=PanelKind.TABLE,
        data_keys=("diagnostics.per_target",),
    ),
)


def _dependency_availability(
    dependencies: Iterable[str],
    *,
    implemented: bool,
    experimental: bool = False,
    synthetic_available: bool = False,
) -> AvailabilityDescriptor:
    dependency_names = tuple(dependencies)
    if not implemented:
        return AvailabilityDescriptor(
            status=AvailabilityStatus.UNAVAILABLE,
            reason="The training plugin is not enabled for durable job execution in this build",
            missing_dependencies=(),
        )
    missing = tuple(
        dependency
        for dependency in dependency_names
        if importlib.util.find_spec(dependency) is None
    )
    if missing:
        return AvailabilityDescriptor(
            status=(
                AvailabilityStatus.EXPERIMENTAL if experimental else AvailabilityStatus.AVAILABLE
            ),
            reason=(
                "Optional dependencies are missing from the API process; the selected "
                "compute backend must pass preflight"
            ),
            missing_dependencies=missing,
            synthetic_available=synthetic_available,
        )
    return AvailabilityDescriptor(
        status=(AvailabilityStatus.EXPERIMENTAL if experimental else AvailabilityStatus.AVAILABLE),
        synthetic_available=synthetic_available,
    )


def _descriptor(
    *,
    key: str,
    label: str,
    family: ModelFamily,
    model_kind: str,
    implementation_status: str,
    dependencies: tuple[str, ...],
    artifact_formats: tuple[str, ...],
    config_schema: dict,
    devices: tuple[str, ...],
    panels: tuple[PanelDescriptor, ...],
    additional_metrics: tuple[MetricDescriptor, ...] = (),
    produces_sentence_scores: bool = False,
    resume: bool = False,
    requires_device_preflight: bool = False,
    synthetic_available: bool = False,
) -> ModelTypeDescriptor:
    implemented = implementation_status in {"implemented", "experimental"}
    return ModelTypeDescriptor(
        key=key,
        label=label,
        family=family,
        model_kind=model_kind,
        task_type=TASK_TYPE,
        description=f"{label} for sentence-aligned Evidence Block detection",
        implementation_status=implementation_status,
        availability=_dependency_availability(
            dependencies,
            implemented=implemented,
            experimental=implementation_status == "experimental",
            synthetic_available=synthetic_available,
        ),
        capabilities=ModelCapabilities(
            resume=resume,
            supported_devices=devices,
            artifact_formats=artifact_formats,
            required_worker_dependencies=dependencies,
            requires_device_preflight=requires_device_preflight,
            produces_sentence_scores=produces_sentence_scores,
            produces_sequence_labels=not produces_sentence_scores,
        ),
        config_schema=config_schema,
        metrics=COMMON_EVIDENCE_METRICS + additional_metrics,
        panels=COMMON_PANELS + panels,
    )


def builtin_model_descriptors() -> tuple[ModelTypeDescriptor, ...]:
    conventional_search_panel = PanelDescriptor(
        key="cross_validation",
        label="Cross-validation and search",
        kind=PanelKind.TABLE,
        data_keys=("diagnostics.cross_validation",),
        optional=True,
    )
    feature_panel = PanelDescriptor(
        key="feature_importance",
        label="Feature importance",
        kind=PanelKind.BAR_CHART,
        data_keys=("diagnostics.feature_importance",),
        optional=True,
    )
    learning_panel = PanelDescriptor(
        key="learning_curves",
        label="Learning curves",
        kind=PanelKind.LINE_CHART,
        data_keys=("metric_series.train_loss", "metric_series.validation_loss"),
        optional=True,
    )
    resource_panel = PanelDescriptor(
        key="resource_utilization",
        label="Resource utilization",
        kind=PanelKind.LINE_CHART,
        data_keys=("metric_series.throughput", "metric_series.device_memory_bytes"),
        optional=True,
    )
    adapter_panel = PanelDescriptor(
        key="adapter_details",
        label="Adapter details",
        kind=PanelKind.TABLE,
        data_keys=("diagnostics.adapter", "diagnostics.base_model"),
    )
    llm_metric_descriptors = (
        MetricDescriptor(
            key="diagnostics.token_loss",
            label="Token loss",
            direction="minimize",
            value_type="number",
            comparable=False,
        ),
        MetricDescriptor(
            key="diagnostics.perplexity",
            label="Perplexity",
            direction="minimize",
            value_type="number",
            comparable=False,
            description="Comparable only with identical tokenizer, vocabulary, and text",
        ),
        MetricDescriptor(
            key="diagnostics.tokens_per_second",
            label="Tokens per second",
            value_type="number",
            comparable=False,
        ),
        MetricDescriptor(
            key="diagnostics.structured_output_validity",
            label="Structured-output validity",
            comparable=False,
        ),
    )
    llm_diagnostics_panel = PanelDescriptor(
        key="llm_diagnostics",
        label="LLM fine-tuning diagnostics",
        kind=PanelKind.METRIC_CARDS,
        data_keys=tuple(metric.key for metric in llm_metric_descriptors),
        optional=True,
    )

    return (
        _descriptor(
            key="evidence_crf",
            label="CRF baseline",
            family=ModelFamily.CONVENTIONAL_ML,
            model_kind="crf",
            implementation_status="implemented",
            dependencies=("sklearn_crfsuite",),
            artifact_formats=("crfsuite",),
            config_schema=EvidenceCRFConfig.model_json_schema(),
            devices=("cpu",),
            panels=(conventional_search_panel, feature_panel),
        ),
        _descriptor(
            key="evidence_svm",
            label="SVM baseline",
            family=ModelFamily.CONVENTIONAL_ML,
            model_kind="svm",
            implementation_status="implemented",
            dependencies=("sklearn", "skops"),
            artifact_formats=("skops",),
            config_schema=EvidenceSVMConfig.model_json_schema(),
            devices=("cpu",),
            panels=(conventional_search_panel, feature_panel),
            produces_sentence_scores=True,
        ),
        _descriptor(
            key="evidence_random_forest",
            label="Random Forest baseline",
            family=ModelFamily.CONVENTIONAL_ML,
            model_kind="random_forest",
            implementation_status="implemented",
            dependencies=("sklearn", "skops"),
            artifact_formats=("skops",),
            config_schema=EvidenceRandomForestConfig.model_json_schema(),
            devices=("cpu",),
            panels=(conventional_search_panel, feature_panel),
            produces_sentence_scores=True,
        ),
        _descriptor(
            key="evidence_bilstm",
            label="BiLSTM",
            family=ModelFamily.DEEP_LEARNING,
            model_kind="bilstm",
            implementation_status="implemented",
            dependencies=("torch", "safetensors"),
            artifact_formats=("safetensors", "json"),
            config_schema=EvidenceBiLSTMConfig.model_json_schema(),
            devices=("cpu", "mps", "cuda", "ascend"),
            panels=(learning_panel, resource_panel),
            # Deployable packages intentionally omit optimizer/scheduler/RNG state.
            resume=False,
            requires_device_preflight=True,
        ),
        _descriptor(
            key="evidence_cnn",
            label="Sentence CNN",
            family=ModelFamily.DEEP_LEARNING,
            model_kind="cnn",
            implementation_status="implemented",
            dependencies=("torch", "safetensors"),
            artifact_formats=("safetensors", "json"),
            config_schema=EvidenceCNNConfig.model_json_schema(),
            devices=("cpu", "mps", "cuda", "ascend"),
            panels=(learning_panel, resource_panel),
            resume=False,
            requires_device_preflight=True,
        ),
        _descriptor(
            key="evidence_block_sentence_tagger",
            label="Evidence transformer",
            family=ModelFamily.DEEP_LEARNING,
            model_kind="transformer",
            implementation_status="implemented",
            dependencies=("torch", "transformers", "safetensors"),
            artifact_formats=("safetensors", "huggingface_json", "tokenizer"),
            config_schema=EvidenceBlockSentenceTaggerConfig.model_json_schema(),
            devices=("cpu", "mps", "cuda", "ascend"),
            panels=(learning_panel, resource_panel),
            resume=False,
            requires_device_preflight=True,
            synthetic_available=True,
        ),
        _descriptor(
            key="evidence_lora",
            label="LLM LoRA",
            family=ModelFamily.LLM_FINETUNE,
            model_kind="lora",
            implementation_status="implemented",
            dependencies=("torch", "transformers", "peft", "safetensors"),
            artifact_formats=("peft_adapter_safetensors", "json"),
            config_schema=EvidenceLoRAConfig.model_json_schema(),
            devices=("cuda", "ascend"),
            panels=(
                learning_panel,
                resource_panel,
                adapter_panel,
                llm_diagnostics_panel,
            ),
            additional_metrics=llm_metric_descriptors,
            resume=False,
            requires_device_preflight=True,
        ),
        _descriptor(
            key="evidence_qlora",
            label="LLM QLoRA",
            family=ModelFamily.LLM_FINETUNE,
            model_kind="qlora",
            implementation_status="experimental",
            dependencies=(
                "torch",
                "transformers",
                "peft",
                "bitsandbytes",
                "safetensors",
            ),
            artifact_formats=("peft_adapter_safetensors", "json"),
            config_schema=EvidenceQLoRAConfig.model_json_schema(),
            devices=("cuda",),
            panels=(
                learning_panel,
                resource_panel,
                adapter_panel,
                llm_diagnostics_panel,
            ),
            additional_metrics=llm_metric_descriptors,
            resume=False,
            requires_device_preflight=True,
        ),
    )


def get_builtin_model_descriptor(key: str) -> ModelTypeDescriptor:
    for descriptor in builtin_model_descriptors():
        if descriptor.key == key:
            return descriptor
    raise KeyError(key)
