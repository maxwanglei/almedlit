import importlib.util
import json

import pytest
from pydantic import ValidationError as PydanticValidationError

from al_medlit.training.model_types.evidence_conventional.config import (
    EvidenceRandomForestConfig,
    EvidenceSVMConfig,
)
from al_medlit.training.model_types.evidence_conventional.model import (
    canonical_sentence_groups,
    load_conventional_model,
    predict_window,
)
from al_medlit.training.registry import model_types
from al_medlit.training.runner import RunnerError, run_training_job, sha256_file


def _row(document_id: str, split: str, labels: tuple[str, ...]) -> dict:
    return {
        "schema_version": "training-windows-v1",
        "document_id": document_id,
        "target": {"id": 7, "text": "Find pharmacokinetic evidence"},
        "split": split,
        "sentences": [
            {
                "ordinal": index,
                "text": (
                    "Drug concentration and clearance were measured."
                    if label in {"B", "I"}
                    else "This sentence is general background."
                ),
                "label": label,
                "reviewed": True,
            }
            for index, label in enumerate(labels)
        ],
    }


def _manifest(tmp_path, model_type: str, config: dict, rows: list[dict]):
    dataset = tmp_path / "training.jsonl"
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "al-medlit-job-v1",
        "kind": "training",
        "job_key": f"training:{model_type}:attempt-1",
        "model_type": model_type,
        "target_version_id": None,
        "config": config,
        "dataset": {
            "path": dataset.name,
            "checksum_sha256": sha256_file(dataset),
        },
        "checkpoint_manifest": {
            "model_type": model_type,
            "training_mode": "conditioned",
            "trained_target_version_ids": [7],
        },
        "output_directory": "outputs",
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_conventional_plugins_have_typed_configs_and_honest_preflight():
    for key, dependencies in {
        "evidence_svm": ("sklearn", "skops"),
        "evidence_random_forest": ("sklearn", "skops"),
        "evidence_crf": ("sklearn_crfsuite",),
    }.items():
        plugin = model_types.get(key)
        preflight = plugin.preflight()
        missing = tuple(
            dependency
            for dependency in dependencies
            if importlib.util.find_spec(dependency) is None
        )
        assert preflight["available"] is (not missing)
        assert preflight["missing_dependencies"] == missing
        assert plugin.descriptor.family.value == "conventional_ml"
        assert plugin.descriptor.implementation_status == "implemented"

    validated = model_types.get("evidence_svm").validate_config({})
    assert isinstance(validated, EvidenceSVMConfig)
    assert validated.word_ngram_range == (1, 2)
    with pytest.raises(PydanticValidationError):
        EvidenceRandomForestConfig(unknown_option=True)


def test_overlapping_windows_are_deduplicated_before_feature_fitting():
    first = _row("doc-1", "train", ("B", "I"))
    overlap = _row("doc-1", "train", ("B", "I"))

    groups = canonical_sentence_groups([first, overlap])

    assert len(groups) == 1
    assert [record.ordinal for record in groups[0]] == [0, 1]
    overlap["sentences"][1]["label"] = "O"
    with pytest.raises(ValueError, match="inconsistent"):
        canonical_sentence_groups([first, overlap])


def test_runner_rejects_document_split_leakage_before_optional_dependency_use(tmp_path):
    manifest = _manifest(
        tmp_path,
        "evidence_svm",
        {},
        [
            _row("same-document", "train", ("B", "O")),
            _row("same-document", "validation", ("B", "O")),
            _row("test-document", "test", ("B", "O")),
        ],
    )

    with pytest.raises(RunnerError, match="leakage"):
        run_training_job(manifest)


def test_runner_requires_all_production_splits_before_optional_dependency_use(tmp_path):
    manifest = _manifest(
        tmp_path,
        "evidence_random_forest",
        {},
        [
            _row("train-document", "train", ("B", "O")),
            _row("validation-document", "validation", ("B", "O")),
        ],
    )

    with pytest.raises(RunnerError, match="missing: test"):
        run_training_job(manifest)


@pytest.mark.parametrize(
    ("model_type", "config"),
    [
        ("evidence_svm", {"max_features": 100, "bootstrap_samples": 0}),
        (
            "evidence_random_forest",
            {"max_features": 100, "n_estimators": 5, "bootstrap_samples": 0},
        ),
    ],
)
def test_sentence_scorer_runner_packages_reloads_and_evaluates_when_installed(
    tmp_path,
    model_type,
    config,
):
    if any(importlib.util.find_spec(name) is None for name in ("sklearn", "skops")):
        pytest.skip("conventional-ML optional dependencies are not installed")
    manifest = _manifest(
        tmp_path,
        model_type,
        config,
        [
            _row("train-positive", "train", ("B", "I")),
            _row("train-mixed", "train", ("O", "B")),
            _row("validation", "validation", ("B", "O")),
            _row("test", "test", ("O", "B")),
        ],
    )

    result = run_training_job(manifest)

    checkpoint = tmp_path / "outputs/checkpoint"
    assert (checkpoint / "model.skops").is_file()
    assert result["checkpoint_manifest"]["package_format"] == "skops"
    metrics = json.loads((tmp_path / "outputs/metrics.json").read_text())
    assert set(metrics["evaluations"]) == {"validation", "test"}
    assert metrics["selection_metric_key"] == "macro_block_iou_f1_0_50"
    loaded = load_conventional_model(checkpoint)
    scores = predict_window(
        loaded,
        target_text="Find pharmacokinetic evidence",
        sentences=["Clearance was measured.", "Background only."],
    )
    assert len(scores) == 2
    assert all(0 <= float(score) <= 1 for score in scores)


def test_crf_runner_packages_native_model_and_reloads_when_installed(tmp_path):
    if importlib.util.find_spec("sklearn_crfsuite") is None:
        pytest.skip("sklearn-crfsuite is not installed")
    manifest = _manifest(
        tmp_path,
        "evidence_crf",
        {"max_iterations": 10, "bootstrap_samples": 0},
        [
            _row("train-positive", "train", ("B", "I", "O")),
            _row("train-negative", "train", ("O", "O", "B")),
            _row("validation", "validation", ("B", "I", "O")),
            _row("test", "test", ("O", "B", "I")),
        ],
    )

    result = run_training_job(manifest)

    checkpoint = tmp_path / "outputs/checkpoint"
    assert (checkpoint / "model.crfsuite").is_file()
    assert result["checkpoint_manifest"]["package_format"] == "crfsuite"
    loaded = load_conventional_model(checkpoint)
    labels = predict_window(
        loaded,
        target_text="Find pharmacokinetic evidence",
        sentences=["Clearance was measured.", "Background only."],
    )
    assert len(labels) == 2
    assert set(labels).issubset({"O", "B", "I"})
