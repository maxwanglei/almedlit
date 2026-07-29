"""Dependency-isolated training and loading for conventional Evidence models.

The module deliberately imports scikit-learn, skops, and sklearn-crfsuite only
inside executable operations.  API processes can therefore expose honest
descriptors even when a particular worker image lacks the optional extra.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from al_medlit.training.model_types.evidence_conventional.config import (
    EvidenceConventionalConfig,
    EvidenceCRFConfig,
    EvidenceRandomForestConfig,
    EvidenceSVMConfig,
)

CONVENTIONAL_MODEL_TYPES = {
    "evidence_crf",
    "evidence_svm",
    "evidence_random_forest",
}
MODEL_SCHEMA_VERSION = "al-medlit-conventional-evidence-v1"


class MissingConventionalDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SentenceRecord:
    document_id: str
    target_version_id: str
    target_text: str
    ordinal: int
    text: str
    label: Literal["O", "B", "I"]


@dataclass(frozen=True, slots=True)
class LoadedConventionalModel:
    model_type: str
    config: EvidenceSVMConfig | EvidenceRandomForestConfig | EvidenceCRFConfig
    estimator: object

    @property
    def produces_sentence_scores(self) -> bool:
        return self.model_type in {"evidence_svm", "evidence_random_forest"}


def dependency_preflight(model_type: str) -> dict[str, Any]:
    dependencies = (
        ("sklearn_crfsuite",)
        if model_type == "evidence_crf"
        else ("sklearn", "skops")
    )
    missing = tuple(
        dependency
        for dependency in dependencies
        if importlib.util.find_spec(dependency) is None
    )
    return {
        "available": not missing,
        "model_type": model_type,
        "required_dependencies": dependencies,
        "missing_dependencies": missing,
        "supported_devices": ("cpu",),
        "reason": (
            None
            if not missing
            else "The selected worker is missing optional conventional-ML dependencies"
        ),
    }


def canonical_sentence_groups(rows: list[dict]) -> list[list[SentenceRecord]]:
    """Merge overlapping windows into deterministic document-target sequences."""

    groups: dict[tuple[str, str], dict[int, SentenceRecord]] = {}
    for row_index, row in enumerate(rows):
        target = row.get("target") or {}
        target_id = str(target.get("id", "unknown"))
        document_id = str(
            row.get("document_id", row.get("stable_key", f"window-{row_index}"))
        )
        group = groups.setdefault((document_id, target_id), {})
        for sentence_index, sentence in enumerate(row.get("sentences") or []):
            label = str(sentence.get("label", "IGNORE"))
            if label == "IGNORE" or sentence.get("reviewed") is False:
                continue
            if label not in {"O", "B", "I"}:
                raise ValueError(f"Unsupported Evidence label {label!r}")
            ordinal = int(sentence.get("ordinal", sentence_index))
            record = SentenceRecord(
                document_id=document_id,
                target_version_id=target_id,
                target_text=str(target.get("text", "")),
                ordinal=ordinal,
                text=str(sentence.get("text", "")),
                label=label,
            )
            previous = group.get(ordinal)
            if previous is not None and previous != record:
                raise ValueError(
                    "Overlapping training windows contain inconsistent sentence data"
                )
            group[ordinal] = record
    return [
        [sentences[ordinal] for ordinal in sorted(sentences)]
        for _, sentences in sorted(groups.items())
        if sentences
    ]


def _prepared_text(
    target_text: str,
    sentence_text: str,
    *,
    target_conditioning: bool,
) -> str:
    if target_conditioning:
        return f"[TARGET] {target_text}\n[SENTENCE] {sentence_text}"
    return sentence_text


def _require_sklearn_and_skops():
    try:
        import sklearn
        import skops.io as skops_io
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:  # pragma: no cover - depends on worker image
        raise MissingConventionalDependencyError(
            "SVM and Random Forest training require the optional "
            "scikit-learn/skops conventional-ML extra"
        ) from exc
    return {
        "sklearn": sklearn,
        "skops_io": skops_io,
        "DummyClassifier": DummyClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "TfidfVectorizer": TfidfVectorizer,
        "Pipeline": Pipeline,
        "LinearSVC": LinearSVC,
    }


def _vectorizer(config: EvidenceConventionalConfig, dependencies: dict):
    return dependencies["TfidfVectorizer"](
        lowercase=config.lowercase,
        strip_accents=config.strip_accents,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        ngram_range=config.word_ngram_range,
        sublinear_tf=config.sublinear_tf,
        token_pattern=r"(?u)\b\w\w+\b",
    )


def _class_weight(config: EvidenceConventionalConfig) -> str | None:
    return "balanced" if config.class_weighting == "balanced" else None


def _fit_sentence_scorer(
    model_type: Literal["evidence_svm", "evidence_random_forest"],
    rows: list[dict],
    config: EvidenceSVMConfig | EvidenceRandomForestConfig,
):
    dependencies = _require_sklearn_and_skops()
    groups = canonical_sentence_groups(rows)
    records = [record for group in groups for record in group]
    if not records:
        raise ValueError("The training split contains no reviewed Evidence sentences")
    texts = [
        _prepared_text(
            record.target_text,
            record.text,
            target_conditioning=config.target_conditioning,
        )
        for record in records
    ]
    labels = [int(record.label in {"B", "I"}) for record in records]
    label_counts = Counter(labels)
    if len(label_counts) == 1:
        classifier = dependencies["DummyClassifier"](
            strategy="constant",
            constant=next(iter(label_counts)),
        )
    elif model_type == "evidence_svm":
        assert isinstance(config, EvidenceSVMConfig)
        classifier = dependencies["LinearSVC"](
            C=config.c,
            loss=config.loss,
            tol=config.tolerance,
            max_iter=config.max_iterations,
            class_weight=_class_weight(config),
            random_state=config.seed,
        )
    else:
        assert isinstance(config, EvidenceRandomForestConfig)
        classifier = dependencies["RandomForestClassifier"](
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features_per_split,
            n_jobs=config.n_jobs,
            class_weight=_class_weight(config),
            random_state=config.seed,
        )
    pipeline = dependencies["Pipeline"](
        [("tfidf", _vectorizer(config, dependencies)), ("classifier", classifier)]
    )
    pipeline.fit(texts, labels)
    return pipeline, records, dependencies


def _feature_diagnostics(pipeline, *, limit: int = 30) -> list[dict[str, float | str]]:
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
    elif hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    else:
        return []
    ranked = sorted(
        range(len(feature_names)),
        key=lambda index: abs(float(values[index])),
        reverse=True,
    )[:limit]
    return [
        {"feature": str(feature_names[index]), "importance": float(values[index])}
        for index in ranked
    ]


def _write_metadata(
    destination: Path,
    *,
    model_type: str,
    config: EvidenceSVMConfig | EvidenceRandomForestConfig | EvidenceCRFConfig,
    runtime: dict,
) -> None:
    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": model_type,
        "task_contract_key": "evidence_blocks",
        "task_contract_version": "1",
        "config": config.model_dump(mode="json"),
        "runtime": runtime,
    }
    (destination / "al-medlit-conventional.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def train_sentence_scorer(
    model_type: Literal["evidence_svm", "evidence_random_forest"],
    rows: list[dict],
    config: EvidenceSVMConfig | EvidenceRandomForestConfig,
    destination: str | Path,
) -> dict:
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    pipeline, records, dependencies = _fit_sentence_scorer(model_type, rows, config)
    model_path = target / "model.skops"
    dependencies["skops_io"].dump(pipeline, model_path)
    feature_count = len(pipeline.named_steps["tfidf"].get_feature_names_out())
    diagnostics = _feature_diagnostics(pipeline)
    runtime = {
        "python_package": "scikit-learn",
        "scikit_learn_version": dependencies["sklearn"].__version__,
        "feature_count": feature_count,
        "training_sentence_count": len(records),
    }
    _write_metadata(target, model_type=model_type, config=config, runtime=runtime)
    return {
        "synthetic_mode": False,
        "model_family": "conventional_ml",
        "package_format": "skops",
        "training_sentence_count": len(records),
        "feature_count": feature_count,
        "diagnostics": {"feature_importance": diagnostics},
    }


def _crf_tokens(text: str, config: EvidenceCRFConfig) -> tuple[str, ...]:
    tokens = tuple(re.findall(config.token_pattern, text))
    if config.lowercase:
        return tuple(token.lower() for token in tokens)
    return tokens


def count_crf_tokens(text: str, config: EvidenceCRFConfig) -> int:
    return max(1, len(_crf_tokens(text, config)))


def _crf_sentence_features(
    sequence: list[SentenceRecord],
    index: int,
    config: EvidenceCRFConfig,
) -> dict[str, bool | float | str]:
    record = sequence[index]
    tokens = _crf_tokens(record.text, config)
    target_tokens = set(_crf_tokens(record.target_text, config))
    token_set = set(tokens)
    maximum_ordinal = max(item.ordinal for item in sequence) or 1
    features: dict[str, bool | float | str] = {
        "bias": 1.0,
        "sentence.first": tokens[0] if tokens else "",
        "sentence.last": tokens[-1] if tokens else "",
        "sentence.token_count": float(len(tokens)),
        "sentence.character_count": float(len(record.text)),
        "sentence.relative_position": record.ordinal / maximum_ordinal,
        "sentence.target_overlap": (
            len(token_set & target_tokens) / len(target_tokens) if target_tokens else 0.0
        ),
        "BOS": index == 0,
        "EOS": index == len(sequence) - 1,
    }
    for token in sorted(token_set)[:64]:
        features[f"sentence.token={token}"] = True
    if config.target_conditioning:
        for token in sorted(target_tokens)[:32]:
            features[f"target.token={token}"] = True
    if index:
        previous_tokens = _crf_tokens(sequence[index - 1].text, config)
        features["previous.last"] = previous_tokens[-1] if previous_tokens else ""
    if index + 1 < len(sequence):
        next_tokens = _crf_tokens(sequence[index + 1].text, config)
        features["next.first"] = next_tokens[0] if next_tokens else ""
    return features


def _contiguous_sequences(groups: list[list[SentenceRecord]]) -> list[list[SentenceRecord]]:
    sequences: list[list[SentenceRecord]] = []
    for group in groups:
        current: list[SentenceRecord] = []
        previous_ordinal: int | None = None
        for record in group:
            if previous_ordinal is not None and record.ordinal != previous_ordinal + 1:
                sequences.append(current)
                current = []
            current.append(record)
            previous_ordinal = record.ordinal
        if current:
            sequences.append(current)
    return sequences


def _require_crfsuite():
    try:
        import sklearn_crfsuite
    except ImportError as exc:  # pragma: no cover - depends on worker image
        raise MissingConventionalDependencyError(
            "CRF training requires the optional sklearn-crfsuite dependency"
        ) from exc
    return sklearn_crfsuite


def train_crf(rows: list[dict], config: EvidenceCRFConfig, destination: str | Path) -> dict:
    sklearn_crfsuite = _require_crfsuite()
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    sequences = _contiguous_sequences(canonical_sentence_groups(rows))
    if not sequences:
        raise ValueError("The training split contains no reviewed Evidence sequences")
    features = [
        [_crf_sentence_features(sequence, index, config) for index in range(len(sequence))]
        for sequence in sequences
    ]
    labels = [[record.label for record in sequence] for sequence in sequences]
    model_path = target / "model.crfsuite"
    estimator = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=config.c1,
        c2=config.c2,
        max_iterations=config.max_iterations,
        all_possible_transitions=config.all_possible_transitions,
        model_filename=str(model_path),
        keep_tempfiles=True,
    )
    estimator.fit(features, labels)
    if not model_path.is_file():  # pragma: no cover - guards dependency API drift
        raise RuntimeError("sklearn-crfsuite did not produce its native model file")
    runtime = {
        "python_package": "sklearn-crfsuite",
        "training_sequence_count": len(sequences),
        "training_sentence_count": sum(map(len, sequences)),
    }
    _write_metadata(target, model_type="evidence_crf", config=config, runtime=runtime)
    return {
        "synthetic_mode": False,
        "model_family": "conventional_ml",
        "package_format": "crfsuite",
        "training_sequence_count": len(sequences),
        "training_sentence_count": sum(map(len, sequences)),
    }


def _load_metadata(root: Path) -> dict:
    try:
        metadata = json.loads(
            (root / "al-medlit-conventional.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Conventional model metadata is missing or invalid") from exc
    if metadata.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("Unsupported conventional model schema")
    if metadata.get("model_type") not in CONVENTIONAL_MODEL_TYPES:
        raise ValueError("Unsupported conventional model type")
    return metadata


def load_conventional_model(checkpoint_directory: str | Path) -> LoadedConventionalModel:
    root = Path(checkpoint_directory)
    metadata = _load_metadata(root)
    model_type = str(metadata["model_type"])
    if model_type == "evidence_crf":
        config = EvidenceCRFConfig.model_validate(metadata["config"])
        sklearn_crfsuite = _require_crfsuite()
        model_path = root / "model.crfsuite"
        if not model_path.is_file():
            raise ValueError("CRFsuite model file is missing")
        estimator = sklearn_crfsuite.CRF(
            model_filename=str(model_path),
            keep_tempfiles=True,
        )
        return LoadedConventionalModel(model_type, config, estimator)

    dependencies = _require_sklearn_and_skops()
    model_path = root / "model.skops"
    if not model_path.is_file():
        raise ValueError("skops model file is missing")
    untrusted_types = dependencies["skops_io"].get_untrusted_types(file=model_path)
    allowed_prefixes = ("numpy.", "scipy.", "sklearn.")
    disallowed = sorted(
        type_name
        for type_name in untrusted_types
        if not str(type_name).startswith(allowed_prefixes)
    )
    if disallowed:
        raise ValueError(
            "The skops package contains types outside the plugin allowlist: "
            + ", ".join(disallowed)
        )
    estimator = dependencies["skops_io"].load(
        model_path,
        trusted=untrusted_types,
    )
    expected_steps = {"tfidf", "classifier"}
    if set(getattr(estimator, "named_steps", {})) != expected_steps:
        raise ValueError("The skops package is not an Evidence sentence-scoring pipeline")
    vectorizer = estimator.named_steps["tfidf"]
    classifier = estimator.named_steps["classifier"]
    if (
        vectorizer.__class__.__module__ != "sklearn.feature_extraction.text"
        or vectorizer.__class__.__name__ != "TfidfVectorizer"
    ):
        raise ValueError("The skops package uses an unapproved feature vectorizer")
    approved_classifier = {
        "evidence_svm": {"LinearSVC", "DummyClassifier"},
        "evidence_random_forest": {"RandomForestClassifier", "DummyClassifier"},
    }[model_type]
    if (
        not classifier.__class__.__module__.startswith("sklearn.")
        or classifier.__class__.__name__ not in approved_classifier
    ):
        raise ValueError("The skops package uses an unapproved classifier")
    config_cls = (
        EvidenceSVMConfig if model_type == "evidence_svm" else EvidenceRandomForestConfig
    )
    config = config_cls.model_validate(metadata["config"])
    return LoadedConventionalModel(model_type, config, estimator)


def _positive_probabilities(estimator, texts: list[str]) -> list[float]:
    classifier = estimator.named_steps["classifier"]
    if hasattr(estimator, "decision_function"):
        values = estimator.decision_function(texts)
        return [1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, float(value))))) for value in values]
    probabilities = estimator.predict_proba(texts)
    classes = [int(value) for value in classifier.classes_]
    if 1 not in classes:
        return [0.0] * len(texts)
    if 0 not in classes:
        return [1.0] * len(texts)
    positive_index = classes.index(1)
    return [float(row[positive_index]) for row in probabilities]


def predict_window(
    model: LoadedConventionalModel,
    *,
    target_text: str,
    sentences: list[str],
) -> list[float] | list[str]:
    if model.model_type == "evidence_crf":
        assert isinstance(model.config, EvidenceCRFConfig)
        records = [
            SentenceRecord("inference", "target", target_text, index, text, "O")
            for index, text in enumerate(sentences)
        ]
        features = [
            _crf_sentence_features(records, index, model.config)
            for index in range(len(records))
        ]
        return list(model.estimator.predict_single(features))

    assert isinstance(model.config, EvidenceConventionalConfig)
    texts = [
        _prepared_text(
            target_text,
            sentence,
            target_conditioning=model.config.target_conditioning,
        )
        for sentence in sentences
    ]
    return _positive_probabilities(model.estimator, texts)


def actual_token_count(model: LoadedConventionalModel, text: str) -> int:
    if model.model_type == "evidence_crf":
        assert isinstance(model.config, EvidenceCRFConfig)
        return count_crf_tokens(text, model.config)
    analyzer = model.estimator.named_steps["tfidf"].build_analyzer()
    return max(1, len(analyzer(text)))
