"""Domain operations for the canonical learning workflow."""

import hashlib
import math
import re
from typing import Any

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    ValidationError,
)
from al_medlit.workflow import access, models, schemas

from .common import (
    _canonical_hash,
    _commit,
    _next_version,
    _scoped,
)
from .holdouts import lock_holdout_policy, protected_dataset_item_ids

MAX_DATASET_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DATASET_UPLOAD_ITEMS = 100_000
MAX_JSONL_LINE_BYTES = 1024 * 1024
_PINNED_HUGGING_FACE_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_AUTOMATIC_SELECTION_STRATEGIES = {
    "all",
    "random",
    "uncertainty",
    "diversity",
    "disagreement",
    "error_based",
    "hybrid_uncertainty_diversity",
}
_FEEDBACK_SELECTION_STRATEGIES = {
    "uncertainty",
    "diversity",
    "disagreement",
    "error_based",
    "hybrid_uncertainty_diversity",
}
_SENSITIVE_TRAINING_CONFIG_FRAGMENTS = {
    "access_token",
    "api_key",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "storage_key",
}




def _selection_limit(parameters: dict, eligible_count: int, strategy: str) -> int:
    allowed = {"limit", "batch_size"}
    if strategy in _FEEDBACK_SELECTION_STRATEGIES:
        allowed.add("candidate_key")
    if strategy in {"diversity", "hybrid_uncertainty_diversity"}:
        allowed.add("allow_stable_hash_fallback")
    if strategy == "hybrid_uncertainty_diversity":
        allowed.update({"uncertainty_weight", "diversity_weight"})
    unknown = set(parameters) - allowed
    if unknown:
        raise ValidationError(
            "Unsupported automatic selection parameters: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    if "limit" in parameters and "batch_size" in parameters:
        raise ValidationError("Specify only one of selection limit or batch_size")
    value = parameters.get("limit", parameters.get("batch_size", eligible_count))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError("Selection limit must be a positive integer")
    candidate_key = parameters.get("candidate_key")
    if candidate_key is not None and (not isinstance(candidate_key, str) or not candidate_key):
        raise ValidationError("Selection candidate_key must be a non-empty string")
    allow_hash_fallback = parameters.get("allow_stable_hash_fallback")
    if allow_hash_fallback is not None and not isinstance(allow_hash_fallback, bool):
        raise ValidationError("Selection allow_stable_hash_fallback must be a boolean")
    return min(value, eligible_count)


def _filter_value_set(eligibility_filter: dict, key: str) -> set[str] | None:
    value = eligibility_filter.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValidationError(f"Selection eligibility filter {key!r} must be a string list")
    return set(value)


def _unavailable_selection_item_ids(
    db: Session,
    run: models.SelectionRun | models.FeedbackRun,
    *,
    include_submitted: bool = True,
) -> set[int]:
    """Find submitted or reserved items for the run's pinned dataset and task."""
    source_dataset_id = db.query(models.DatasetVersion.dataset_id).filter(
        models.DatasetVersion.id == run.dataset_version_id,
    ).scalar()
    related_versions = db.query(models.DatasetVersion.id).filter(
        models.DatasetVersion.dataset_id == source_dataset_id,
    )
    round_scope = (
        models.AnnotationRound.project_id == run.project_id,
        models.AnnotationRound.dataset_version_id.in_(related_versions),
        models.AnnotationRound.task_version_id == run.task_version_id,
    )
    unavailable = {
        item_id
        for (item_id,) in db.query(models.RoundItem.dataset_item_id)
        .join(
            models.AnnotationRound,
            models.RoundItem.annotation_round_id == models.AnnotationRound.id,
        )
        .filter(
            *round_scope,
            models.RoundItem.project_id == run.project_id,
            models.AnnotationRound.status == "open",
        )
        .all()
    }
    if not include_submitted:
        return _current_unavailable_items(db, run, unavailable)
    submitted_decisions: dict[tuple[int, int], set[int]] = {}
    for round_id, actor_id, decision_ids in (
        db.query(
            models.RoundSubmission.annotation_round_id,
            models.RoundSubmission.annotator_user_id,
            models.RoundSubmission.decision_ids,
        )
        .join(
            models.AnnotationRound,
            models.RoundSubmission.annotation_round_id == models.AnnotationRound.id,
        )
        .filter(*round_scope, models.RoundSubmission.project_id == run.project_id)
        .all()
    ):
        submitted_decisions.setdefault((round_id, actor_id), set()).update(decision_ids)
    if not submitted_decisions:
        return _current_unavailable_items(db, run, unavailable)

    # Membership lives in JSON; match it in Python for both SQLite and PostgreSQL.
    decisions = (
        db.query(
            models.RoundAnnotationDecision.id,
            models.RoundAnnotationDecision.annotator_user_id,
            models.RoundItem.annotation_round_id,
            models.RoundItem.dataset_item_id,
        )
        .join(
            models.RoundItem,
            models.RoundAnnotationDecision.round_item_id == models.RoundItem.id,
        )
        .join(
            models.AnnotationRound,
            models.RoundItem.annotation_round_id == models.AnnotationRound.id,
        )
        .filter(
            *round_scope,
            models.RoundAnnotationDecision.project_id == run.project_id,
            models.RoundItem.project_id == run.project_id,
        )
        .all()
    )
    for decision_id, actor_id, round_id, item_id in decisions:
        if decision_id in submitted_decisions.get((round_id, actor_id), ()):
            unavailable.add(item_id)
    return _current_unavailable_items(db, run, unavailable)


def _current_unavailable_items(
    db: Session, run: models.SelectionRun | models.FeedbackRun, unavailable: set[int],
) -> set[int]:
    stable_keys = db.query(models.DatasetItem.stable_key).filter(
        models.DatasetItem.id.in_(unavailable),
    )
    return {item_id for (item_id,) in db.query(models.DatasetItem.id).filter(
        models.DatasetItem.dataset_version_id == run.dataset_version_id,
        models.DatasetItem.stable_key.in_(stable_keys),
    )}


def _eligible_selection_items(
    db: Session,
    run: models.SelectionRun,
) -> list[models.DatasetItem]:
    eligibility_filter = run.eligibility_filter or {}
    allowed_filter_keys = {
        "include_stable_keys",
        "exclude_stable_keys",
        "include_group_keys",
        "exclude_group_keys",
        "splits",
    }
    unknown = set(eligibility_filter) - allowed_filter_keys
    if unknown:
        raise ValidationError(
            "Unsupported selection eligibility filters: "
            + ", ".join(sorted(str(key) for key in unknown))
        )
    include_stable = _filter_value_set(eligibility_filter, "include_stable_keys")
    exclude_stable = _filter_value_set(eligibility_filter, "exclude_stable_keys") or set()
    include_groups = _filter_value_set(eligibility_filter, "include_group_keys")
    exclude_groups = _filter_value_set(eligibility_filter, "exclude_group_keys") or set()
    allowed_splits = _filter_value_set(eligibility_filter, "splits")
    unavailable_item_ids = _unavailable_selection_item_ids(db, run)

    split_assignments: dict[str, str] = {}
    protected_splits: set[str] = set()
    if run.split_map_id is not None:
        split_map = _scoped(
            db,
            models.SplitMap,
            run.split_map_id,
            run.project_id,
            "Split map",
        )
        split_assignments = dict(split_map.assignments or {})
        protected_splits = set(split_map.protected_splits or [])
        if allowed_splits is not None and allowed_splits & protected_splits:
            raise ValidationError(
                "Protected splits cannot be requested by an active-learning selection"
            )

    items = (
        db.query(models.DatasetItem)
        .filter(
            models.DatasetItem.project_id == run.project_id,
            models.DatasetItem.dataset_version_id == run.dataset_version_id,
        )
        .order_by(models.DatasetItem.stable_key, models.DatasetItem.id)
        .all()
    )
    protected_item_ids = protected_dataset_item_ids(db, items, run.task_version_id)
    eligible: list[models.DatasetItem] = []
    for item in items:
        if item.id in unavailable_item_ids or item.id in protected_item_ids:
            continue
        split = split_assignments.get(item.stable_key)
        if split in protected_splits:
            continue
        if allowed_splits is not None and split not in allowed_splits:
            continue
        if include_stable is not None and item.stable_key not in include_stable:
            continue
        if item.stable_key in exclude_stable:
            continue
        if include_groups is not None and item.group_key not in include_groups:
            continue
        if item.group_key is not None and item.group_key in exclude_groups:
            continue
        eligible.append(item)
    return eligible


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _candidate_scalar_signal(
    candidate: models.FeedbackCandidate,
    signal_key: str,
    *,
    allow_candidate_score: bool = False,
) -> tuple[float, str] | None:
    for container_name, values in (
        ("explanation", candidate.explanation),
        ("output", candidate.output),
    ):
        if not isinstance(values, dict) or signal_key not in values:
            continue
        value = _finite_float(values[signal_key])
        if value is None:
            raise ValidationError(
                f"Feedback candidate {candidate.id} has an invalid finite numeric "
                f"{container_name}.{signal_key} signal"
            )
        return value, f"{container_name}.{signal_key}"
    if allow_candidate_score and candidate.score is not None:
        value = _finite_float(candidate.score)
        if value is None or not 0.0 <= value <= 1.0:
            raise ValidationError(
                f"Feedback candidate {candidate.id} requires a finite confidence score "
                "in [0, 1] for uncertainty selection"
            )
        # Model scores are confidence. Convert before choosing the most
        # uncertain candidate per item or normalizing hybrid signals.
        return 1.0 - value, "score"
    return None


def _candidate_embedding(
    candidate: models.FeedbackCandidate,
) -> tuple[tuple[float, ...], str] | None:
    for container_name, values in (
        ("explanation", candidate.explanation),
        ("output", candidate.output),
    ):
        if not isinstance(values, dict) or "embedding" not in values:
            continue
        raw_embedding = values["embedding"]
        if not isinstance(raw_embedding, list) or not raw_embedding:
            raise ValidationError(
                f"Feedback candidate {candidate.id} has an invalid non-empty "
                f"{container_name}.embedding vector"
            )
        embedding = tuple(_finite_float(value) for value in raw_embedding)
        if any(value is None for value in embedding):
            raise ValidationError(
                f"Feedback candidate {candidate.id} has a non-numeric or non-finite "
                f"{container_name}.embedding vector"
            )
        return tuple(float(value) for value in embedding), f"{container_name}.embedding"
    return None


def _feedback_selection_candidates(
    db: Session,
    run: models.SelectionRun,
    eligible: list[models.DatasetItem],
) -> list[models.FeedbackCandidate]:
    if run.feedback_set_version_id is None:
        return []
    candidate_key = (run.parameters or {}).get("candidate_key")
    eligible_ids = {item.id for item in eligible}
    candidates = (
        db.query(models.FeedbackCandidate)
        .filter(
            models.FeedbackCandidate.project_id == run.project_id,
            models.FeedbackCandidate.feedback_set_version_id == run.feedback_set_version_id,
        )
        .order_by(models.FeedbackCandidate.dataset_item_id, models.FeedbackCandidate.id)
        .all()
    )
    return [
        candidate
        for candidate in candidates
        if candidate.dataset_item_id in eligible_ids
        and (candidate_key is None or candidate.candidate_key == candidate_key)
    ]


def _candidate_groups(
    candidates: list[models.FeedbackCandidate],
) -> dict[int, list[models.FeedbackCandidate]]:
    groups: dict[int, list[models.FeedbackCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.dataset_item_id, []).append(candidate)
    return groups


def _best_scalar_signals(
    groups: dict[int, list[models.FeedbackCandidate]],
    signal_key: str,
    *,
    allow_candidate_score: bool = False,
) -> tuple[
    dict[int, tuple[float, models.FeedbackCandidate, str]],
    list[int],
]:
    best_by_item: dict[int, tuple[float, models.FeedbackCandidate, str]] = {}
    missing_item_ids: list[int] = []
    for item_id, item_candidates in groups.items():
        scored: list[tuple[float, models.FeedbackCandidate, str]] = []
        for candidate in item_candidates:
            signal = _candidate_scalar_signal(
                candidate,
                signal_key,
                allow_candidate_score=allow_candidate_score,
            )
            if signal is not None:
                scored.append((signal[0], candidate, signal[1]))
        if not scored:
            missing_item_ids.append(item_id)
            continue
        best_by_item[item_id] = max(
            scored,
            key=lambda value: (value[0], -value[1].id),
        )
    return best_by_item, sorted(missing_item_ids)


def _best_embeddings(
    groups: dict[int, list[models.FeedbackCandidate]],
) -> tuple[
    dict[int, tuple[tuple[float, ...], models.FeedbackCandidate, str]],
    list[int],
]:
    best_by_item: dict[int, tuple[tuple[float, ...], models.FeedbackCandidate, str]] = {}
    missing_item_ids: list[int] = []
    for item_id, item_candidates in groups.items():
        available: list[tuple[tuple[float, ...], models.FeedbackCandidate, str]] = []
        for candidate in item_candidates:
            signal = _candidate_embedding(candidate)
            if signal is not None:
                available.append((signal[0], candidate, signal[1]))
        if not available:
            missing_item_ids.append(item_id)
            continue
        best_by_item[item_id] = min(available, key=lambda value: value[1].id)
    return best_by_item, sorted(missing_item_ids)


def _confidence_fallback_reason(candidate: models.FeedbackCandidate, source: str) -> dict:
    if source != "score":
        return {}
    return {"signal_transform": "1 - confidence", "confidence": candidate.score}


def _signal_reason(
    strategy: str,
    signal_name: str,
    score: float,
    candidate: models.FeedbackCandidate,
    source: str,
) -> dict:
    return {
        "strategy": strategy,
        "selection_basis": "deterministic_rank",
        "probability_basis": "deterministic_rank",
        "signal": signal_name,
        "signal_value": score,
        "signal_source": source,
        "feedback_candidate_id": candidate.id,
        "candidate_key": candidate.candidate_key,
        **_confidence_fallback_reason(candidate, source),
    }


def _rank_scalar_strategy(
    *,
    strategy: str,
    signal_name: str,
    signals: dict[int, tuple[float, models.FeedbackCandidate, str]],
    item_by_id: dict[int, models.DatasetItem],
) -> list[tuple[models.DatasetItem, float, float, dict]]:
    ranked = [
        (
            item_by_id[item_id],
            score,
            1.0,
            _signal_reason(strategy, signal_name, score, candidate, source),
        )
        for item_id, (score, candidate, source) in signals.items()
    ]
    ranked.sort(key=lambda value: (-value[1], value[0].stable_key, value[0].id))
    return ranked


def _embedding_distance(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    distance = math.dist(left, right)
    if not math.isfinite(distance):
        raise ValidationError("Diversity embeddings produce a non-finite Euclidean distance")
    return distance


def _validate_embedding_dimensions(
    embeddings: dict[int, tuple[tuple[float, ...], models.FeedbackCandidate, str]],
) -> int:
    dimensions = {len(value[0]) for value in embeddings.values()}
    if len(dimensions) != 1:
        raise ValidationError("Diversity embeddings must all use the same vector dimension")
    return next(iter(dimensions))


def _rank_embedding_diversity(
    embeddings: dict[int, tuple[tuple[float, ...], models.FeedbackCandidate, str]],
    item_by_id: dict[int, models.DatasetItem],
    limit: int,
) -> list[tuple[models.DatasetItem, float, float, dict]]:
    dimension = _validate_embedding_dimensions(embeddings)
    centroid = tuple(
        sum(value[0][axis] for value in embeddings.values()) / len(embeddings)
        for axis in range(dimension)
    )
    remaining = dict(embeddings)
    scores = (
        {
            item_id: _embedding_distance(embedding, centroid)
            for item_id, (embedding, _, _) in remaining.items()
        }
        if limit > 0 else {}
    )
    ranked: list[tuple[models.DatasetItem, float, float, dict]] = []
    while remaining and len(ranked) < limit:
        item_id = min(
            remaining,
            key=lambda item_id: (
                -scores[item_id], item_by_id[item_id].stable_key, item_id,
            ),
        )
        item = item_by_id[item_id]
        embedding, candidate, source = remaining.pop(item_id)
        score = scores.pop(item_id)
        metric = "minimum_distance_to_selected" if ranked else "distance_to_centroid"
        ranked.append(
            (
                item,
                score,
                1.0,
                {
                    **_signal_reason(
                        "diversity",
                        "embedding",
                        score,
                        candidate,
                        source,
                    ),
                    "algorithm": "greedy_max_min",
                    "distance_metric": "euclidean",
                    "ranking_metric": metric,
                    "embedding_dimension": dimension,
                },
            )
        )
        if len(ranked) >= limit:
            break
        # A candidate's nearest selected vector can only change when a new
        # vector is selected. Cache that minimum instead of rescanning the
        # entire selected prefix on each iteration (O(N * L) distances).
        for remaining_id, (vector, _, _) in remaining.items():
            distance = _embedding_distance(vector, embedding)
            scores[remaining_id] = (
                min(scores[remaining_id], distance) if len(ranked) > 1 else distance
            )
    return ranked


def _stable_hash_score(run: models.SelectionRun, item: models.DatasetItem) -> float:
    digest = hashlib.sha256(
        f"{run.seed}\x00{item.stable_key}\x00{item.content_hash}".encode()
    ).digest()
    return int.from_bytes(digest, "big") / ((1 << (len(digest) * 8)) - 1)


def _rank_stable_hash_diversity(
    run: models.SelectionRun,
    eligible: list[models.DatasetItem],
) -> list[tuple[models.DatasetItem, float, float, dict]]:
    ranked = [
        (
            item,
            _stable_hash_score(run, item),
            1.0,
            {
                "strategy": "diversity",
                "selection_basis": "deterministic_rank",
                "probability_basis": "deterministic_rank",
                "signal": "stable_hash",
                "signal_source": "stable_hash_fallback",
                "fallback_explicitly_enabled": True,
                "hash_algorithm": "sha256",
                "hash_input_fields": ["seed", "stable_key", "content_hash"],
                "seed": run.seed,
            },
        )
        for item in eligible
    ]
    ranked.sort(key=lambda value: (-value[1], value[0].stable_key, value[0].id))
    return ranked


def _hybrid_weights(parameters: dict) -> tuple[float, float, dict]:
    uncertainty = parameters.get("uncertainty_weight", 0.5)
    diversity = parameters.get("diversity_weight", 0.5)
    uncertainty_value = _finite_float(uncertainty)
    diversity_value = _finite_float(diversity)
    if (
        uncertainty_value is None
        or diversity_value is None
        or uncertainty_value < 0
        or diversity_value < 0
        or uncertainty_value + diversity_value <= 0
    ):
        raise ValidationError(
            "Hybrid selection weights must be finite, non-negative numbers with a positive total"
        )
    total = uncertainty_value + diversity_value
    if not math.isfinite(total):
        raise ValidationError("Hybrid selection weight total must be finite")
    return (
        uncertainty_value / total,
        diversity_value / total,
        {
            "uncertainty": uncertainty_value,
            "diversity": diversity_value,
            "normalized_uncertainty": uncertainty_value / total,
            "normalized_diversity": diversity_value / total,
        },
    )


def _min_max_normalize(values: dict[int, float]) -> tuple[dict[int, float], dict]:
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        normalized = {item_id: 1.0 for item_id in values}
        constant_value = 1.0
    else:
        span = maximum - minimum
        if not math.isfinite(span):
            raise ValidationError("Hybrid selection signal range is too large to normalize safely")
        normalized = {item_id: (value - minimum) / span for item_id, value in values.items()}
        constant_value = None
    return normalized, {
        "method": "min_max",
        "minimum": minimum,
        "maximum": maximum,
        "constant_result": constant_value,
    }


def _embedding_centroid_scores(
    embeddings: dict[int, tuple[tuple[float, ...], models.FeedbackCandidate, str]],
) -> tuple[dict[int, float], int]:
    dimension = _validate_embedding_dimensions(embeddings)
    centroid = tuple(
        sum(value[0][axis] for value in embeddings.values()) / len(embeddings)
        for axis in range(dimension)
    )
    return (
        {item_id: _embedding_distance(value[0], centroid) for item_id, value in embeddings.items()},
        dimension,
    )


def _rank_hybrid_strategy(
    run: models.SelectionRun,
    eligible: list[models.DatasetItem],
    candidates: list[models.FeedbackCandidate],
    parameters: dict,
) -> list[tuple[models.DatasetItem, float, float, dict]]:
    if not candidates:
        raise ValidationError(
            "Hybrid uncertainty-diversity selection requires eligible feedback candidates"
        )
    groups = _candidate_groups(candidates)
    uncertainty, missing_uncertainty = _best_scalar_signals(
        groups,
        "uncertainty",
        allow_candidate_score=True,
    )
    if missing_uncertainty:
        raise ValidationError(
            "Hybrid selection requires an uncertainty signal for every candidate-covered "
            "item; missing dataset item IDs: "
            + ", ".join(str(item_id) for item_id in missing_uncertainty[:5])
        )

    diversity, missing_diversity = _best_scalar_signals(
        groups,
        "diversity_score",
    )
    diversity_scores: dict[int, float]
    diversity_details: dict[int, dict]
    if not missing_diversity:
        diversity_scores = {item_id: value[0] for item_id, value in diversity.items()}
        diversity_details = {
            item_id: {
                "signal": "diversity_score",
                "signal_source": source,
                "feedback_candidate_id": candidate.id,
                "candidate_key": candidate.candidate_key,
            }
            for item_id, (_, candidate, source) in diversity.items()
        }
    else:
        embeddings, missing_embeddings = _best_embeddings(groups)
        if not missing_embeddings:
            diversity_scores, dimension = _embedding_centroid_scores(embeddings)
            diversity_details = {
                item_id: {
                    "signal": "embedding_centroid_distance",
                    "signal_source": source,
                    "feedback_candidate_id": candidate.id,
                    "candidate_key": candidate.candidate_key,
                    "embedding_dimension": dimension,
                    "distance_metric": "euclidean",
                }
                for item_id, (_, candidate, source) in embeddings.items()
            }
        elif parameters.get("allow_stable_hash_fallback") is True:
            item_by_id = {item.id: item for item in eligible}
            diversity_scores = {
                item_id: _stable_hash_score(run, item_by_id[item_id]) for item_id in groups
            }
            diversity_details = {
                item_id: {
                    "signal": "stable_hash",
                    "signal_source": "stable_hash_fallback",
                    "fallback_explicitly_enabled": True,
                    "hash_algorithm": "sha256",
                    "hash_input_fields": ["seed", "stable_key", "content_hash"],
                    "seed": run.seed,
                }
                for item_id in groups
            }
        else:
            raise ValidationError(
                "Hybrid selection requires diversity_score or same-dimension embedding "
                "signals for every candidate-covered item; enable "
                "allow_stable_hash_fallback explicitly to use the hash fallback"
            )

    uncertainty_scores = {item_id: value[0] for item_id, value in uncertainty.items()}
    normalized_uncertainty, uncertainty_normalization = _min_max_normalize(uncertainty_scores)
    normalized_diversity, diversity_normalization = _min_max_normalize(diversity_scores)
    uncertainty_weight, diversity_weight, weights = _hybrid_weights(parameters)
    item_by_id = {item.id: item for item in eligible}
    ranked: list[tuple[models.DatasetItem, float, float, dict]] = []
    for item_id, uncertainty_value in uncertainty.items():
        uncertainty_score, candidate, source = uncertainty_value
        combined = (
            uncertainty_weight * normalized_uncertainty[item_id]
            + diversity_weight * normalized_diversity[item_id]
        )
        ranked.append(
            (
                item_by_id[item_id],
                combined,
                1.0,
                {
                    "strategy": "hybrid_uncertainty_diversity",
                    "selection_basis": "deterministic_rank",
                    "probability_basis": "deterministic_rank",
                    "score_formula": "weighted_min_max_normalized_sum",
                    "weights": weights,
                    "uncertainty": {
                        "raw_score": uncertainty_score,
                        "normalized_score": normalized_uncertainty[item_id],
                        "signal_source": source,
                        "feedback_candidate_id": candidate.id,
                        "candidate_key": candidate.candidate_key,
                        "normalization": uncertainty_normalization,
                        **_confidence_fallback_reason(candidate, source),
                    },
                    "diversity": {
                        "raw_score": diversity_scores[item_id],
                        "normalized_score": normalized_diversity[item_id],
                        "normalization": diversity_normalization,
                        **diversity_details[item_id],
                    },
                },
            )
        )
    ranked.sort(key=lambda value: (-value[1], value[0].stable_key, value[0].id))
    return ranked


def _materialized_selection_items(
    db: Session,
    run: models.SelectionRun,
    *,
    eligible: list[models.DatasetItem] | None = None,
) -> list[schemas.SelectionItem]:
    if run.strategy not in _AUTOMATIC_SELECTION_STRATEGIES:
        supported = ", ".join(sorted(_AUTOMATIC_SELECTION_STRATEGIES))
        raise ValidationError(
            f"Automatic selection does not yet support {run.strategy!r}; "
            f"supported strategies: {supported}"
        )
    if eligible is None:
        eligible = _eligible_selection_items(db, run)
    if not eligible:
        raise ValidationError(
            "No eligible dataset items remain after excluding submitted annotations, "
            "open-round items, protected splits, and eligibility filtering"
        )
    parameters = run.parameters or {}
    limit = _selection_limit(parameters, len(eligible), run.strategy)

    ranked: list[tuple[models.DatasetItem, float | None, float | None, dict]]
    if run.strategy == "all":
        ranked = [
            (
                item,
                None,
                1.0,
                {
                    "strategy": "all",
                    "selection_basis": "stable_key_order",
                    "probability_basis": "deterministic_inclusion",
                    "eligible_count": len(eligible),
                    "selection_limit": limit,
                },
            )
            for item in eligible
        ]
    elif run.strategy == "random":
        ranked_random = sorted(
            eligible,
            key=lambda item: (
                hashlib.sha256(
                    f"{run.seed}\x00{item.stable_key}\x00{item.content_hash}".encode()
                ).hexdigest(),
                item.stable_key,
                item.id,
            ),
        )
        ranked = [
            (
                item,
                None,
                limit / len(eligible),
                {
                    "strategy": "random",
                    "selection_basis": "seeded_sha256_order",
                    "probability_basis": "uniform_without_replacement",
                    "eligible_count": len(eligible),
                    "selection_limit": limit,
                    "hash_algorithm": "sha256",
                    "hash_input_fields": ["seed", "stable_key", "content_hash"],
                    "seed": run.seed,
                },
            )
            for item in ranked_random
        ]
    else:
        allow_hash_fallback = parameters.get("allow_stable_hash_fallback") is True
        if run.feedback_set_version_id is None and not (
            run.strategy == "diversity" and allow_hash_fallback
        ):
            if run.strategy == "diversity":
                raise ValidationError(
                    "Diversity selection requires a pinned feedback_set_version_id; "
                    "enable allow_stable_hash_fallback explicitly to use the hash fallback"
                )
            raise ValidationError(
                f"{run.strategy} selection requires a pinned feedback_set_version_id"
            )
        candidates = _feedback_selection_candidates(db, run, eligible)
        item_by_id = {item.id: item for item in eligible}
        groups = _candidate_groups(candidates)
        if run.feedback_set_version_id is not None and set(item_by_id) - set(groups):
            raise ValidationError("Feedback selection requires predictions for every eligible item")
        if run.strategy in {"uncertainty", "disagreement", "error_based"}:
            signal_name = {
                "uncertainty": "uncertainty",
                "disagreement": "disagreement_score",
                "error_based": "error_score",
            }[run.strategy]
            signals, missing_item_ids = _best_scalar_signals(
                groups,
                signal_name,
                allow_candidate_score=run.strategy == "uncertainty",
            )
            if not signals:
                raise ValidationError(
                    f"No eligible feedback candidates provide a finite {signal_name} signal"
                )
            if missing_item_ids:
                raise ValidationError(
                    f"{run.strategy} selection requires {signal_name} for every "
                    "candidate-covered item; missing dataset item IDs: "
                    + ", ".join(str(item_id) for item_id in missing_item_ids[:5])
                )
            ranked = _rank_scalar_strategy(
                strategy=run.strategy,
                signal_name=signal_name,
                signals=signals,
                item_by_id=item_by_id,
            )
        elif run.strategy == "diversity":
            diversity, missing_diversity = _best_scalar_signals(
                groups,
                "diversity_score",
            )
            if groups and not missing_diversity:
                ranked = _rank_scalar_strategy(
                    strategy="diversity",
                    signal_name="diversity_score",
                    signals=diversity,
                    item_by_id=item_by_id,
                )
            else:
                embeddings, missing_embeddings = _best_embeddings(groups)
                if groups and not missing_embeddings:
                    ranked = _rank_embedding_diversity(
                        embeddings,
                        item_by_id,
                        limit,
                    )
                elif allow_hash_fallback:
                    ranked = _rank_stable_hash_diversity(run, eligible)
                else:
                    raise ValidationError(
                        "Diversity selection requires diversity_score or same-dimension "
                        "embedding signals for every candidate-covered item; enable "
                        "allow_stable_hash_fallback explicitly to use the hash fallback"
                    )
        else:
            ranked = _rank_hybrid_strategy(
                run,
                eligible,
                candidates,
                parameters,
            )
        limit = min(limit, len(ranked))

    return [
        schemas.SelectionItem(
            dataset_item_id=item.id,
            rank=rank,
            score=score,
            probability=probability,
            reason=reason,
        )
        for rank, (item, score, probability, reason) in enumerate(
            ranked[:limit],
            start=1,
        )
    ]


def _selection_input_hash(run: models.SelectionRun) -> str:
    return _canonical_hash({
        field: getattr(run, field)
        for field in (
            "project_id",
            "cycle_id",
            "dataset_version_id",
            "task_version_id",
            "feedback_set_version_id",
            "split_map_id",
            "strategy",
            "parameters",
            "eligibility_filter",
            "seed",
            "status",
        )
    })


def materialize_selection_run(
    db: Session,
    *,
    project_id: int,
    selection_run_id: int,
    actor: User,
) -> models.SelectionSetVersion:
    run = _scoped(
        db,
        models.SelectionRun,
        selection_run_id,
        project_id,
        "Selection run",
    )
    # Ranking is read-only and can be expensive. Defer mutation locks until
    # publication, then reject a stale result instead of reranking under locks.
    input_hash = _selection_input_hash(run)
    eligible_ids: set[int] | None = None
    items: list[schemas.SelectionItem] | None = None
    if run.status != "completed":
        eligible = _eligible_selection_items(db, run)
        eligible_ids = {item.id for item in eligible}
        items = _materialized_selection_items(db, run, eligible=eligible)

    access.authorize_project_write(
        db,
        actor,
        project_id,
        min_role="trainer",
        module="learning",
    )
    lock_holdout_policy(db, project_id)
    run = (
        db.query(models.SelectionRun)
        .filter(models.SelectionRun.id == selection_run_id)
        .populate_existing()
        .with_for_update()
        .one()
    )
    if run.status == "completed":
        existing = (
            db.query(models.SelectionSetVersion)
            .filter(models.SelectionSetVersion.selection_run_id == run.id)
            .order_by(models.SelectionSetVersion.version_number.desc())
            .first()
        )
        if existing is None:
            raise ConflictError("Completed selection run has no materialized selection set")
        return existing
    if items is None or _selection_input_hash(run) != input_hash:
        raise ConflictError("Selection run inputs changed during ranking; retry materialization")
    current_eligible_ids = {item.id for item in _eligible_selection_items(db, run)}
    if current_eligible_ids != eligible_ids:
        raise ConflictError("Selection eligibility changed during ranking; retry materialization")
    return create_selection_set(
        db,
        schemas.SelectionSetCreate(
            project_id=project_id,
            selection_run_id=run.id,
            items=items,
        ),
        actor,
    )


def create_selection_set(
    db: Session, data: schemas.SelectionSetCreate, actor: User
) -> models.SelectionSetVersion:
    run = _scoped(db, models.SelectionRun, data.selection_run_id, data.project_id, "Selection run")
    lock_holdout_policy(db, data.project_id)
    run = (
        db.query(models.SelectionRun)
        .filter(models.SelectionRun.id == run.id)
        .populate_existing()
        .with_for_update()
        .one()
    )
    if run.status == "completed":
        raise ConflictError("A completed selection run cannot be rewritten")
    item_ids = [item.dataset_item_id for item in data.items]
    ranks = [item.rank for item in data.items]
    if len(item_ids) != len(set(item_ids)) or len(ranks) != len(set(ranks)):
        raise ValidationError("Selection item IDs and ranks must be unique")
    selected_items = [
        _scoped(db, models.DatasetItem, item_id, data.project_id, "Dataset item")
        for item_id in item_ids
    ]
    if any(item.dataset_version_id != run.dataset_version_id for item in selected_items):
        raise ValidationError("Selected items must belong to the selection dataset version")

    protected_ids = protected_dataset_item_ids(db, selected_items, run.task_version_id)
    leaked = {item.stable_key for item in selected_items if item.id in protected_ids}
    if run.split_map_id is not None:
        split_map = _scoped(db, models.SplitMap, run.split_map_id, data.project_id, "Split map")
        protected_splits = set(split_map.protected_splits)
        protected_keys = {
            key for key, split in split_map.assignments.items() if split in protected_splits
        }
        leaked.update(
            item.stable_key for item in selected_items if item.stable_key in protected_keys
        )
    if leaked:
        raise ValidationError(
            "Protected evaluation items cannot enter active-learning selection: "
            + ", ".join(sorted(leaked)[:5])
        )

    payloads = [item.model_dump() for item in sorted(data.items, key=lambda value: value.rank)]
    selection_set = models.SelectionSetVersion(
        project_id=data.project_id,
        selection_run_id=run.id,
        version_number=_next_version(
            db,
            models.SelectionSetVersion,
            models.SelectionSetVersion.selection_run_id,
            run.id,
        ),
        items=payloads,
        item_count=len(payloads),
        content_hash=_canonical_hash(payloads),
        created_by_user_id=actor.id,
    )
    db.add(selection_set)
    run.status = "completed"
    _commit(db, "Could not finalize the selection set")
    db.refresh(selection_set)
    return selection_set
