"""Keep source split maps compatible with established prepared boundaries."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy.orm import Session

from al_medlit.core.exceptions import ValidationError
from al_medlit.workflow import models

from .preparation import PREPARATION_KIND, _identity_tokens, _protected_source_context

_NON_TEST_SPLITS = {"train", "validation", "pool"}
_VERSION_QUERY_BATCH_SIZE = 400


def _provenance_items(version: models.DatasetVersion) -> dict:
    """The caller must first establish that this snapshot has a prepared owner."""
    provenance = version.provenance or {}
    items = provenance.get("items", {})
    return items if isinstance(items, dict) else {}


def _item_identity(
    version: models.DatasetVersion,
    item: models.DatasetItem,
    provenance_items: dict,
) -> tuple[set[int], set[str]]:
    ids = {item.id}
    tokens = _identity_tokens(version.dataset_id, item)
    provenance = provenance_items.get(item.stable_key, {})
    if not isinstance(provenance, dict):
        return ids, tokens
    identity_tokens = provenance.get("identity_tokens", [])
    if isinstance(identity_tokens, list):
        tokens.update(token for token in identity_tokens if isinstance(token, str))
    origins = provenance.get("origins", [])
    if not isinstance(origins, list):
        return ids, tokens
    for origin in origins:
        if not isinstance(origin, dict):
            continue
        if isinstance(origin.get("dataset_item_id"), int):
            ids.add(origin["dataset_item_id"])
        origin_tokens = origin.get("identity_tokens", [])
        if isinstance(origin_tokens, list):
            tokens.update(token for token in origin_tokens if isinstance(token, str))
        if isinstance(origin.get("dataset_id"), int) and isinstance(origin.get("stable_key"), str):
            tokens.add(f"source:{origin['dataset_id']}:item:{origin['stable_key']}")
    return ids, tokens


def source_split_constraints(
    db: Session,
    dataset_version: models.DatasetVersion,
    items: Iterable[models.DatasetItem],
    task_version_id: int | None = None,
    *,
    prepared_only: bool = False,
) -> dict[str, set[str]]:
    """Return allowed partitions for source identities with existing governance.

    Prepared test examples stay in test; prepared train/validation examples may
    be reused for learning but cannot become a new immutable test holdout. A
    taskless source split map must preserve those boundaries across all tasks.
    Uploaded provenance cannot assert ancestry to a prepared example.
    """
    if prepared_only:
        protected_ids: set[int] = set()
        protected_tokens: set[str] = set()
    else:
        protected_ids, protected_tokens = _protected_source_context(
            db, dataset_version.project_id, task_version_id
        )
    learning_ids: set[int] = set()
    learning_tokens: set[str] = set()

    # Resolve trusted snapshots in one query. Trust is independent of the task
    # whose boundaries are being checked, since a derived source can be reused
    # for another task while still representing the same original article.
    prepared_rows = (
        db.query(models.TrainingDatasetVersion, models.DatasetVersion, models.SplitMap)
        .join(
            models.DatasetVersion,
            models.DatasetVersion.id == models.TrainingDatasetVersion.dataset_version_id,
        )
        .join(
            models.SplitMap,
            models.SplitMap.id == models.TrainingDatasetVersion.split_map_id,
        )
        .filter(
            models.TrainingDatasetVersion.project_id == dataset_version.project_id,
            models.DatasetVersion.project_id == dataset_version.project_id,
            models.SplitMap.project_id == dataset_version.project_id,
        )
        .all()
    )
    trusted_versions: dict[int, models.DatasetVersion] = {}
    relevant_splits: dict[int, list[models.SplitMap]] = {}
    for owner, version, split in prepared_rows:
        if (owner.preparation_manifest or {}).get("kind") != PREPARATION_KIND or (
            version.provenance or {}
        ).get("ingestion") != PREPARATION_KIND:
            continue
        trusted_versions[version.id] = version
        if task_version_id is None or owner.task_version_id == task_version_id:
            relevant_splits.setdefault(version.id, []).append(split)

    provenance_by_version = {
        version_id: _provenance_items(version) for version_id, version in trusted_versions.items()
    }
    version_ids = sorted(relevant_splits)
    # Batching avoids one query per item/version and parameter limits when a
    # project retains many immutable prepared snapshots.
    for offset in range(0, len(version_ids), _VERSION_QUERY_BATCH_SIZE):
        prepared_items = db.query(models.DatasetItem).filter(
            models.DatasetItem.project_id == dataset_version.project_id,
            models.DatasetItem.dataset_version_id.in_(
                version_ids[offset : offset + _VERSION_QUERY_BATCH_SIZE]
            ),
        )
        for item in prepared_items:
            ids, tokens = _item_identity(
                trusted_versions[item.dataset_version_id],
                item,
                provenance_by_version[item.dataset_version_id],
            )
            for split in relevant_splits[item.dataset_version_id]:
                assignment = split.assignments.get(item.stable_key)
                if assignment == "test" and "test" in set(split.protected_splits or []):
                    protected_ids.update(ids)
                    protected_tokens.update(tokens)
                elif assignment in {"train", "validation"}:
                    learning_ids.update(ids)
                    learning_tokens.update(tokens)

    current_provenance = provenance_by_version.get(dataset_version.id, {})
    constraints: dict[str, set[str]] = {}
    for item in items:
        ids, tokens = _item_identity(dataset_version, item, current_provenance)
        protected = bool(ids & protected_ids or tokens & protected_tokens)
        learning = bool(ids & learning_ids or tokens & learning_tokens)
        if protected and learning:
            raise ValidationError(
                "Existing protected test and prepared training assignments conflict for "
                f"dataset item {item.stable_key!r}"
            )
        if protected:
            constraints[item.stable_key] = {"test"}
        elif learning:
            constraints[item.stable_key] = set(_NON_TEST_SPLITS)
    return constraints


def validate_source_split_assignments(
    db: Session,
    dataset_version: models.DatasetVersion,
    items: Iterable[models.DatasetItem],
    assignments: dict[str, str],
    task_version_id: int | None = None,
    *,
    prepared_only: bool = False,
    protected_splits: Iterable[str] = ("test",),
) -> None:
    """Reject assignments that leak holdouts or poison prepared series updates."""
    constraints = source_split_constraints(
        db, dataset_version, items, task_version_id, prepared_only=prepared_only
    )
    protected = set(protected_splits)
    for stable_key, allowed in constraints.items():
        assignment = assignments.get(stable_key)
        if assignment not in allowed or ("test" not in allowed and assignment in protected):
            raise ValidationError(
                "Split assignments must preserve protected test and prepared training "
                f"boundaries for dataset item {stable_key!r}"
            )
