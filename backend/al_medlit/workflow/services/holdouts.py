"""Resolve protected source identities for training, selection, and annotation."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import tuple_
from sqlalchemy.orm import Session

from al_medlit.core.exceptions import NotFoundError, ValidationError
from al_medlit.project.models import Project
from al_medlit.workflow import models

from .common import _canonical_hash
from .preparation import (
    _identity_tokens,
    _prepared_provenance_items,
    _protected_source_context,
)

_NON_TEST_SPLITS = {"train", "validation", "pool"}
_VERSION_QUERY_BATCH_SIZE = 400


def lock_holdout_policy(db: Session, project_id: int) -> None:
    """Keep split governance stable before acquiring selection or annotation locks."""
    project = (
        db.query(Project.id)
        .filter(Project.id == project_id)
        .with_for_update(read=True)
        .first()
    )
    if project is None:
        raise NotFoundError("Project not found")


def validate_annotation_assistance(
    db: Session,
    items: list[models.DatasetItem],
    task_version_id: int,
    assistance_policy: str,
) -> None:
    """Require blind labeling of protected source identities."""
    if assistance_policy != "blind" and protected_dataset_item_ids(db, items, task_version_id):
        raise ValidationError("Protected evaluation items require blind annotation")


def _related_dataset_items(
    db: Session,
    items: list[models.DatasetItem],
    versions: dict[int, models.DatasetVersion],
) -> tuple[dict[int, models.DatasetItem], dict[int, set[int]], dict[int, dict]]:
    """Load current group peers and trusted ancestors before checking protection."""
    project_id = items[0].project_id
    candidates = {item.id: item for item in items}
    ancestry: dict[int, dict] = {}
    origins: dict[int, set[int]] = {}
    seen_groups: set[tuple[int, str]] = set()
    while True:
        previous_count = len(candidates)
        pending = [item for item in candidates.values() if item.id not in origins]
        missing_versions = sorted({item.dataset_version_id for item in pending} - versions.keys())
        for offset in range(0, len(missing_versions), _VERSION_QUERY_BATCH_SIZE):
            versions.update(
                (version.id, version)
                for version in db.query(models.DatasetVersion).filter(
                    models.DatasetVersion.project_id == project_id,
                    models.DatasetVersion.id.in_(
                        missing_versions[offset : offset + _VERSION_QUERY_BATCH_SIZE]
                    ),
                )
            )
        missing_origin_ids: set[int] = set()
        for item in pending:
            if item.dataset_version_id not in ancestry:
                ancestry[item.dataset_version_id] = _prepared_provenance_items(
                    db, versions[item.dataset_version_id]
                )
            metadata = ancestry[item.dataset_version_id].get(item.stable_key)
            source_origins = metadata.get("origins", []) if isinstance(metadata, dict) else []
            source_origins = source_origins if isinstance(source_origins, list) else []
            origins[item.id] = {
                origin["dataset_item_id"]
                for origin in source_origins
                if isinstance(origin, dict) and isinstance(origin.get("dataset_item_id"), int)
            }
            missing_origin_ids.update(origins[item.id])

        groups = sorted({
            (item.dataset_version_id, item.group_key)
            for item in pending
            if item.group_key is not None
        } - seen_groups)
        seen_groups.update(groups)
        for offset in range(0, len(groups), _VERSION_QUERY_BATCH_SIZE):
            peers = db.query(models.DatasetItem).filter(
                models.DatasetItem.project_id == project_id,
                tuple_(models.DatasetItem.dataset_version_id, models.DatasetItem.group_key).in_(
                    groups[offset : offset + _VERSION_QUERY_BATCH_SIZE]
                ),
            )
            candidates.update((item.id, item) for item in peers)
        missing_ids = sorted(missing_origin_ids - candidates.keys())
        for offset in range(0, len(missing_ids), _VERSION_QUERY_BATCH_SIZE):
            source_items = db.query(models.DatasetItem).filter(
                models.DatasetItem.project_id == project_id,
                models.DatasetItem.id.in_(missing_ids[offset : offset + _VERSION_QUERY_BATCH_SIZE]),
            )
            candidates.update((item.id, item) for item in source_items)
        if len(candidates) == previous_count:
            return candidates, origins, ancestry


def protected_dataset_item_ids(
    db: Session, items: list[models.DatasetItem], task_version_id: int
) -> set[int]:
    if not items:
        return set()
    project_id = items[0].project_id
    versions: dict[int, models.DatasetVersion] = {}
    candidates, origins, ancestry = _related_dataset_items(db, items, versions)
    identity_ids: set[int] = set()
    identity_tokens: set[str] = set()
    identities: dict[int, tuple[set[int], set[str]]] = {}
    for item in candidates.values():
        ids, tokens = _item_identity(
            versions[item.dataset_version_id], item, ancestry[item.dataset_version_id]
        )
        identities[item.id] = ids, tokens
        identity_ids.update(ids)
        identity_tokens.update(tokens)
    protected_ids, protected_tokens = _protected_source_context(
        db, project_id, task_version_id,
        identity_ids=identity_ids, identity_tokens=identity_tokens,
    )
    protected = {
        item.id
        for item in candidates.values()
        if (
            identities[item.id][0] & protected_ids
            or identities[item.id][1] & protected_tokens
        )
    }
    while True:
        protected_groups = {
            (item.dataset_version_id, item.group_key)
            for item in candidates.values()
            if item.id in protected and item.group_key is not None
        }
        related = {
            item.id
            for item in candidates.values()
            if (item.dataset_version_id, item.group_key) in protected_groups
            or origins[item.id] & protected
        }
        if related.issubset(protected):
            return {item.id for item in items if item.id in protected}
        protected.update(related)


def _item_identity(
    version: models.DatasetVersion,
    item: models.DatasetItem,
    provenance_items: dict,
) -> tuple[set[int], set[str]]:
    ids = {item.id}
    tokens = _identity_tokens(version.dataset_id, item)
    source_metadata = version.provenance or {}
    if source_metadata.get("ingestion") in {"direct_upload", "public_registry_snapshot"} and not (
        source_metadata.get("stable_key_field")
    ):
        # Parser-generated row positions change when an upload is reordered.
        # Its content/article identity still carries governance across revisions.
        tokens.discard(f"source:{version.dataset_id}:item:{item.stable_key}")
        if item.group_key is None:
            tokens.discard(f"source:{version.dataset_id}:group:{item.stable_key}")
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
        origin_tokens = origin.get("identity_tokens")
        if isinstance(origin_tokens, list):
            tokens.update(token for token in origin_tokens if isinstance(token, str))
        # Recorded identities intentionally omit unstable positional upload
        # keys. Reconstruct a key only for older origins without token lists.
        if (
            not isinstance(origin_tokens, list)
            and isinstance(origin.get("dataset_id"), int)
            and isinstance(origin.get("stable_key"), str)
        ):
            tokens.add(f"source:{origin['dataset_id']}:item:{origin['stable_key']}")
    return ids, tokens


def source_split_constraints(
    db: Session,
    dataset_version: models.DatasetVersion,
    items: Iterable[models.DatasetItem],
    task_version_id: int | None = None,
    *,
    prepared_only: bool = False,
    input_mapping: dict[str, str] | None = None,
) -> dict[str, set[str]]:
    """Return allowed partitions for source identities with existing governance.

    Existing test examples stay in test; prior train/validation examples may
    be reused for learning but cannot become a new immutable test holdout. A
    taskless source split map must preserve those boundaries across all tasks.
    Uploaded provenance cannot assert ancestry to a prepared example.
    """
    from .holdout_queries import historical_assignments

    items = list(items)
    if not items:
        return {}
    current_provenance = _prepared_provenance_items(db, dataset_version)
    identities = []
    identity_ids: set[int] = set()
    identity_tokens: set[str] = set()
    for item in items:
        ids, tokens = _item_identity(dataset_version, item, current_provenance)
        if input_mapping and all(field in item.payload for field in input_mapping.values()):
            payload = {target: item.payload[field] for target, field in input_mapping.items()}
            tokens.add(f"input:{_canonical_hash(payload)}")
        identities.append((item, ids, tokens))
        identity_ids.update(ids)
        identity_tokens.update(tokens)
    if prepared_only:
        protected_ids: set[int] = set()
        protected_tokens: set[str] = set()
    else:
        protected_ids, protected_tokens = _protected_source_context(
            db, dataset_version.project_id, task_version_id,
            identity_ids=identity_ids, identity_tokens=identity_tokens,
        )
    learning_ids: set[int] = set()
    learning_tokens: set[str] = set()
    mapped_inputs = None
    match_all_inputs = False
    if input_mapping:
        target, source_field = next(iter(input_mapping.items()))
        values = [item.payload[source_field] for item in items if source_field in item.payload]
        if all(isinstance(value, str) for value in values):
            mapped_inputs = (target, sorted(set(values)))
        else:
            # Structured inputs need the canonical Python comparison below.
            match_all_inputs = True
    for record in historical_assignments(
        db, dataset_version.project_id, task_version_id, training_only=True,
        identity_ids=identity_ids, identity_tokens=identity_tokens,
        mapped_inputs=mapped_inputs, match_all_inputs=match_all_inputs,
    ):
        item = record.item
        ids, tokens = _item_identity(
            record.version, item, {item.stable_key: record.provenance}
        )
        if input_mapping:
            fields = {
                target: target if target in item.payload else record.input_field
                for target in input_mapping
            }
            if all(isinstance(field, str) and field in item.payload for field in fields.values()):
                payload = {target: item.payload[field] for target, field in fields.items()}
                tokens.add(f"input:{_canonical_hash(payload)}")
        if record.assignment == "test" and record.protected:
            protected_ids.update(ids)
            protected_tokens.update(tokens)
        elif record.assignment in {"train", "validation"}:
            learning_ids.update(ids)
            learning_tokens.update(tokens)

    # Classify whole current article groups, including unlabeled rows that can
    # connect a previously used item to new siblings or another source origin.
    parents = list(range(len(identities)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    identity_owner: dict[int | str, int] = {}
    for index, (_, ids, tokens) in enumerate(identities):
        for identity in [*ids, *tokens]:
            owner = identity_owner.setdefault(identity, index)
            parents[root(index)] = root(owner)
    groups: dict[int, list] = {}
    for index, identity in enumerate(identities):
        groups.setdefault(root(index), []).append(identity)

    constraints: dict[str, set[str]] = {}
    for group in groups.values():
        protected = any(
            ids & protected_ids or tokens & protected_tokens for _, ids, tokens in group
        )
        learning = any(ids & learning_ids or tokens & learning_tokens for _, ids, tokens in group)
        if protected and learning:
            raise ValidationError(
                "Existing protected test and prepared training assignments conflict for "
                f"dataset item {group[0][0].stable_key!r}"
            )
        for item, _, _ in group:
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
