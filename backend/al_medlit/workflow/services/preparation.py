"""Prepare immutable training snapshots without changing their source datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import ConflictError, ValidationError
from al_medlit.project.models import Project
from al_medlit.workflow import models, schemas
from al_medlit.workflow import preparation_schemas as api

from .common import (
    _canonical_hash,
    _scoped,
    _task_output_hash,
    _validate_task_input,
    _validate_task_output,
)

PREPARATION_KIND = "training_preparation_v1"
MAX_PREPARATION_ITEMS = 100_000
_UNLABELED = object()


@dataclass
class _Example:
    stable_key: str
    payload: dict
    label: Any
    origins: list[dict] = field(default_factory=list)
    identity_tokens: set[str] = field(default_factory=set)
    fixed_splits: set[str] = field(default_factory=set)
    allowed_splits: set[str] = field(
        default_factory=lambda: {"train", "validation", "test", "pool"}
    )
    group_key: str = ""
    model_assisted: bool = False


@dataclass
class _Preparation:
    preview: api.TrainingPreparationPreview
    examples: list[_Example]
    assignments: dict[str, str]
    source_labels: list[dict]
    task: models.TaskVersion


def _issue(issues: list, code: str, message: str, source_index=None, item_key=None) -> None:
    # A bounded response remains useful for large malformed uploads.
    if len(issues) < 100:
        issues.append(
            api.PreparationIssue(
                code=code,
                message=message,
                source_index=source_index,
                item_key=item_key,
            )
        )


def _identity_tokens(dataset_id: int, item: models.DatasetItem) -> set[str]:
    group = item.group_key if item.group_key is not None else item.stable_key
    tokens = {
        f"source:{dataset_id}:group:{group}",
        f"source:{dataset_id}:item:{item.stable_key}",
        # Item keys can change when positional uploads are reordered or copied
        # into another source. The server-computed payload hash stays stable.
        f"content:{item.content_hash}",
    }
    payload = item.payload
    document_id = payload.get("document_id")
    if type(document_id) is int and document_id > 0:
        # Corpus snapshots retain document identity even if the source dataset
        # or the document's text/metadata changes.
        tokens.add(f"project:{item.project_id}:document:{document_id}")
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    pmid = payload.get("pmid") or metadata.get("pmid")
    if pmid is None and payload.get("source") in {"pmc", "pubmed", "pubmed_abstract"}:
        pmid = payload.get("external_id")
    if isinstance(pmid, (str, int)) and str(pmid).strip().isdigit():
        tokens.add(f"pmid:{str(pmid).strip()}")
    return tokens


def _prepared_provenance_items(db: Session, version: models.DatasetVersion) -> dict:
    """Only server-created preparations can assert ancestry to other source items."""
    provenance = version.provenance or {}
    if provenance.get("ingestion") != PREPARATION_KIND:
        return {}
    owner = db.query(models.TrainingDatasetVersion.id).filter(
        models.TrainingDatasetVersion.project_id == version.project_id,
        models.TrainingDatasetVersion.dataset_version_id == version.id,
        models.TrainingDatasetVersion.preparation_manifest["kind"].as_string() == PREPARATION_KIND,
    ).first()
    if owner is None:
        return {}
    items = provenance.get("items", {})
    return items if isinstance(items, dict) else {}


def _protected_source_context(
    db: Session, project_id: int, task_version_id: int | None = None,
    *, identity_ids: set[int] | None = None, identity_tokens: set[str] | None = None,
) -> tuple[set[int], set[str]]:
    """Follow immutable prepared split membership back to original inputs."""
    from .holdout_queries import historical_assignments
    from .holdouts import _item_identity

    protected_ids: set[int] = set()
    protected_tokens: set[str] = set()
    for record in historical_assignments(
        db, project_id, task_version_id, protected_only=True,
        identity_ids=identity_ids, identity_tokens=identity_tokens,
    ):
        item = record.item
        ids, tokens = _item_identity(
            record.version, item, {item.stable_key: record.provenance}
        )
        protected_ids.update(ids)
        protected_tokens.update(tokens)
    return protected_ids, protected_tokens


def _unique_dataset_name(db: Session, project_id: int, base: str) -> str:
    names = {
        value
        for (value,) in db.query(models.Dataset.name).filter(
            models.Dataset.project_id == project_id
        )
    }
    candidate = base
    number = 2
    while candidate in names:
        suffix = f" ({number})"
        candidate = base[: 255 - len(suffix)] + suffix
        number += 1
    return candidate


def _assisted_label_set_keys(
    db: Session, label_set: models.LabelSetVersion
) -> tuple[set[str], set[int]]:
    """Retain model assistance through published labels and their version ancestry."""
    assisted_keys: set[str] = set()
    assisted_rounds: set[int] = set()
    visited: set[int] = set()
    current = label_set
    while current.id not in visited:
        visited.add(current.id)
        if current.source_annotation_round_id is not None:
            annotation_round = _scoped(
                db, models.AnnotationRound, current.source_annotation_round_id,
                label_set.project_id, "Source annotation round",
            )
            if annotation_round.assistance_policy != "blind":
                assisted_keys.update(current.labels)
                assisted_rounds.add(annotation_round.id)
        elif current.source_decision_ids:
            # Historical composed sets can retain decisions without a single
            # source round. Resolve their per-item assistance rather than
            # treating the composed source kind as evidence of blind labeling.
            decision_ids = current.source_decision_ids
            for offset in range(0, len(decision_ids), 400):
                assisted = (
                    db.query(models.DatasetItem.stable_key, models.AnnotationRound.id)
                    .join(
                        models.RoundItem, models.RoundItem.dataset_item_id == models.DatasetItem.id,
                    )
                    .join(
                        models.RoundAnnotationDecision,
                        models.RoundAnnotationDecision.round_item_id == models.RoundItem.id,
                    )
                    .join(
                        models.AnnotationRound,
                        models.AnnotationRound.id == models.RoundItem.annotation_round_id,
                    )
                    .filter(
                        models.RoundAnnotationDecision.project_id == label_set.project_id,
                        models.RoundAnnotationDecision.id.in_(decision_ids[offset : offset + 400]),
                        models.AnnotationRound.assistance_policy != "blind",
                    )
                )
                for key, round_id in assisted:
                    assisted_keys.add(key)
                    assisted_rounds.add(round_id)
        if current.parent_version_id is None:
            break
        current = _scoped(
            db, models.LabelSetVersion, current.parent_version_id,
            label_set.project_id, "Parent label set version",
        )
        if (
            current.dataset_version_id != label_set.dataset_version_id
            or current.task_version_id != label_set.task_version_id
        ):
            raise ValidationError("Parent labels must match the selected dataset and task versions")
    return assisted_keys & label_set.labels.keys(), assisted_rounds


def _resolve_labels(
    db: Session,
    project_id: int,
    task_id: int,
    source: api.TrainingSource,
    items: list[models.DatasetItem],
    name: str,
) -> dict:
    ref = source.model_dump(exclude_none=True)
    if source.label_set_version_id is not None:
        label_set = _scoped(
            db, models.LabelSetVersion, source.label_set_version_id, project_id, "Source label set"
        )
        if (
            label_set.dataset_version_id != source.dataset_version_id
            or label_set.task_version_id != task_id
        ):
            raise ValidationError(
                "Source label set must match the selected dataset and task versions"
            )
        from .labels import _validate_training_label_set

        _validate_training_label_set(label_set)
        assisted_keys, assisted_rounds = _assisted_label_set_keys(db, label_set)
        ref.update(source_kind=label_set.source_kind, label_content_hash=label_set.content_hash)
        if assisted_rounds:
            ref["assisted_annotation_round_ids"] = sorted(assisted_rounds)
        return {
            "labels": label_set.labels, "ref": ref, "existing": label_set,
            "assisted_keys": assisted_keys,
        }
    if source.annotation_round_id is not None:
        from .labels import resolve_round_labels

        submission_ids = source.submission_ids
        if submission_ids is None:
            submission_ids = [
                row_id
                for (row_id,) in db.query(models.RoundSubmission.id)
                .filter(
                    models.RoundSubmission.project_id == project_id,
                    models.RoundSubmission.annotation_round_id == source.annotation_round_id,
                )
                .order_by(models.RoundSubmission.id)
            ]
        if not submission_ids:
            raise ValidationError("This annotation round has no finalized submissions yet")
        resolved = resolve_round_labels(
            db,
            source.annotation_round_id,
            schemas.RoundLabelSetCreate(
                project_id=project_id,
                name=name,
                source_kind="human",
                submission_ids=submission_ids,
                publication_mode="submitted_snapshot",
            ),
            allow_open=True,
        )
        if (
            resolved["dataset_version_id"] != source.dataset_version_id
            or resolved["task_version_id"] != task_id
        ):
            raise ValidationError(
                "Submitted annotations must match the selected dataset and task versions"
            )
        ref.update(
            source_kind="human",
            submission_ids=resolved["source_submission_ids"],
            source_decision_ids=resolved["source_decision_ids"],
            label_content_hash=_canonical_hash(resolved["labels"]),
        )
        annotation_round = _scoped(
            db, models.AnnotationRound, source.annotation_round_id,
            project_id, "Source annotation round",
        )
        ref["assistance_policy"] = annotation_round.assistance_policy
        assisted_keys = set()
        if annotation_round.assistance_policy != "blind":
            assisted_keys.update(resolved["labels"])
            ref["assisted_annotation_round_ids"] = [annotation_round.id]
        return {
            "labels": resolved["labels"], "ref": ref, "existing": None,
            "assisted_keys": assisted_keys,
        }
    labels = {}
    for item in items:
        value = item.payload.get(source.label_field)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        labels[item.stable_key] = value
    ref.update(source_kind="imported", label_content_hash=_canonical_hash(labels))
    return {"labels": labels, "ref": ref, "existing": None}


def _parent_version(db: Session, project_id: int, data: api.TrainingPreparationRequest):
    if data.training_dataset_id is None:
        return None
    series = _scoped(
        db, models.TrainingDataset, data.training_dataset_id, project_id, "Training dataset"
    )
    if series.task_version_id != data.task_version_id:
        raise ValidationError("A training dataset series keeps its original task version")
    parent = _scoped(
        db,
        models.TrainingDatasetVersion,
        data.parent_version_id,
        project_id,
        "Parent training version",
    )
    if parent.training_dataset_id != series.id:
        raise ValidationError("Parent version belongs to a different training dataset")
    legacy_alias = (
        parent.version_number == 1
        and (parent.preparation_manifest or {}).get("kind") != PREPARATION_KIND
        and data.name == parent.name
    )
    if series.name != data.name and not legacy_alias:
        raise ValidationError("Use the existing training dataset name when creating a new version")
    latest = (
        db.query(models.TrainingDatasetVersion)
        .filter(models.TrainingDatasetVersion.training_dataset_id == series.id)
        .order_by(models.TrainingDatasetVersion.version_number.desc())
        .first()
    )
    if latest is None or latest.id != parent.id:
        raise ConflictError("This training dataset changed; refresh before creating a new version")
    return parent


def _inherit_parent(db: Session, parent, examples: list[_Example], issues: list) -> None:
    if parent is None:
        return
    split = db.get(models.SplitMap, parent.split_map_id)
    version = db.get(models.DatasetVersion, parent.dataset_version_id)
    previous = list(
        db.query(models.DatasetItem).filter(models.DatasetItem.dataset_version_id == version.id)
    )
    current = {example.stable_key: example for example in examples}
    by_origin: dict[int, _Example] = {}
    by_logical_origin: dict[tuple[int, str], _Example] = {}
    for example in examples:
        for origin in example.origins:
            by_origin[origin["dataset_item_id"]] = example
            by_logical_origin[(origin["dataset_id"], origin["stable_key"])] = example
    parent_labels = None
    for item in previous:
        assignment = split.assignments.get(item.stable_key)
        protected = assignment in set(split.protected_splits)
        example = current.get(item.stable_key) or by_origin.get(item.id)
        if example is None and (version.provenance or {}).get("ingestion") != PREPARATION_KIND:
            example = by_logical_origin.get((version.dataset_id, item.stable_key))
        if example is not None and protected:
            prior_input = item.payload
            if (version.provenance or {}).get("ingestion") != PREPARATION_KIND:
                old_input_field = (parent.preprocessing or {}).get("input_field", "text")
                prior_input = {}
                for target in example.payload:
                    source_field = target if target in item.payload else old_input_field
                    if source_field in item.payload:
                        prior_input[target] = item.payload[source_field]
            if _canonical_hash(prior_input) != _canonical_hash(example.payload):
                example = None
        if example is None:
            if protected:
                _issue(
                    issues,
                    "missing_holdout",
                    "An update must retain every established protected test example",
                    item_key=item.stable_key,
                )
            continue
        if protected and example.label is _UNLABELED:
            # Review rounds omit holdouts. Reuse their frozen labels only for
            # selected examples whose mapped inputs still match the parent.
            if parent_labels is None:
                from .labels import _compose_label_sets

                label_sets = [
                    _scoped(
                        db, models.LabelSetVersion, label_id, parent.project_id, "Parent labels"
                    )
                    for label_id in parent.label_set_version_ids
                ]
                _, parent_labels = _compose_label_sets(label_sets, parent.composition)
            if item.stable_key in parent_labels:
                example.label = parent_labels[item.stable_key]
        if assignment is not None:
            example.fixed_splits.add(assignment)


def _assign_splits(
    examples: list[_Example], data: api.TrainingPreparationRequest, issues: list,
    *, warnings: list | None = None,
) -> tuple[dict[str, str], int]:
    # Union common article/group identities, including identities joined by
    # duplicate examples across two sources. No related rows cross partitions.
    parents = {item.stable_key: item.stable_key for item in examples}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    token_owner: dict[str, str] = {}
    for item in examples:
        for token in sorted(item.identity_tokens):
            owner = token_owner.setdefault(token, item.stable_key)
            parents[root(item.stable_key)] = root(owner)
    groups: dict[str, list[_Example]] = {}
    for item in examples:
        groups.setdefault(root(item.stable_key), []).append(item)
    # Unlabeled rows can connect labeled siblings to prior learning data.
    # Keep those bridges when resolving identities, but assign only groups
    # containing labeled examples to the prepared snapshot.
    groups = {
        key: group for key, group in groups.items()
        if any(item.label is not _UNLABELED for item in group)
    }
    if len(groups) < 3:
        _issue(
            issues,
            "insufficient_groups",
            "At least three independent labeled groups are required across the selected sources",
        )
    assigned_groups: dict[str, str] = {}
    allowed_by_group: dict[str, set[str]] = {}
    for key, group in groups.items():
        tokens = sorted({token for item in group for token in item.identity_tokens})
        group_key = "group:" + _canonical_hash(tokens)
        model_assisted = any(item.model_assisted for item in group)
        for item in group:
            item.group_key = group_key
            item.model_assisted = model_assisted
        allowed = set.intersection(*(item.allowed_splits for item in group))
        allowed_by_group[key] = allowed
        fixed = {split for item in group for split in item.fixed_splits}
        if not allowed or len(fixed) > 1 or not fixed.issubset(allowed):
            _issue(
                issues,
                "split_conflict",
                "A shared example or article has conflicting split requirements; "
                "articles used for training, validation, or model-assisted review "
                "cannot become new test holdouts",
                item_key=group[0].stable_key,
            )
        elif fixed:
            assigned_groups[key] = next(iter(fixed))
    unassigned = sorted(
        set(groups) - set(assigned_groups),
        key=lambda key: (
            hashlib.sha256(f"{data.seed}\0{groups[key][0].group_key}".encode()).digest(),
            key,
        ),
    )
    total = len(groups)
    desired_train = min(max(1, round(total * data.train_percent / 100)), max(1, total - 2))
    desired_validation = min(
        max(1, round(total * data.validation_percent / 100)),
        max(1, total - desired_train - 1),
    )
    desired_test = max(1, total - desired_train - desired_validation)
    counts = {
        split: sum(value == split for value in assigned_groups.values())
        for split in ("train", "validation", "test", "pool")
    }
    frozen_test = data.parent_version_id is not None
    if not frozen_test:
        # Source holdouts count toward a new series' quota; they do not freeze
        # it. Reserve all remaining test groups before learning can use them.
        missing_learning_splits = sum(counts[split] == 0 for split in ("train", "validation"))
        eligible_test = [key for key in unassigned if "test" in allowed_by_group[key]]
        reserve_count = min(
            max(0, desired_test - counts["test"]),
            len(eligible_test),
            max(0, len(unassigned) - missing_learning_splits),
        )
        reserved = eligible_test[-reserve_count:] if reserve_count else []
        assigned_groups.update((key, "test") for key in reserved)
        counts["test"] += len(reserved)
        reserved_keys = set(reserved)
        unassigned = [key for key in unassigned if key not in reserved_keys]
    needed = [
        split for split in ("train", "validation", "test") if split not in assigned_groups.values()
    ]
    # Reserve scarce eligible groups first, especially the only article that
    # has never been used for learning. Equal availability preserves seed order.
    needed.sort(key=lambda split: sum(split in allowed_by_group[key] for key in unassigned))
    for split in needed:
        candidate = next((key for key in unassigned if split in allowed_by_group[key]), None)
        if candidate is not None:
            unassigned.remove(candidate)
            assigned_groups[candidate] = split
            counts[split] += 1
    for key in unassigned:
        allowed = allowed_by_group[key]
        if not allowed:
            continue
        if counts["train"] < desired_train and "train" in allowed:
            split = "train"
        elif counts["validation"] < desired_validation and "validation" in allowed:
            split = "validation"
        elif "train" in allowed and "validation" in allowed and (
            frozen_test or counts["test"] >= desired_test or "test" not in allowed
        ):
            # Updates retain their holdout; new series stop at their test quota.
            ratio = counts["train"] / max(1, counts["train"] + counts["validation"])
            split = (
                "train"
                if ratio < data.train_percent / (data.train_percent + data.validation_percent)
                else "validation"
            )
        elif "test" in allowed:
            split = "test"
        else:
            split = next(split for split in ("train", "validation", "pool") if split in allowed)
        assigned_groups[key] = split
        counts[split] += 1
    assignments = {
        item.stable_key: assigned_groups.get(key, "pool")
        for key, group in groups.items()
        for item in group
        if item.label is not _UNLABELED
    }
    for split in ("train", "validation", "test"):
        if split not in assignments.values():
            _issue(
                issues,
                "empty_split",
                f"The combined labeled examples cannot provide a non-empty {split} split"
                + (
                    "; include a labeled article without prior training, validation, "
                    "or model assistance"
                    if split == "test" else ""
                ),
            )
    requested = {"train": desired_train, "validation": desired_validation, "test": desired_test}
    if warnings is not None and not issues and any(
        counts[split] != target for split, target in requested.items()
    ):
        requested_counts = ", ".join(f"{split} {count}" for split, count in requested.items())
        actual_counts = ", ".join(f"{split} {counts[split]}" for split in requested)
        if counts["pool"]:
            actual_counts += f", pool {counts['pool']}"
        reason = (
            "Existing-series test membership stays frozen as articles are added."
            if frozen_test else
            "Fixed assignments and prior training boundaries prevent the requested percentages."
        )
        _issue(
            warnings,
            "split_percentage_adjusted",
            f"Requested group counts: {requested_counts}; actual group counts: {actual_counts}. "
            + reason,
        )
    return assignments, len(groups)


def _build_preparation(
    db: Session, project_id: int, data: api.TrainingPreparationRequest
) -> _Preparation:
    from .holdouts import _item_identity, source_split_constraints

    task = _scoped(db, models.TaskVersion, data.task_version_id, project_id, "Task version")
    issues: list[api.PreparationIssue] = []
    warnings: list[api.PreparationIssue] = []
    parent = _parent_version(db, project_id, data)
    counts: list[api.PreparationSourceCounts] = []
    resolved: list[dict] = []
    examples: dict[str, _Example] = {}
    unlabeled_holdouts: list[tuple[int, str, str | None]] = []
    duplicates = 0
    total = 0
    for index, source in enumerate(data.sources):
        version = _scoped(
            db,
            models.DatasetVersion,
            source.dataset_version_id,
            project_id,
            "Source dataset version",
        )
        items = list(
            db.query(models.DatasetItem)
            .filter(models.DatasetItem.dataset_version_id == version.id)
            .order_by(models.DatasetItem.stable_key)
        )
        total += len(items)
        if total > MAX_PREPARATION_ITEMS:
            raise ValidationError("A preparation can contain at most 100,000 source examples")
        if not items:
            _issue(
                issues, "empty_source", "The source dataset must contain imported examples", index
            )
        try:
            split_constraints = source_split_constraints(
                db, version, items, task.id, input_mapping=source.input_mapping
            )
        except ValidationError as exc:
            _issue(issues, "split_conflict", exc.message, index)
            split_constraints = {}
        split = None
        if source.split_map_id is not None:
            split = _scoped(
                db, models.SplitMap, source.split_map_id, project_id, "Source split map"
            )
            if split.dataset_version_id != version.id:
                raise ValidationError("A source split map must belong to that dataset version")
            if set(split.protected_splits) & {"train", "validation"}:
                _issue(
                    issues,
                    "protected_training_split",
                    "Protected train or validation partitions cannot be used for training",
                    index,
                )
        try:
            labels = _resolve_labels(db, project_id, task.id, source, items, data.name)
        except (ValidationError, ConflictError) as exc:
            _issue(issues, "invalid_labels", exc.message, index)
            labels = {"labels": {}, "ref": source.model_dump(exclude_none=True), "existing": None}
        labels["ref"].update(
            dataset_id=version.dataset_id,
            dataset_content_hash=version.content_hash,
            source_index=index,
            license_info=version.license_info,
        )
        labels["dataset_version"] = version
        resolved.append(labels)
        inherited_items = _prepared_provenance_items(db, version)
        labeled_count = sum(item.stable_key in labels["labels"] for item in items)
        counts.append(
            api.PreparationSourceCounts(
                dataset_version_id=version.id,
                total_count=len(items),
                labeled_count=labeled_count,
                excluded_unlabeled_count=len(items) - labeled_count,
            )
        )
        for item in items:
            _, tokens = _item_identity(version, item, inherited_items)
            inherited = inherited_items.get(item.stable_key, {})
            tokens.update(inherited.get("identity_tokens", []))
            protected = bool(
                (split is not None and split.assignments.get(item.stable_key) == "test")
                or split_constraints.get(item.stable_key) == {"test"}
            )
            labeled = item.stable_key in labels["labels"]
            label = labels["labels"][item.stable_key] if labeled else _UNLABELED
            missing = sorted(set(source.input_mapping.values()) - set(item.payload))
            if missing:
                if labeled:
                    _issue(
                        issues,
                        "missing_input",
                        f"Missing mapped input fields: {', '.join(missing)}",
                        index,
                        item.stable_key,
                    )
                elif protected:
                    unlabeled_holdouts.append((index, item.stable_key, None))
                continue
            payload = {
                target: item.payload[field] for target, field in source.input_mapping.items()
            }
            if labeled:
                try:
                    _validate_task_input(task, payload, label="Mapped source example")
                    _validate_task_output(task, label, label="Source training label")
                except ValidationError as exc:
                    _issue(issues, "incompatible_example", exc.message, index, item.stable_key)
                    continue
            content = _canonical_hash(payload)
            key = f"example:{content}"
            if not labeled and protected:
                unlabeled_holdouts.append((index, item.stable_key, key))
            tokens.add(f"input:{content}")
            origin = {
                "dataset_id": version.dataset_id,
                "dataset_version_id": version.id,
                "dataset_item_id": item.id,
                "stable_key": item.stable_key,
                "source_index": index,
                "identity_tokens": sorted(tokens),
            }
            origins = [origin, *inherited.get("origins", [])]
            example = examples.get(key)
            if example is None:
                example = _Example(stable_key=key, payload=payload, label=label)
                examples[key] = example
            elif labeled and example.label is _UNLABELED:
                example.label = label
            elif labeled:
                duplicates += 1
                if _task_output_hash(task, example.label) != _task_output_hash(task, label):
                    _issue(
                        issues,
                        "label_conflict",
                        "Identical inputs have conflicting labels across the selected sources",
                        index,
                        item.stable_key,
                    )
            example.origins.extend(origins)
            example.identity_tokens.update(tokens)
            if (
                item.stable_key in labels.get("assisted_keys", set())
                or inherited.get("model_assisted") is True
            ):
                example.model_assisted = True
                example.allowed_splits.discard("test")
            if item.stable_key in split_constraints:
                example.allowed_splits.intersection_update(split_constraints[item.stable_key])
            if split is not None and item.stable_key in split.assignments:
                example.fixed_splits.add(split.assignments[item.stable_key])
            if protected:
                example.fixed_splits.add("test")
    rows = sorted(examples.values(), key=lambda example: example.stable_key)
    _inherit_parent(db, parent, rows, issues)
    # Labels can come from another selected source or the immutable parent;
    # keep unlabeled occurrences' provenance and split rules until then.
    for index, item_key, key in unlabeled_holdouts:
        if key is None or examples[key].label is _UNLABELED:
            _issue(
                issues,
                "unlabeled_holdout",
                "Every selected protected test example needs a finalized label",
                index,
                item_key,
            )
    assignments, group_count = _assign_splits(rows, data, issues, warnings=warnings)
    rows = [example for example in rows if example.label is not _UNLABELED]
    split_counts = {
        split: sum(value == split for value in assignments.values())
        for split in ("train", "validation", "test", "pool")
    }
    refs = [source["ref"] for source in resolved]
    fingerprint = _canonical_hash(
        {
            "task_version_id": task.id,
            "task_content_hash": task.content_hash,
            "parent_version_id": data.parent_version_id,
            "sources": refs,
            "seed": data.seed,
            "train_percent": data.train_percent,
            "validation_percent": data.validation_percent,
            "examples": [
                {
                    "key": row.stable_key,
                    "payload": row.payload,
                    "label": row.label,
                    "group_key": row.group_key,
                    "origins": row.origins,
                }
                for row in rows
            ],
            "assignments": assignments,
        }
    )
    preview = api.TrainingPreparationPreview(
        ready=not issues,
        issues=issues,
        warnings=warnings,
        source_counts=counts,
        input_count=total,
        labeled_count=sum(count.labeled_count for count in counts),
        excluded_unlabeled_count=sum(count.excluded_unlabeled_count for count in counts),
        duplicate_count=duplicates,
        item_count=len(rows),
        group_count=group_count,
        split_counts=split_counts,
        manifest_hash=fingerprint,
        resolved_sources=[
            {key: value for key, value in ref.items() if key in api.TrainingSource.model_fields}
            for ref in refs
        ],
    )
    return _Preparation(preview, rows, assignments, resolved, task)


def preview_training_preparation(
    db: Session, project_id: int, data: api.TrainingPreparationRequest
):
    return _build_preparation(db, project_id, data).preview


def create_legacy_training_series(
    db: Session, *, project_id: int, name: str, task_version_id: int, actor: User
) -> models.TrainingDataset:
    """Compatibility creation; old snapshots keep their labels and exact IDs."""
    db.query(Project).filter(Project.id == project_id).with_for_update().one()
    names = {
        value
        for (value,) in db.query(models.TrainingDataset.name).filter(
            models.TrainingDataset.project_id == project_id
        )
    }
    candidate = name
    number = 2
    while candidate in names:
        suffix = f" ({number})"
        candidate = name[: 255 - len(suffix)] + suffix
        number += 1
    series = models.TrainingDataset(
        project_id=project_id,
        name=candidate,
        task_version_id=task_version_id,
        created_by_user_id=actor.id,
    )
    db.add(series)
    db.flush()
    return series


def _persist_source_labels(
    db: Session, project_id: int, task_id: int, sources: list[dict], actor: User
) -> list[dict]:
    from .labels import create_label_set_version

    refs = []
    for source in sources:
        ref = dict(source["ref"])
        existing = source["existing"]
        if existing is None:
            label_name = f"Prepared source {ref['label_content_hash'][:16]}"
            existing = (
                db.query(models.LabelSetVersion)
                .filter(
                    models.LabelSetVersion.dataset_version_id == ref["dataset_version_id"],
                    models.LabelSetVersion.task_version_id == task_id,
                    models.LabelSetVersion.name == label_name,
                )
                .order_by(models.LabelSetVersion.version_number.desc())
                .first()
            )
            # Human provenance must not collapse two equal label outputs from
            # different finalized submissions into one source snapshot.
            source_submissions = ref.get("submission_ids", [])
            source_decisions = ref.get("source_decision_ids", [])
            if existing is not None and (
                existing.source_kind != ref["source_kind"]
                or _canonical_hash(existing.labels) != _canonical_hash(source["labels"])
                or existing.source_annotation_round_id != ref.get("annotation_round_id")
                or existing.source_submission_ids != source_submissions
                or existing.source_decision_ids != source_decisions
            ):
                existing = None
            if existing is None:
                existing = create_label_set_version(
                    db,
                    schemas.LabelSetVersionCreate(
                        project_id=project_id,
                        dataset_version_id=ref["dataset_version_id"],
                        task_version_id=task_id,
                        name=label_name,
                        source_kind=ref["source_kind"],
                        composition_policy="replace",
                        labels=source["labels"],
                    ),
                    actor,
                    source_annotation_round_id=ref.get("annotation_round_id"),
                    source_submission_ids=source_submissions,
                    source_decision_ids=source_decisions,
                    commit=False,
                )
        ref["label_set_version_id"] = existing.id
        ref["label_content_hash"] = existing.content_hash
        refs.append(ref)
    return refs


def _result(db: Session, version: models.TrainingDatasetVersion):
    series = db.get(models.TrainingDataset, version.training_dataset_id)
    return api.TrainingPreparationResult(
        training_dataset=api.TrainingDatasetRead.model_validate(series),
        training_dataset_version=api.PreparedTrainingVersionRead.model_validate(version),
        preview=api.TrainingPreparationPreview.model_validate(
            version.preparation_manifest["preview"]
        ),
    )


def prepare_training_dataset(
    db: Session, project_id: int, data: api.TrainingPreparationRequest, actor: User
):
    if not data.idempotency_key:
        raise ValidationError("An idempotency_key is required to prepare training data")
    request_hash = _canonical_hash(data.model_dump(exclude={"idempotency_key"}))
    # Project lock also serializes first creation/name conflicts across actors.
    db.query(Project).filter(Project.id == project_id).with_for_update().one()
    existing = (
        db.query(models.TrainingDatasetVersion)
        .filter(
            models.TrainingDatasetVersion.project_id == project_id,
            models.TrainingDatasetVersion.idempotency_key == data.idempotency_key,
        )
        .first()
    )
    if existing is not None:
        if existing.request_hash != request_hash:
            raise ConflictError(
                "This preparation idempotency key was already used with different inputs"
            )
        return _result(db, existing)
    try:
        prepared = _build_preparation(db, project_id, data)
        if (
            data.preview_manifest_hash
            and prepared.preview.manifest_hash != data.preview_manifest_hash
        ):
            raise ConflictError(
                "Source annotations or split governance changed; preview the training data again"
            )
        if not prepared.preview.ready:
            raise ValidationError("; ".join(issue.message for issue in prepared.preview.issues[:5]))
        if data.training_dataset_id is not None:
            series = (
                db.query(models.TrainingDataset)
                .filter(models.TrainingDataset.id == data.training_dataset_id)
                .with_for_update()
                .one()
            )
            parent = _parent_version(db, project_id, data)
            number = parent.version_number + 1
        else:
            if (
                db.query(models.TrainingDataset.id)
                .filter(
                    models.TrainingDataset.project_id == project_id,
                    models.TrainingDataset.name == data.name,
                )
                .first()
                is not None
            ):
                raise ConflictError(
                    "A training dataset with this name already exists; "
                    "create a new version or choose another name"
                )
            series = models.TrainingDataset(
                project_id=project_id,
                name=data.name,
                task_version_id=data.task_version_id,
                created_by_user_id=actor.id,
            )
            db.add(series)
            db.flush()
            number = 1
        refs = _persist_source_labels(
            db, project_id, data.task_version_id, prepared.source_labels, actor
        )
        internal_name = _unique_dataset_name(
            db, project_id, f"Training snapshot {series.id} v{number}"
        )
        # Generated snapshots stay ordinary datasets so all existing trainer,
        # evaluation, and immutable run lineage contracts continue to work.
        dataset = models.Dataset(
            project_id=project_id,
            name=internal_name,
            description="Immutable inputs prepared from pinned source datasets",
            source_type="generated",
            purposes=["training_source"],
            created_by_user_id=actor.id,
        )
        db.add(dataset)
        db.flush()
        item_provenance = {
            row.stable_key: {
                "origins": row.origins,
                "identity_tokens": sorted(row.identity_tokens),
                "model_assisted": row.model_assisted,
            }
            for row in prepared.examples
        }
        provenance = {
            "ingestion": PREPARATION_KIND,
            "training_dataset_id": series.id,
            "version_number": number,
            "sources": refs,
            "items": item_provenance,
            "manifest_hash": prepared.preview.manifest_hash,
        }
        dataset_version = models.DatasetVersion(
            project_id=project_id,
            dataset_id=dataset.id,
            version_number=1,
            source_uri=f"training://{series.id}/versions/{number}",
            source_revision=prepared.preview.manifest_hash,
            source_format="other",
            data_schema=prepared.task.input_schema,
            provenance=provenance,
            license_info={
                "status": "inherited_from_sources",
                "sources": [
                    {
                        "dataset_version_id": ref["dataset_version_id"],
                        "license_info": ref["license_info"],
                    }
                    for ref in refs
                ],
            },
            content_hash=_canonical_hash(
                {
                    "provenance": provenance,
                    "items": [
                        {"key": row.stable_key, "payload": row.payload} for row in prepared.examples
                    ],
                }
            ),
            item_count=len(prepared.examples),
            created_by_user_id=actor.id,
        )
        db.add(dataset_version)
        db.flush()
        for row in prepared.examples:
            db.add(
                models.DatasetItem(
                    project_id=project_id,
                    dataset_version_id=dataset_version.id,
                    stable_key=row.stable_key,
                    group_key=row.group_key,
                    payload=row.payload,
                    content_hash=_canonical_hash(row.payload),
                )
            )
        labels = {row.stable_key: row.label for row in prepared.examples}
        label_set = models.LabelSetVersion(
            project_id=project_id,
            dataset_version_id=dataset_version.id,
            task_version_id=data.task_version_id,
            name=f"{series.name[:220]} composed labels",
            version_number=1,
            source_kind="composed",
            composition_policy="replace",
            labels=labels,
            label_count=len(labels),
            content_hash=_canonical_hash({"labels": labels, "sources": refs}),
            source_submission_ids=[],
            source_decision_ids=[],
            created_by_user_id=actor.id,
        )
        db.add(label_set)
        split = models.SplitMap(
            project_id=project_id,
            dataset_version_id=dataset_version.id,
            name=f"{series.name[:230]} v{number} split",
            strategy="source_governed_group_hash_v1",
            seed=data.seed,
            group_key_field="group_key",
            assignments=prepared.assignments,
            protected_splits=["test"],
            content_hash=_canonical_hash(prepared.assignments),
            created_by_user_id=actor.id,
        )
        db.add(split)
        db.flush()
        input_fields = list(prepared.examples[0].payload)
        input_field = "text" if "text" in input_fields else input_fields[0]
        output_properties = prepared.task.output_schema.get("properties", {})
        target_field = next(iter(output_properties)) if len(output_properties) == 1 else "label"
        manifest = {
            "kind": PREPARATION_KIND,
            "sources": refs,
            "preview": prepared.preview.model_dump(),
            "parent_version_id": data.parent_version_id,
            "request": data.model_dump(exclude={"idempotency_key"}),
        }
        version = models.TrainingDatasetVersion(
            project_id=project_id,
            name=series.name,
            training_dataset_id=series.id,
            version_number=number,
            parent_version_id=data.parent_version_id,
            dataset_version_id=dataset_version.id,
            task_version_id=data.task_version_id,
            label_set_version_ids=[label_set.id],
            split_map_id=split.id,
            composition=[{"label_set_version_id": label_set.id, "policy": "replace"}],
            preprocessing={"input_field": input_field, "target_field": target_field},
            preparation_manifest=manifest,
            idempotency_key=data.idempotency_key,
            request_hash=request_hash,
            content_hash=_canonical_hash(
                {
                    "series_id": series.id,
                    "number": number,
                    "manifest_hash": prepared.preview.manifest_hash,
                }
            ),
            created_by_user_id=actor.id,
        )
        db.add(version)
        db.commit()
        db.refresh(version)
        return _result(db, version)
    except IntegrityError as exc:
        db.rollback()
        winner = (
            db.query(models.TrainingDatasetVersion)
            .filter(
                models.TrainingDatasetVersion.project_id == project_id,
                models.TrainingDatasetVersion.idempotency_key == data.idempotency_key,
            )
            .first()
        )
        if winner is not None and winner.request_hash == request_hash:
            return _result(db, winner)
        raise ConflictError(
            "Training preparation changed concurrently; retry with the same request key"
        ) from exc
    except Exception:
        db.rollback()
        raise


def list_training_datasets(db: Session, project_id: int):
    output = []
    for series in (
        db.query(models.TrainingDataset)
        .filter(models.TrainingDataset.project_id == project_id)
        .order_by(models.TrainingDataset.name, models.TrainingDataset.id)
    ):
        versions = list(
            db.query(models.TrainingDatasetVersion)
            .filter(models.TrainingDatasetVersion.training_dataset_id == series.id)
            .order_by(models.TrainingDatasetVersion.version_number.desc())
        )
        output.append(
            api.TrainingDatasetSeriesRead(
                **api.TrainingDatasetRead.model_validate(series).model_dump(),
                versions=[
                    api.PreparedTrainingVersionRead.model_validate(version) for version in versions
                ],
                latest_version_id=versions[0].id if versions else None,
            )
        )
    return output
