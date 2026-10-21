"""Read governed historical items without loading whole snapshots or manifests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sqlalchemy import String, and_, case, cast, func, literal, or_, select, true, type_coerce
from sqlalchemy.orm import Session

from al_medlit.core.types import JSONType
from al_medlit.workflow import models

_PREPARATION_KIND = "training_preparation_v1"
_IDENTITY_WHITESPACE = (
    " \t\n\r\v\f\x1c\x1d\x1e\x1f\x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)


def _json_member(db: Session, value, key, *, text: bool = False):
    if db.get_bind().dialect.name == "postgresql":
        if isinstance(key, str):
            key = literal(key, type_=String())
        return value.op("->>" if text else "->", return_type=String() if text else JSONType())(key)
    # Quoting the key also handles dots, quotes, and other JSON path characters.
    path = literal("$.") + func.json_quote(key)
    result = func.json_extract(value, path)
    return cast(result, String) if text else type_coerce(result, JSONType())


def _json_entries(db: Session, value, *, array: bool = False):
    if db.get_bind().dialect.name == "postgresql":
        function = func.jsonb_array_elements_text if array else func.jsonb_each_text
    else:
        function = func.json_each
    return (
        function(value).table_valued("key", "value", joins_implicitly=True)
        if not array
        else (function(value).table_valued("value", joins_implicitly=True))
    )


def _array_values(db: Session, values):
    entries = _json_entries(db, literal(list(values), type_=JSONType()), array=True)
    return select(entries.c.value)


def _has_value(db: Session, array, values):
    entries = _json_entries(db, array, array=True)
    return select(1).select_from(entries).where(entries.c.value.in_(values)).exists()


def _mapped_input_matches(db: Session, payload, field, values: list[Any]):
    if all(isinstance(value, str) for value in values):
        return _json_member(db, payload, field, text=True).in_(_array_values(db, values))

    candidates = literal(values, type_=JSONType())
    if db.get_bind().dialect.name == "postgresql":
        entries = func.jsonb_array_elements(candidates).table_valued("value")
        return _json_member(db, payload, field).in_(select(entries.c.value))

    entries = func.json_each(candidates).table_valued("value", "type")
    candidate = case(
        (entries.c.type.in_(("array", "object")), entries.c.value),
        else_=func.json_quote(entries.c.value),
    )
    actual = func.json_quote(_json_member(db, payload, field))
    left = func.json_tree(actual).table_valued("fullkey", "type", "atom")
    right = func.json_tree(candidate).table_valued("fullkey", "type", "atom")
    correlations = (models.DatasetItem, models.TrainingDatasetVersion, entries)
    left_nodes = select(left.c.fullkey, left.c.type, left.c.atom).correlate(*correlations)
    right_nodes = select(right.c.fullkey, right.c.type, right.c.atom).correlate(*correlations)
    # Object key order is immaterial; containers and array positions remain significant.
    path = literal("$.") + func.json_quote(field)
    return (
        select(1)
        .select_from(entries)
        .where(
            func.json_type(payload, path).is_not(None),
            ~left_nodes.except_(right_nodes).exists(),
            ~right_nodes.except_(left_nodes).exists(),
        )
        .correlate(models.DatasetItem, models.TrainingDatasetVersion)
        .exists()
    )


@dataclass
class HistoricalAssignment:
    item: models.DatasetItem
    version: SimpleNamespace
    provenance: dict
    assignment: str
    protected: bool
    input_field: str


def historical_assignments(
    db: Session,
    project_id: int,
    task_version_id: int | None,
    *,
    protected_only: bool = False,
    training_only: bool = False,
    identity_ids: set[int] | None = None,
    identity_tokens: set[str] | None = None,
    mapped_inputs: tuple[str, list[Any]] | None = None,
):
    """Project only each matched item's provenance; ownership remains server-verified.

    Identity predicates deliberately admit false positives, which callers check
    with their usual Python identity rules. This keeps article matching portable
    without transferring every historical payload and per-version provenance map.
    """
    split = models.SplitMap
    version = models.DatasetVersion
    item = models.DatasetItem
    owner = models.TrainingDatasetVersion
    owners = select(owner.id).where(owner.split_map_id == split.id).correlate(split)
    task_owners = owners.where(owner.task_version_id == task_version_id)
    trusted = (
        select(owner.id)
        .where(
            owner.project_id == project_id,
            owner.dataset_version_id == version.id,
            owner.preparation_manifest["kind"].as_string() == _PREPARATION_KIND,
        )
        .correlate(version)
        .exists()
    )
    prepared = and_(version.provenance["ingestion"].as_string() == _PREPARATION_KIND, trusted)
    provenance = type_coerce(
        case(
            (prepared, _json_member(db, version.provenance["items"], item.stable_key)),
            else_=literal({}, type_=JSONType()),
        ),
        JSONType(),
    )
    entries = _json_entries(db, split.assignments)
    protected = _has_value(db, split.protected_splits, select(entries.c.value).correlate(entries))
    input_field = owner.preprocessing["input_field"].as_string() if training_only else literal(None)
    query = (
        db.query(
            item,
            version.dataset_id,
            version.provenance["ingestion"].as_string(),
            version.provenance["stable_key_field"].as_string(),
            provenance,
            entries.c.value,
            protected,
            input_field,
        )
        .select_from(split)
        .join(entries, true())
        .join(
            item,
            and_(
                item.dataset_version_id == split.dataset_version_id,
                item.stable_key == entries.c.key,
                item.project_id == project_id,
            ),
        )
        .join(
            version, and_(version.id == item.dataset_version_id, version.project_id == project_id)
        )
        .filter(split.project_id == project_id)
    )
    if training_only:
        query = query.join(
            owner,
            and_(
                owner.split_map_id == split.id,
                owner.dataset_version_id == version.id,
                owner.project_id == project_id,
            ),
        )
        if task_version_id is not None:
            query = query.filter(owner.task_version_id == task_version_id)
        query = query.filter(entries.c.value.in_(("train", "validation", "test")))
    elif task_version_id is not None:
        query = query.filter(or_(~owners.exists(), task_owners.exists()))
    if protected_only:
        query = query.filter(protected)

    if identity_ids is not None and identity_tokens is not None:
        tokens = _array_values(db, sorted(identity_tokens))
        ids = _array_values(db, sorted(str(value) for value in identity_ids))
        dataset = cast(version.dataset_id, String)
        group = func.coalesce(item.group_key, item.stable_key)
        predicates = [
            cast(item.id, String).in_(ids),
            (literal("content:") + item.content_hash).in_(tokens),
            (literal("source:") + dataset + ":item:" + item.stable_key).in_(tokens),
            (literal("source:") + dataset + ":group:" + group).in_(tokens),
            (
                literal("project:")
                + cast(item.project_id, String)
                + ":document:"
                + cast(item.payload["document_id"].as_string(), String)
            ).in_(tokens),
            _has_value(db, _json_member(db, provenance, "identity_tokens"), tokens),
        ]
        for value in (
            item.payload["pmid"].as_string(),
            item.payload["metadata"]["pmid"].as_string(),
            item.payload["external_id"].as_string(),
        ):
            # Strip the same whitespace as Python str.strip(), including tabs.
            predicates.append(
                (literal("pmid:") + func.trim(cast(value, String), _IDENTITY_WHITESPACE)).in_(
                    tokens
                )
            )
        # Older prepared records can keep identities only on their origins.
        origin_array = _json_member(db, provenance, "origins")
        if db.get_bind().dialect.name == "postgresql":
            origins = func.jsonb_array_elements(origin_array).table_valued(
                "value", joins_implicitly=True
            )
        else:
            origins = _json_entries(db, origin_array, array=True)
        origin = type_coerce(origins.c.value, JSONType())
        origin_ids = _json_member(db, origin, "dataset_item_id", text=True)
        origin_key = (
            literal("source:")
            + _json_member(db, origin, "dataset_id", text=True)
            + ":item:"
            + _json_member(db, origin, "stable_key", text=True)
        )
        predicates.append(
            select(1)
            .select_from(origins)
            .where(
                or_(
                    origin_ids.in_(ids),
                    origin_key.in_(tokens),
                    _has_value(db, _json_member(db, origin, "identity_tokens"), tokens),
                )
            )
            .correlate(item, version)
            .exists()
        )
        if mapped_inputs is not None:
            target, values = mapped_inputs
            # Legacy snapshots can retain the old input field alongside labels.
            # Checking both admits extra candidates; the Python check stays exact.
            predicates.extend(
                [
                    _mapped_input_matches(db, item.payload, target, values),
                    _mapped_input_matches(
                        db, item.payload, func.coalesce(input_field, "text"), values,
                    ),
                ]
            )
        query = query.filter(or_(*predicates))

    for values in db.execute(query.statement):
        (
            row,
            dataset_id,
            ingestion,
            stable_key_field,
            metadata,
            assignment,
            is_protected,
            field,
        ) = values
        yield HistoricalAssignment(
            row,
            SimpleNamespace(
                id=row.dataset_version_id,
                project_id=project_id,
                dataset_id=dataset_id,
                provenance={"ingestion": ingestion, "stable_key_field": stable_key_field},
            ),
            metadata if isinstance(metadata, dict) else {},
            assignment,
            bool(is_protected),
            field or "text",
        )
