"""Input and prediction schemas for neural Evidence plugins."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceTextDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1, max_length=255)
    target_id: int | str
    target_text: str = Field(min_length=1)
    sentences: tuple[str, ...] = Field(min_length=1)
    sentence_ordinals: tuple[int, ...] | None = None
    labels: tuple[Literal["O", "B", "I", "IGNORE"], ...] | None = None

    @model_validator(mode="after")
    def aligned_labels(self):
        if self.labels is not None and len(self.labels) != len(self.sentences):
            raise ValueError("labels must align one-to-one with sentences")
        if self.sentence_ordinals is not None:
            if len(self.sentence_ordinals) != len(self.sentences):
                raise ValueError("sentence_ordinals must align one-to-one with sentences")
            if tuple(sorted(set(self.sentence_ordinals))) != self.sentence_ordinals:
                raise ValueError("sentence_ordinals must be unique and increasing")
        return self


class EvidenceSequencePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    target_id: int | str
    sentence_ordinals: tuple[int, ...]
    labels: tuple[Literal["O", "B", "I"], ...]
    probabilities: tuple[tuple[float, float, float], ...]

    @model_validator(mode="after")
    def aligned_scores(self):
        if not (len(self.labels) == len(self.probabilities) == len(self.sentence_ordinals)):
            raise ValueError("Prediction labels and probabilities must align")
        return self


class NeuralEpochMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: int = Field(ge=1)
    train_loss: float = Field(ge=0)
    validation_loss: float | None = Field(default=None, ge=0)
    learning_rate: float = Field(ge=0)
    validation_macro_block_iou_f1_0_50: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    validation_macro_exact_block_f1: float | None = Field(default=None, ge=0, le=1)


class NeuralCheckpointScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    macro_block_iou_f1_0_50: float = Field(ge=0, le=1)
    macro_exact_block_f1: float = Field(ge=0, le=1)


class NeuralTrainingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    history: tuple[NeuralEpochMetric, ...]
    best_epoch: int = Field(ge=1)
    selection_metric: Literal[
        "macro_block_iou_f1_0_50",
        "validation_loss",
        "train_loss",
    ]
    selection_value: float = Field(ge=0)
    selection_tiebreaker_exact_block_f1: float | None = Field(default=None, ge=0, le=1)
    stopped_early: bool
    device: str


def documents_from_window_rows(rows: Iterable[dict]) -> tuple[EvidenceTextDocument, ...]:
    """Collapse overlapping prepared windows into document/target sequences."""

    groups: dict[tuple[str, str], dict] = {}
    for row_number, row in enumerate(rows, 1):
        document_id = row.get("document_id")
        target = row.get("target") or {}
        target_id = target.get("id")
        target_text = str(target.get("text", "")).strip()
        if document_id is None or target_id is None or not target_text:
            raise ValueError(
                f"Prepared neural row {row_number} requires document and target identity"
            )
        key = (str(document_id), str(target_id))
        group = groups.setdefault(
            key,
            {
                "document_id": str(document_id),
                "target_id": target_id,
                "target_text": target_text,
                "sentences": {},
            },
        )
        if group["target_text"] != target_text:
            raise ValueError("Overlapping neural windows disagree on target text")
        for index, sentence in enumerate(row.get("sentences") or []):
            ordinal = int(sentence.get("ordinal", index))
            text = str(sentence.get("text", ""))
            if not text:
                raise ValueError("Prepared neural sentences cannot be empty")
            label = str(sentence.get("label", "IGNORE"))
            if label not in {"O", "B", "I", "IGNORE"}:
                raise ValueError(f"Unsupported Evidence label {label!r}")
            if sentence.get("reviewed") is False:
                label = "IGNORE"
            prepared = (text, label)
            existing = group["sentences"].get(ordinal)
            if existing is not None and existing != prepared:
                raise ValueError("Overlapping neural windows disagree on sentence text or label")
            group["sentences"][ordinal] = prepared

    documents = []
    for group in groups.values():
        ordinals = tuple(sorted(group["sentences"]))
        ordered = [group["sentences"][key] for key in ordinals]
        if not ordered:
            continue
        documents.append(
            EvidenceTextDocument(
                document_id=group["document_id"],
                target_id=group["target_id"],
                target_text=group["target_text"],
                sentences=tuple(item[0] for item in ordered),
                sentence_ordinals=ordinals,
                labels=tuple(item[1] for item in ordered),
            )
        )
    return tuple(sorted(documents, key=lambda item: (item.document_id, str(item.target_id))))
