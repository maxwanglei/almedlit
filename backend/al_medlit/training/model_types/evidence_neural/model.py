"""Torch BiLSTM/CNN implementations with safe package primitives."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from al_medlit.training.model_types.evidence_neural.config import (
    EvidenceBiLSTMConfig,
    EvidenceCNNConfig,
    EvidenceNeuralConfig,
)
from al_medlit.training.model_types.evidence_neural.data import (
    EvidenceSequencePrediction,
    EvidenceTextDocument,
    NeuralCheckpointScore,
    NeuralEpochMetric,
    NeuralTrainingSummary,
)
from al_medlit.training.model_types.evidence_neural.device import require_neural_device
from al_medlit.training.model_types.evidence_neural.vocabulary import (
    EvidenceVocabulary,
    build_vocabulary,
)

LABEL_TO_ID = {"O": 0, "B": 1, "I": 2, "IGNORE": -100}
ID_TO_LABEL = ("O", "B", "I")


class MissingNeuralDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EncodedEvidenceDocument:
    source: EvidenceTextDocument
    sentence_token_ids: tuple[tuple[int, ...], ...]
    label_ids: tuple[int, ...] | None


@dataclass(slots=True)
class EvidenceNeuralBundle:
    model: object
    vocabulary: EvidenceVocabulary
    config: EvidenceBiLSTMConfig | EvidenceCNNConfig

    def save_pretrained(self, destination: str | Path) -> dict:
        """Write a deployable package without pickle or executable payloads."""

        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - optional dependency branch
            raise MissingNeuralDependencyError(
                "Neural packaging requires the optional safetensors dependency"
            ) from exc

        target = Path(destination)
        if target.exists() and target.is_symlink():
            raise ValueError("Checkpoint destination cannot be a symlink")
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "al-medlit-model.json"
        vocabulary_path = target / "vocabulary.json"
        weights_path = target / "model.safetensors"
        config_path.write_text(self.config.model_dump_json(indent=2) + "\n", encoding="utf-8")
        self.vocabulary.save(vocabulary_path)
        tensors = {
            name: tensor.detach().to("cpu").contiguous()
            for name, tensor in self.model.state_dict().items()
        }
        save_file(
            tensors,
            weights_path,
            metadata={
                "format": "pt",
                "model_kind": self.config.model_kind,
                "task_contract": "evidence_blocks@1",
            },
        )
        files = tuple(
            _file_descriptor(path, target) for path in (config_path, vocabulary_path, weights_path)
        )
        manifest = {
            "schema_version": "evidence-neural-package-v1",
            "model_family": "deep_learning",
            "model_type": f"evidence_{self.config.model_kind}",
            "task_contract": {"key": "evidence_blocks", "version": "1"},
            "safe_serialization": True,
            "files": files,
        }
        (target / "package.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest


def canonical_checkpoint_is_better(
    candidate: NeuralCheckpointScore,
    incumbent: NeuralCheckpointScore | None,
) -> bool:
    """Apply the Evidence task's validation-only checkpoint ordering."""

    if incumbent is None:
        return True
    return (
        candidate.macro_block_iou_f1_0_50,
        candidate.macro_exact_block_f1,
    ) > (
        incumbent.macro_block_iou_f1_0_50,
        incumbent.macro_exact_block_f1,
    )


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency branch
        raise MissingNeuralDependencyError(
            "BiLSTM/CNN execution requires the optional torch dependency"
        ) from exc
    return torch


def _seed_everything(torch, config: EvidenceNeuralConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available() and hasattr(npu, "manual_seed_all"):
        npu.manual_seed_all(config.seed)
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        if cudnn is not None:
            cudnn.benchmark = False


def _model_class(torch, config: EvidenceBiLSTMConfig | EvidenceCNNConfig):
    if isinstance(config, EvidenceBiLSTMConfig):

        class EvidenceBiLSTM(torch.nn.Module):
            def __init__(self, vocabulary_size: int) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(
                    vocabulary_size,
                    config.embedding_dimension,
                    padding_idx=0,
                )
                self.encoder = torch.nn.LSTM(
                    input_size=config.embedding_dimension,
                    hidden_size=config.hidden_dimension,
                    num_layers=config.recurrent_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
                )
                self.dropout = torch.nn.Dropout(config.dropout)
                self.classifier = torch.nn.Linear(config.hidden_dimension * 2, 3)

            def forward(self, *, input_ids, token_mask, sentence_mask, labels=None):
                embedded = self.embedding(input_ids)
                token_weights = token_mask.unsqueeze(-1).to(embedded.dtype)
                sentence_embeddings = (embedded * token_weights).sum(dim=2)
                sentence_embeddings = sentence_embeddings / token_weights.sum(dim=2).clamp(min=1)
                lengths = sentence_mask.sum(dim=1).to("cpu").clamp(min=1)
                packed = torch.nn.utils.rnn.pack_padded_sequence(
                    sentence_embeddings,
                    lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_output, _ = self.encoder(packed)
                encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                    total_length=input_ids.shape[1],
                )
                logits = self.classifier(self.dropout(encoded))
                return _output_with_loss(torch, logits, labels)

        return EvidenceBiLSTM

    class EvidenceSentenceCNN(torch.nn.Module):
        def __init__(self, vocabulary_size: int) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(
                vocabulary_size,
                config.embedding_dimension,
                padding_idx=0,
            )
            self.token_convolutions = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        config.embedding_dimension,
                        config.convolution_channels,
                        kernel_size=kernel,
                    )
                    for kernel in config.token_kernel_sizes
                ]
            )
            sentence_features = config.convolution_channels * len(config.token_kernel_sizes)
            self.context_convolution = torch.nn.Conv1d(
                sentence_features,
                config.convolution_channels,
                kernel_size=config.sentence_context_kernel_size,
                padding=config.sentence_context_kernel_size // 2,
            )
            self.dropout = torch.nn.Dropout(config.dropout)
            self.classifier = torch.nn.Linear(config.convolution_channels, 3)

        def forward(self, *, input_ids, token_mask, sentence_mask, labels=None):
            batch_size, sentence_count, token_count = input_ids.shape
            flattened = input_ids.reshape(batch_size * sentence_count, token_count)
            embedded = self.embedding(flattened).transpose(1, 2)
            maximum_kernel = max(config.token_kernel_sizes)
            if token_count < maximum_kernel:
                embedded = torch.nn.functional.pad(
                    embedded,
                    (0, maximum_kernel - token_count),
                )
            features = []
            for convolution in self.token_convolutions:
                convolved = torch.relu(convolution(embedded))
                features.append(torch.amax(convolved, dim=2))
            sentence_features = torch.cat(features, dim=1).reshape(
                batch_size,
                sentence_count,
                -1,
            )
            contextual = torch.relu(
                self.context_convolution(sentence_features.transpose(1, 2))
            ).transpose(1, 2)
            logits = self.classifier(self.dropout(contextual))
            return _output_with_loss(torch, logits, labels)

    return EvidenceSentenceCNN


def _output_with_loss(torch, logits, labels):
    loss = None
    if labels is not None:
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3),
            labels.reshape(-1),
            ignore_index=-100,
        )
    return {"loss": loss, "logits": logits}


def build_neural_bundle(
    config: EvidenceBiLSTMConfig | EvidenceCNNConfig,
    vocabulary: EvidenceVocabulary,
) -> EvidenceNeuralBundle:
    torch = _import_torch()
    _seed_everything(torch, config)
    model_class = _model_class(torch, config)
    return EvidenceNeuralBundle(
        model=model_class(len(vocabulary)),
        vocabulary=vocabulary,
        config=config,
    )


def encode_documents(
    documents: Iterable[EvidenceTextDocument],
    bundle: EvidenceNeuralBundle,
) -> tuple[EncodedEvidenceDocument, ...]:
    encoded: list[EncodedEvidenceDocument] = []
    for document in documents:
        if len(document.sentences) > bundle.config.max_sentences_per_document:
            raise ValueError(
                "Prepared neural inputs must be windowed; sentences are never silently truncated"
            )
        sentences = document.sentences
        labels = (
            tuple(LABEL_TO_ID[label] for label in document.labels[: len(sentences)])
            if document.labels is not None
            else None
        )
        encoded.append(
            EncodedEvidenceDocument(
                source=document,
                sentence_token_ids=tuple(
                    bundle.vocabulary.encode_sentence(
                        target_text=document.target_text,
                        sentence=sentence,
                        max_tokens=bundle.config.max_tokens_per_sentence,
                    )
                    for sentence in sentences
                ),
                label_ids=labels,
            )
        )
    return tuple(encoded)


def fit_neural_model(
    config: EvidenceBiLSTMConfig | EvidenceCNNConfig,
    training_documents: Sequence[EvidenceTextDocument],
    *,
    validation_documents: Sequence[EvidenceTextDocument] = (),
    vocabulary: EvidenceVocabulary | None = None,
    validation_selector: Callable[[EvidenceNeuralBundle, int], NeuralCheckpointScore] | None = None,
) -> tuple[EvidenceNeuralBundle, NeuralTrainingSummary]:
    if not training_documents:
        raise ValueError("At least one training document is required")
    if any(document.labels is None for document in training_documents):
        raise ValueError("Every training document requires aligned labels")
    if any(document.labels is None for document in validation_documents):
        raise ValueError("Every validation document requires aligned labels")
    if not any(
        label != "IGNORE" for document in training_documents for label in document.labels or ()
    ):
        raise ValueError("Training data contains no supervised Evidence labels")
    if validation_documents and not any(
        label != "IGNORE" for document in validation_documents for label in document.labels or ()
    ):
        raise ValueError("Validation data contains no supervised Evidence labels")
    selected_vocabulary = vocabulary or build_vocabulary(training_documents, config)
    bundle = build_neural_bundle(config, selected_vocabulary)
    torch = _import_torch()
    preflight = require_neural_device(config.device)
    device = torch.device(preflight.torch_device)
    bundle.model.to(device)
    training = encode_documents(training_documents, bundle)
    validation = encode_documents(validation_documents, bundle)
    optimizer = torch.optim.AdamW(
        bundle.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batch_count = math.ceil(len(training) / config.batch_size)
    updates_per_epoch = math.ceil(batch_count / config.gradient_accumulation_steps)
    total_updates = max(1, config.epochs * updates_per_epoch)
    warmup_updates = int(total_updates * config.warmup_ratio)
    scheduler = None
    if config.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _linear_schedule(step, total_updates, warmup_updates),
        )

    best_state = None
    best_checkpoint_score = None
    best_rank = (-math.inf, -math.inf)
    best_value = math.inf
    best_tiebreaker = None
    best_epoch = 1
    stale_epochs = 0
    history: list[NeuralEpochMetric] = []
    for epoch in range(1, config.epochs + 1):
        order = list(range(len(training)))
        random.Random(config.seed + epoch).shuffle(order)
        ordered_training = tuple(training[index] for index in order)
        bundle.model.train()
        optimizer.zero_grad(set_to_none=True)
        cumulative_loss = 0.0
        for batch_number, batch in enumerate(
            _batches(ordered_training, config.batch_size),
            start=1,
        ):
            tensors = _collate(torch, batch, device=device, include_labels=True)
            output = bundle.model(**tensors)
            loss = output["loss"]
            cumulative_loss += float(loss.detach().to("cpu"))
            (loss / config.gradient_accumulation_steps).backward()
            should_update = (
                batch_number % config.gradient_accumulation_steps == 0
                or batch_number == batch_count
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    bundle.model.parameters(),
                    config.gradient_clip_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        train_loss = cumulative_loss / batch_count
        validation_loss = (
            _mean_loss(torch, bundle, validation, device=device) if validation else None
        )
        selection_value = validation_loss if validation_loss is not None else train_loss
        checkpoint_score = None
        if validation_selector is not None:
            bundle.model.eval()
            try:
                checkpoint_score = NeuralCheckpointScore.model_validate(
                    validation_selector(bundle, epoch)
                )
            finally:
                bundle.model.to(device)
                bundle.model.train()
        if checkpoint_score is not None:
            current_rank = (
                checkpoint_score.macro_block_iou_f1_0_50,
                checkpoint_score.macro_exact_block_f1,
            )
            displayed_selection_value = checkpoint_score.macro_block_iou_f1_0_50
        else:
            current_rank = (-selection_value, 0.0)
            displayed_selection_value = selection_value
        history.append(
            NeuralEpochMetric(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                validation_macro_block_iou_f1_0_50=(
                    checkpoint_score.macro_block_iou_f1_0_50
                    if checkpoint_score is not None
                    else None
                ),
                validation_macro_exact_block_f1=(
                    checkpoint_score.macro_exact_block_f1 if checkpoint_score is not None else None
                ),
            )
        )
        if checkpoint_score is not None:
            # A full tie does not replace the incumbent, so the earlier epoch wins.
            canonical_improved = canonical_checkpoint_is_better(
                checkpoint_score,
                best_checkpoint_score,
            )
            primary_improved = canonical_improved
            primary_tied = False
        else:
            primary_improved = current_rank[0] > (best_rank[0] + config.early_stopping_min_delta)
            primary_tied = abs(current_rank[0] - best_rank[0]) <= (config.early_stopping_min_delta)
        tiebreaker_improved = primary_tied and current_rank[1] > best_rank[1]
        if primary_improved or tiebreaker_improved:
            best_rank = current_rank
            best_checkpoint_score = checkpoint_score
            best_value = displayed_selection_value
            best_tiebreaker = current_rank[1] if checkpoint_score is not None else None
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().to("cpu").clone()
                for name, tensor in bundle.model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if config.early_stopping_patience and stale_epochs >= config.early_stopping_patience:
            break

    if best_state is None:  # Defensive: the first finite epoch always populates it.
        raise RuntimeError("Training did not produce a finite checkpoint")
    bundle.model.load_state_dict(best_state, strict=True)
    bundle.model.to("cpu")
    summary = NeuralTrainingSummary(
        history=tuple(history),
        best_epoch=best_epoch,
        selection_metric=(
            "macro_block_iou_f1_0_50"
            if validation_selector is not None
            else ("validation_loss" if validation else "train_loss")
        ),
        selection_value=best_value,
        selection_tiebreaker_exact_block_f1=best_tiebreaker,
        stopped_early=len(history) < config.epochs,
        device=str(preflight.resolved_device),
    )
    return bundle, summary


def predict_neural_model(
    bundle: EvidenceNeuralBundle,
    documents: Sequence[EvidenceTextDocument],
    *,
    device: str | None = None,
) -> tuple[EvidenceSequencePrediction, ...]:
    if not documents:
        return ()
    torch = _import_torch()
    preflight = require_neural_device(device or bundle.config.device)
    torch_device = torch.device(preflight.torch_device)
    bundle.model.to(torch_device)
    bundle.model.eval()
    encoded = encode_documents(documents, bundle)
    predictions: list[EvidenceSequencePrediction] = []
    with torch.no_grad():
        for batch in _batches(encoded, bundle.config.batch_size):
            tensors = _collate(torch, batch, device=torch_device, include_labels=False)
            logits = bundle.model(**tensors)["logits"]
            probabilities = torch.softmax(logits, dim=-1).detach().to("cpu")
            for row, item in enumerate(batch):
                sentence_count = len(item.sentence_token_ids)
                scores = tuple(
                    tuple(float(value) for value in probabilities[row, index].tolist())
                    for index in range(sentence_count)
                )
                predictions.append(
                    EvidenceSequencePrediction(
                        document_id=item.source.document_id,
                        target_id=item.source.target_id,
                        sentence_ordinals=(
                            item.source.sentence_ordinals
                            if item.source.sentence_ordinals is not None
                            else tuple(range(sentence_count))
                        ),
                        labels=tuple(
                            ID_TO_LABEL[max(range(3), key=lambda label: scores[index][label])]
                            for index in range(sentence_count)
                        ),
                        probabilities=scores,
                    )
                )
    bundle.model.to("cpu")
    return tuple(predictions)


def load_neural_bundle(checkpoint_directory: str | Path) -> EvidenceNeuralBundle:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency branch
        raise MissingNeuralDependencyError(
            "Neural checkpoint loading requires torch and safetensors"
        ) from exc
    root = Path(checkpoint_directory)
    required = (
        root / "al-medlit-model.json",
        root / "vocabulary.json",
        root / "model.safetensors",
    )
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise ValueError("Neural checkpoint is missing a regular, non-symlink package file")
    raw_config = json.loads(required[0].read_text(encoding="utf-8"))
    model_kind = raw_config.get("model_kind")
    if model_kind == "bilstm":
        config = EvidenceBiLSTMConfig.model_validate(raw_config)
    elif model_kind == "cnn":
        config = EvidenceCNNConfig.model_validate(raw_config)
    else:
        raise ValueError("Unsupported neural checkpoint model_kind")
    bundle = build_neural_bundle(config, EvidenceVocabulary.load(required[1]))
    state = load_file(required[2], device="cpu")
    bundle.model.load_state_dict(state, strict=True)
    bundle.model.eval()
    return bundle


def _collate(torch, batch, *, device, include_labels: bool) -> dict:
    sentence_count = max(len(item.sentence_token_ids) for item in batch)
    token_count = max(len(token_ids) for item in batch for token_ids in item.sentence_token_ids)
    input_ids = torch.zeros(
        (len(batch), sentence_count, token_count),
        dtype=torch.long,
        device=device,
    )
    token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    sentence_mask = torch.zeros(
        (len(batch), sentence_count),
        dtype=torch.bool,
        device=device,
    )
    labels = torch.full(
        (len(batch), sentence_count),
        -100,
        dtype=torch.long,
        device=device,
    )
    for row, item in enumerate(batch):
        for sentence_index, token_ids in enumerate(item.sentence_token_ids):
            length = len(token_ids)
            input_ids[row, sentence_index, :length] = torch.tensor(
                token_ids,
                dtype=torch.long,
                device=device,
            )
            token_mask[row, sentence_index, :length] = True
            sentence_mask[row, sentence_index] = True
            if include_labels and item.label_ids is not None:
                labels[row, sentence_index] = item.label_ids[sentence_index]
    tensors = {
        "input_ids": input_ids,
        "token_mask": token_mask,
        "sentence_mask": sentence_mask,
    }
    if include_labels:
        tensors["labels"] = labels
    return tensors


def _batches(items: Sequence, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _mean_loss(torch, bundle, documents, *, device) -> float:
    bundle.model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in _batches(documents, bundle.config.batch_size):
            tensors = _collate(torch, batch, device=device, include_labels=True)
            total += float(bundle.model(**tensors)["loss"].detach().to("cpu"))
            count += 1
    return total / count


def _linear_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-12, step / warmup_steps)
    remaining = total_steps - step
    decay_steps = max(1, total_steps - warmup_steps)
    return max(0.0, remaining / decay_steps)


def _file_descriptor(path: Path, root: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }
