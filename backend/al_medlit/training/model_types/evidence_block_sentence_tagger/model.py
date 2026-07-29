import json
from dataclasses import dataclass
from pathlib import Path

from al_medlit.training.model_types.evidence_block_sentence_tagger.config import (
    EvidenceBlockSentenceTaggerConfig,
)


class MissingMLDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SentenceTaggerBundle:
    model: object
    tokenizer: object
    config: EvidenceBlockSentenceTaggerConfig

    def save_pretrained(self, destination: str | Path) -> None:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(target)
        self.model.encoder.save_pretrained(target / "encoder")
        # The classifier is persisted without executable pickle payloads.
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - exercised without the ml extra
            raise MissingMLDependencyError("Install the optional 'ml' dependencies") from exc
        save_file(
            self.model.classifier.state_dict(),
            target / "sentence_classifier.safetensors",
        )
        (target / "al-medlit-model.json").write_text(
            self.config.model_dump_json(indent=2),
            encoding="utf-8",
        )


def build_sentence_tagger(
    config: EvidenceBlockSentenceTaggerConfig,
) -> SentenceTaggerBundle:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise MissingMLDependencyError(
            "Evidence-block model construction requires the optional torch/transformers ML extra"
        ) from exc

    source = config.local_model_path or config.model_id
    if source is None:
        raise ValueError("A staged model artifact must be resolved to local_model_path first")
    load_kwargs = {
        "trust_remote_code": False,
        "local_files_only": config.local_files_only,
    }
    if config.model_id is not None:
        load_kwargs["revision"] = config.revision
    tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
    encoder = AutoModel.from_pretrained(source, **load_kwargs)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                config.target_marker_token,
                config.sentence_marker_token,
            ]
        }
    )
    encoder.resize_token_embeddings(len(tokenizer))
    encoder_limit = int(getattr(encoder.config, "max_position_embeddings", 0) or 0)
    if encoder_limit and config.max_length > encoder_limit:
        raise ValueError(
            f"Configured context {config.max_length} exceeds encoder capacity {encoder_limit}"
        )

    SentenceTagger = _sentence_tagger_class(torch)
    return SentenceTaggerBundle(
        model=SentenceTagger(encoder),
        tokenizer=tokenizer,
        config=config,
    )


def load_sentence_tagger(checkpoint_directory: str | Path) -> SentenceTaggerBundle:
    """Load the exact tokenizer, encoder, and O/B/I head saved by a worker."""

    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise MissingMLDependencyError(
            "Checkpoint loading requires the optional torch/transformers ML extra"
        ) from exc
    root = Path(checkpoint_directory)
    config = EvidenceBlockSentenceTaggerConfig.model_validate(
        json.loads((root / "al-medlit-model.json").read_text(encoding="utf-8"))
    )
    tokenizer = AutoTokenizer.from_pretrained(
        root,
        trust_remote_code=False,
        local_files_only=True,
    )
    encoder = AutoModel.from_pretrained(
        root / "encoder",
        trust_remote_code=False,
        local_files_only=True,
    )
    SentenceTagger = _sentence_tagger_class(torch)
    model = SentenceTagger(encoder)
    classifier_state = load_file(root / "sentence_classifier.safetensors", device="cpu")
    model.classifier.load_state_dict(classifier_state)
    return SentenceTaggerBundle(model=model, tokenizer=tokenizer, config=config)


def _sentence_tagger_class(torch):
    class SentenceTagger(torch.nn.Module):
        def __init__(self, base_encoder) -> None:
            super().__init__()
            self.encoder = base_encoder
            self.classifier = torch.nn.Linear(base_encoder.config.hidden_size, 3)

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            sentence_marker_positions,
            labels=None,
            class_weights=None,
        ):
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = encoded.last_hidden_state
            safe_positions = sentence_marker_positions.clamp(min=0)
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
            sentence_hidden = hidden[batch_indices, safe_positions]
            logits = self.classifier(sentence_hidden)
            loss = None
            if labels is not None:
                weights = None
                if class_weights is not None:
                    weights = torch.as_tensor(
                        class_weights,
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 3),
                    labels.reshape(-1),
                    weight=weights,
                    ignore_index=-100,
                )
            return {"loss": loss, "logits": logits}

    return SentenceTagger
