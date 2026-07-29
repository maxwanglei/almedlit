import pytest
from pydantic import ValidationError

from al_medlit.training.model_types.evidence_block_sentence_tagger.config import (
    EvidenceBlockSentenceTaggerConfig,
)
from al_medlit.training.model_types.evidence_block_sentence_tagger.dataset import (
    IGNORE_INDEX,
    balanced_class_weights,
    encode_labels,
    encode_sentence_markers,
)


class FakeTokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(10, 10 + len(text.split())))

    def convert_tokens_to_ids(self, token):
        return {"[TARGET]": 3, "[SENT]": 4}[token]


def test_model_config_requires_a_pinned_remote_revision():
    with pytest.raises(ValidationError, match="immutable revision"):
        EvidenceBlockSentenceTaggerConfig(model_id="encoder", max_length=512)

    config = EvidenceBlockSentenceTaggerConfig(
        model_id="allenai/longformer-base-4096",
        revision="0123456789abcdef",
        max_length=4096,
        encoder_max_length=4096,
    )
    assert config.local_files_only is True


def test_model_config_rejects_context_beyond_encoder_capacity():
    with pytest.raises(ValidationError, match="encoder capacity"):
        EvidenceBlockSentenceTaggerConfig(
            local_model_path="/models/tiny",
            max_length=1024,
            encoder_max_length=512,
        )


def test_marker_encoding_preserves_sentence_to_marker_map_without_truncation():
    encoded = encode_sentence_markers(
        FakeTokenizer(),
        target_text="target words",
        sentences=["first sentence", "second"],
        max_length=20,
    )
    assert encoded.input_ids[encoded.sentence_marker_positions[0]] == 4
    assert encoded.input_ids[encoded.sentence_marker_positions[1]] == 4
    assert len(encoded.attention_mask) == len(encoded.input_ids)

    with pytest.raises(ValueError, match="never silently truncated"):
        encode_sentence_markers(
            FakeTokenizer(),
            target_text="target",
            sentences=["too many tokens in this sentence"],
            max_length=4,
        )


def test_ignore_labels_and_balanced_weights():
    assert encode_labels(["O", "B", "I", "IGNORE"]) == (0, 1, 2, IGNORE_INDEX)
    weights = balanced_class_weights([[0, 0, 1, 2, IGNORE_INDEX]])
    assert weights[0] < weights[1]
    assert weights[1] == weights[2]
