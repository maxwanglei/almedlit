import pytest

from al_medlit.annotation import types


def _payload(**overrides):
    base = {
        "annotation_type": "entity",
        "start_offset": 0,
        "end_offset": 5,
        "head_annotation_id": None,
        "tail_annotation_id": None,
    }
    base.update(overrides)
    return base


def test_entity_requires_span():
    with pytest.raises(types.AnnotationValidationError):
        types.validate_create(_payload(start_offset=None, end_offset=None))


def test_relation_requires_head_tail():
    with pytest.raises(types.AnnotationValidationError):
        types.validate_create(_payload(annotation_type="relation"))

    types.validate_create(
        _payload(
            annotation_type="relation",
            head_annotation_id=1,
            tail_annotation_id=2,
        )
    )


def test_unknown_type_rejected():
    with pytest.raises(types.AnnotationValidationError):
        types.validate_create(_payload(annotation_type="frobnicate"))


def test_doc_label_needs_no_span():
    types.validate_create(
        _payload(
            annotation_type="doc_label",
            start_offset=None,
            end_offset=None,
        )
    )


def test_non_relation_type_rejects_head_or_tail():
    with pytest.raises(types.AnnotationValidationError, match="must not set"):
        types.validate_create(_payload(head_annotation_id=1))

    with pytest.raises(types.AnnotationValidationError, match="must not set"):
        types.validate_create(_payload(tail_annotation_id=1))


def test_evidence_block_advertises_sentence_id_selection_without_character_span():
    spec = types.REGISTRY["evidence_block"]

    assert spec.requires_span is False
    assert spec.selection_mode == "sentence_range"
    assert spec.renderer_key == "evidence_block_v1"
    assert spec.relation_endpoint_allowed is False
