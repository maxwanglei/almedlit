from al_medlit.co_learning.error_guideline_learning.phenotyper import (
    infer_error_type_from_correction,
)


def test_boundary_error_detection():
    assert (
        infer_error_type_from_correction(
            original_span="low-dose aspirin therapy",
            corrected_span="aspirin",
            original_label="Drug",
            corrected_label="Drug",
        )
        == "boundary_error"
    )


def test_label_confusion_detection():
    assert (
        infer_error_type_from_correction(
            original_span="rash",
            corrected_span="rash",
            original_label="Disease",
            corrected_label="AdverseEvent",
        )
        == "label_confusion"
    )
