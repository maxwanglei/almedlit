from al_medlit.co_learning.error_guideline_learning.constants import ERROR_TYPES


def infer_error_type_from_correction(
    *,
    original_span: str | None,
    corrected_span: str | None,
    original_label: str | None,
    corrected_label: str | None,
    evidence_changed: bool = False,
    ontology_changed: bool = False,
) -> str:
    """Rule-based first-pass error phenotyper.

    This is intentionally simple. Later it can be replaced by a trained classifier
    or LLM-assisted phenotyper.
    """

    if evidence_changed:
        return "evidence_error"
    if ontology_changed:
        return "ontology_error"

    if original_label and corrected_label and original_label != corrected_label:
        return "label_confusion"

    if original_span and corrected_span:
        if original_span != corrected_span and (
            original_span in corrected_span or corrected_span in original_span
        ):
            return "boundary_error"
        if original_span != corrected_span:
            return "boundary_error"

    if original_span and not corrected_span:
        return "false_positive"
    if corrected_span and not original_span:
        return "missing_entity"

    return "guideline_ambiguity"


def describe_error_type(error_type: str) -> str:
    return ERROR_TYPES.get(error_type, "Unrecognized error phenotype.")
