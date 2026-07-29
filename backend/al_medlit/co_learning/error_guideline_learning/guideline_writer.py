from al_medlit.co_learning.error_guideline_learning.models import ErrorPattern


def draft_guideline_atom_from_pattern(pattern: ErrorPattern) -> tuple[str, str]:
    """Drafts a guideline atom without an LLM.

    This deterministic writer is useful in Phase 1/2. Later, an LLM adapter can
    produce richer text, but manager approval should remain mandatory.
    """

    if pattern.error_type == "boundary_error":
        rule = (
            f"For {pattern.task_type} annotation"
            f"{' of ' + pattern.label_type if pattern.label_type else ''}, "
            "annotate the minimal biomedical concept span required by the schema. "
            "Exclude dose, route, frequency, generic context words, and surrounding phrases "
            "unless they are part of the official named concept."
        )
    elif pattern.error_type == "ontology_error":
        rule = (
            "When normalizing a biomedical mention, prefer the concept granularity specified "
            "by the project schema. Record ambiguous mappings as candidates rather than forcing "
            "an unsupported normalization."
        )
    elif pattern.error_type == "role_confusion":
        rule = (
            "Assign the contextual scientific role of a mention using local evidence. "
            "Distinguish intervention, exposure, comparator, background mention, metabolite, "
            "and outcome-related usage when applicable."
        )
    elif pattern.error_type == "evidence_error":
        rule = (
            "Do not accept an annotation or relation unless the selected evidence span directly "
            "supports it. If support is indirect or absent, mark the case for review."
        )
    elif pattern.error_type == "missing_entity":
        rule = (
            "Check abbreviation-only, synonym, brand, and rare terminology mentions carefully. "
            "Use project-approved ontology and synonym resources to avoid missed entities."
        )
    elif pattern.error_type == "false_positive":
        rule = (
            "Reject mentions that are generic context words, procedures, diseases, outcomes, "
            "or non-target concepts unless they satisfy the project schema."
        )
    else:
        rule = (
            "Clarify this recurring annotation pattern in the guideline. Include positive "
            "and negative examples so annotators and models can apply the rule consistently."
        )

    rationale = (
        f"Drafted from recurring error pattern #{pattern.id}: "
        f"{pattern.error_type} with {pattern.example_count} example(s)."
    )
    return rule, rationale
