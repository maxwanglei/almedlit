ERROR_TYPES = {
    "boundary_error": "Entity boundary is too broad, too narrow, or shifted.",
    "missing_entity": "A true entity was missed.",
    "false_positive": "A non-entity was labeled as an entity.",
    "label_confusion": "Wrong label type was assigned.",
    "role_confusion": "Entity role in scientific context was wrong.",
    "ontology_error": "Normalization or concept mapping was wrong.",
    "evidence_error": "Annotation or relation lacks adequate evidence support.",
    "guideline_ambiguity": "Current guideline is insufficient or unclear.",
    "domain_shift": "New terminology, abbreviation, or rare case causes failure.",
}

DEFAULT_MICRO_QUESTIONS = {
    "boundary_error": "What is the correct span boundary?",
    "ontology_error": "Which normalized concept is correct?",
    "role_confusion": "What role does this biomedical mention play in context?",
    "evidence_error": "Does the selected evidence support this annotation?",
    "guideline_ambiguity": "Should this case become a guideline clarification?",
}
