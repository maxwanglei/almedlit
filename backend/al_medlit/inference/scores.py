"""Portable score rows shared by local and remote evidence inference."""

_LABEL_PROBABILITIES = {
    "O": (1.0, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0),
    "I": (0.0, 0.0, 1.0),
}


def label_probabilities(label: str) -> tuple[float, float, float]:
    return _LABEL_PROBABILITIES[label]


def synthetic_logits(ordinal: int) -> tuple[float, float, float]:
    if ordinal == 0:
        return (0.0, 6.0, 0.0)
    if ordinal == 1:
        return (0.0, 0.0, 6.0)
    return (6.0, 0.0, 0.0)
