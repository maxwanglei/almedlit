from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    key: str
    label: str
    depends_on: tuple[str, ...] = ()
    requires_profile: str = "any"
    requires_team: bool = False


CAPABILITIES: dict[str, Capability] = {
    c.key: c
    for c in [
        Capability("annotation", "Annotation"),
        Capability(
            "multi_annotator_iaa",
            "Multi-annotator IAA",
            depends_on=("annotation",),
            requires_team=True,
        ),
        Capability(
            "training",
            "Model training",
            depends_on=("lineage",),
        ),
        Capability(
            "inference",
            "Inference",
            depends_on=("training", "lineage"),
        ),
        Capability(
            "active_learning",
            "Active learning",
            depends_on=("annotation", "lineage"),
        ),
        Capability(
            "co_learning",
            "Co-learning",
            depends_on=("annotation", "lineage"),
        ),
        Capability("lineage", "Lineage"),
        Capability("export", "Export", depends_on=("annotation",)),
        Capability(
            "hpc_training",
            "HPC training",
            depends_on=("training",),
            requires_profile="lab",
        ),
        Capability(
            "llm_serving",
            "LLM serving",
            depends_on=("inference",),
            requires_profile="lab",
        ),
    ]
}

PRESETS: dict[str, set[str]] = {
    "annotate": {"annotation"},
    "train": {"training"},
    "annotate_train": {"annotation", "export", "training", "inference"},
    "annotate_train_al": {
        "annotation",
        "export",
        "training",
        "inference",
        "active_learning",
    },
    "full": set(CAPABILITIES),
}


def preset_keys(preset: str, overrides: list[str] | None = None) -> set[str]:
    if preset == "custom":
        return set(overrides or [])
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'")
    return set(PRESETS[preset])


def close_dependencies(keys: set[str]) -> set[str]:
    """Add all transitive dependencies for the selected project modules."""
    result = set(keys)
    changed = True
    while changed:
        changed = False
        for key in list(result):
            if key not in CAPABILITIES:
                raise ValueError(f"Unknown capability '{key}'")
            for dep in CAPABILITIES[key].depends_on:
                if dep not in result:
                    result.add(dep)
                    changed = True
    return result


def _profile_satisfies(required: str, profile: str) -> bool:
    if required == "any":
        return True
    return profile == required


def filter_by_environment(
    keys: set[str],
    *,
    profile: str,
    workspace_kind: str,
) -> set[str]:
    out = set()
    for key in keys:
        capability = CAPABILITIES[key]
        if not _profile_satisfies(capability.requires_profile, profile):
            continue
        if capability.requires_team and workspace_kind != "team":
            continue
        out.add(key)
    return out


def effective_capabilities(
    selected: set[str],
    *,
    profile: str,
    workspace_kind: str,
) -> set[str]:
    return filter_by_environment(
        close_dependencies(selected),
        profile=profile,
        workspace_kind=workspace_kind,
    )
