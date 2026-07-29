import pytest

from al_medlit.core import capabilities as cap


def test_close_dependencies_pulls_in_transitive_deps():
    closed = cap.close_dependencies({"active_learning"})
    assert closed == {
        "annotation",
        "lineage",
        "active_learning",
    }


def test_training_does_not_force_annotation():
    assert cap.close_dependencies({"training"}) == {"training", "lineage"}


def test_empty_capability_selection_stays_empty():
    assert cap.close_dependencies(set()) == set()


def test_lab_only_caps_dropped_on_laptop():
    closed = cap.close_dependencies({"hpc_training", "annotation"})
    eff = cap.filter_by_environment(closed, profile="laptop", workspace_kind="team")
    assert "hpc_training" not in eff
    assert "training" in eff


def test_lab_only_caps_present_on_lab():
    closed = cap.close_dependencies({"hpc_training"})
    eff = cap.filter_by_environment(closed, profile="lab", workspace_kind="team")
    assert "hpc_training" in eff


def test_team_only_cap_dropped_for_individual():
    closed = cap.close_dependencies({"multi_annotator_iaa"})
    eff = cap.filter_by_environment(closed, profile="lab", workspace_kind="individual")
    assert "multi_annotator_iaa" not in eff


def test_team_only_cap_present_for_team():
    closed = cap.close_dependencies({"multi_annotator_iaa"})
    eff = cap.filter_by_environment(closed, profile="lab", workspace_kind="team")
    assert "multi_annotator_iaa" in eff


def test_preset_keys_for_annotate():
    assert cap.preset_keys("annotate") == {"annotation"}


def test_training_preset_is_independent_of_annotation():
    assert cap.effective_capabilities(
        cap.preset_keys("train"),
        profile="laptop",
        workspace_kind="individual",
    ) == {"training", "lineage"}


def test_preset_keys_custom_uses_overrides():
    assert cap.preset_keys("custom", ["training"]) == {"training"}


def test_preset_keys_rejects_unknown():
    with pytest.raises(ValueError):
        cap.preset_keys("nonsense")


def test_full_preset_on_individual_laptop_excludes_team_and_lab():
    eff = cap.effective_capabilities(
        cap.preset_keys("full"),
        profile="laptop",
        workspace_kind="individual",
    )
    assert "multi_annotator_iaa" not in eff
    assert "hpc_training" not in eff
    assert "llm_serving" not in eff
    assert {
        "annotation",
        "training",
        "inference",
        "active_learning",
        "co_learning",
        "lineage",
        "export",
    } <= eff


def test_full_preset_on_team_lab_includes_everything():
    eff = cap.effective_capabilities(
        cap.preset_keys("full"),
        profile="lab",
        workspace_kind="team",
    )
    assert eff == set(cap.CAPABILITIES)


def test_workspace_has_capability_columns(db):
    from al_medlit.auth.models import User
    from al_medlit.workspace.models import Workspace

    user = User(username="capowner", password_hash="x")
    db.add(user)
    db.flush()
    ws = Workspace(name="W", kind="individual", created_by=user.id)
    db.add(ws)
    db.flush()
    assert ws.capability_preset == "annotate"
    assert ws.capability_overrides == []


def test_project_has_workspace_id(db):
    from al_medlit.project.models import Project

    assert hasattr(Project, "workspace_id")
