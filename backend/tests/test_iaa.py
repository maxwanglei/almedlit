from types import SimpleNamespace

import pytest

from al_medlit.annotation.models import Annotation
from al_medlit.core.exceptions import ValidationError
from al_medlit.corpus.models import Document
from al_medlit.iaa import alignment, metrics, service
from al_medlit.project.models import Project
from al_medlit.workspace import service as workspace_service


def test_percent_agreement_counts_matching_pairs():
    items = [
        {"a": "Drug", "b": "Drug"},
        {"a": "Drug", "b": "Gene"},
    ]
    assert metrics.percent_agreement(items, ["a", "b"]) == 0.5


def test_percent_agreement_three_raters_averages_over_all_pairs():
    items = [{"a": "X", "b": "X", "c": "Y"}]
    assert metrics.percent_agreement(items, ["a", "b", "c"]) == pytest.approx(1 / 3)


def test_cohens_kappa_perfect_agreement_is_one():
    items = [
        {"a": "X", "b": "X"},
        {"a": "Y", "b": "Y"},
    ]
    assert metrics.cohens_kappa(items, "a", "b") == pytest.approx(1.0)


def test_cohens_kappa_systematic_disagreement_is_minus_one():
    items = [
        {"a": "X", "b": "Y"},
        {"a": "Y", "b": "X"},
    ]
    assert metrics.cohens_kappa(items, "a", "b") == pytest.approx(-1.0)


def test_cohens_kappa_all_same_single_category_returns_one():
    items = [{"a": "X", "b": "X"}]
    assert metrics.cohens_kappa(items, "a", "b") == pytest.approx(1.0)


def test_fleiss_kappa_perfect_agreement_is_one():
    items = [
        {"a": "X", "b": "X", "c": "X"},
        {"a": "Y", "b": "Y", "c": "Y"},
    ]
    assert metrics.fleiss_kappa(items, ["a", "b", "c"]) == pytest.approx(1.0)


def test_fleiss_kappa_two_raters_full_disagreement_is_minus_one():
    items = [{"a": "X", "b": "Y"}]
    assert metrics.fleiss_kappa(items, ["a", "b"]) == pytest.approx(-1.0)


def test_fleiss_kappa_requires_two_raters():
    items = [{"a": "X"}]
    assert metrics.fleiss_kappa(items, ["a"]) is None


def test_pairwise_span_f1_partial_overlap():
    spans_by_annotator = {
        "a": {(1, 0, 5), (1, 6, 9)},
        "b": {(1, 0, 5)},
    }
    assert metrics.pairwise_span_f1(spans_by_annotator, ["a", "b"]) == pytest.approx(2 / 3)


def test_pairwise_span_f1_both_empty_is_one():
    spans_by_annotator = {"a": set(), "b": set()}
    assert metrics.pairwise_span_f1(spans_by_annotator, ["a", "b"]) == pytest.approx(1.0)


def _fake_annotation(annotator_id, document_id, start, end, label):
    return SimpleNamespace(
        annotator_id=annotator_id,
        document_id=document_id,
        start_offset=start,
        end_offset=end,
        label=label,
    )


def test_build_rating_matrix_fills_absent_annotators_with_sentinel():
    annotations = [
        _fake_annotation("alice", 1, 0, 5, "Drug"),
        _fake_annotation("bob", 1, 0, 5, "Drug"),
        _fake_annotation("alice", 1, 6, 9, "Gene"),
    ]
    annotators, items, spans_by_annotator = alignment.build_rating_matrix(annotations)

    assert annotators == ["alice", "bob"]
    assert len(items) == 2
    shared = next(row for row in items if row["alice"] == "Drug" and row["bob"] == "Drug")
    assert shared == {"alice": "Drug", "bob": "Drug"}
    gene_only = next(row for row in items if row["alice"] == "Gene")
    assert gene_only == {"alice": "Gene", "bob": alignment.NONE_LABEL}
    assert spans_by_annotator["alice"] == {(1, 0, 5), (1, 6, 9)}
    assert spans_by_annotator["bob"] == {(1, 0, 5)}


def test_build_rating_matrix_doc_label_uses_none_offsets_as_one_item():
    annotations = [
        _fake_annotation("alice", 7, None, None, "ClinicalTrial"),
        _fake_annotation("bob", 7, None, None, "CaseReport"),
    ]
    annotators, items, _spans = alignment.build_rating_matrix(annotations)
    assert annotators == ["alice", "bob"]
    assert items == [{"alice": "ClinicalTrial", "bob": "CaseReport"}]


def test_build_rating_matrix_duplicate_span_preserves_multi_label_conflict():
    annotations = [
        _fake_annotation("alice", 1, 0, 5, "Gene"),
        _fake_annotation("alice", 1, 0, 5, "Drug"),
        _fake_annotation("bob", 1, 0, 5, "Drug"),
    ]
    _annotators, items, _spans = alignment.build_rating_matrix(annotations)
    assert items == [{"alice": f"{alignment.MULTI_LABEL_PREFIX}Drug|Gene", "bob": "Drug"}]
    assert metrics.percent_agreement(items, ["alice", "bob"]) == 0.0


def test_build_rating_matrix_ignores_annotations_without_annotator():
    annotations = [
        _fake_annotation(None, 1, 0, 5, "Drug"),
        _fake_annotation("alice", 1, 0, 5, "Drug"),
    ]
    annotators, _items, _spans = alignment.build_rating_matrix(annotations)
    assert annotators == ["alice"]


def _make_project_and_document(db, name, workspace_id=None):
    if workspace_id is None:
        workspace_id = workspace_service.ensure_default_workspace(db).id
    project = Project(
        name=name,
        annotation_schema={"labels": {}},
        settings={},
        workspace_id=workspace_id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    document = Document(project_id=project.id, text="Patients receiving aspirin improved.")
    db.add(document)
    db.commit()
    db.refresh(document)
    return project, document


def _add_entity_annotation(
    db,
    project_id,
    document_id,
    annotator_id,
    start,
    end,
    label,
    *,
    structure_version_id=None,
    guideline_version_id=None,
):
    annotation = Annotation(
        project_id=project_id,
        document_id=document_id,
        annotation_type="entity",
        label=label,
        start_offset=start,
        end_offset=end,
        annotator_id=annotator_id,
        structure_version_id=structure_version_id,
        guideline_version_id=guideline_version_id,
    )
    db.add(annotation)
    db.commit()


def test_compute_iaa_reports_insufficient_annotators(db):
    project, document = _make_project_and_document(db, "iaa-insufficient")
    _add_entity_annotation(db, project.id, document.id, "alice", 0, 5, "Drug")

    report = service.compute_iaa(db, project_id=project.id, annotation_type="entity")
    assert report.status == "insufficient_annotators"
    assert report.annotator_ids == ["alice"]
    assert report.cohens_kappa is None


def test_compute_iaa_reports_insufficient_when_no_annotations(db):
    project, _document = _make_project_and_document(db, "iaa-empty")

    report = service.compute_iaa(db, project_id=project.id, annotation_type="entity")
    assert report.status == "insufficient_annotators"
    assert report.annotator_ids == []
    assert report.item_count == 0


def test_compute_iaa_two_annotators_perfect_agreement(db):
    project, document = _make_project_and_document(db, "iaa-perfect")
    _add_entity_annotation(db, project.id, document.id, "alice", 0, 5, "Drug")
    _add_entity_annotation(db, project.id, document.id, "bob", 0, 5, "Drug")

    report = service.compute_iaa(db, project_id=project.id, annotation_type="entity")
    assert report.status == "ok"
    assert report.annotator_ids == ["alice", "bob"]
    assert report.item_count == 1
    assert report.cohens_kappa == pytest.approx(1.0)
    assert report.fleiss_kappa == pytest.approx(1.0)
    assert report.span_detection_f1 == pytest.approx(1.0)
    assert report.percent_agreement == pytest.approx(1.0)


def test_compute_iaa_filters_ordinary_annotations_to_one_version_round(db):
    from al_medlit.corpus import service as corpus_service
    from al_medlit.guideline import service as guideline_service
    from al_medlit.guideline.schemas import GuidelineVersionCreate

    project, document = _make_project_and_document(db, "iaa-version-round")
    structure = corpus_service.rebuild_document_structure(db, document.id)
    guideline = guideline_service.create_guideline_version(
        db,
        GuidelineVersionCreate(
            project_id=project.id,
            version_label="v2",
        ),
    )
    for annotator in ("alice", "bob"):
        _add_entity_annotation(
            db,
            project.id,
            document.id,
            annotator,
            0,
            5,
            "Drug",
            structure_version_id=structure.id,
            guideline_version_id=guideline.id,
        )
        _add_entity_annotation(
            db,
            project.id,
            document.id,
            annotator,
            0,
            5,
            "Legacy",
        )

    report = service.compute_iaa(
        db,
        project_id=project.id,
        annotation_type="entity",
        structure_version_id=structure.id,
        guideline_version_id=guideline.id,
    )
    assert report.status == "ok"
    assert report.item_count == 1
    assert report.percent_agreement == pytest.approx(1.0)


def test_compute_iaa_rejects_unscoped_multiple_ordinary_rounds(db):
    from al_medlit.corpus import service as corpus_service
    from al_medlit.guideline import service as guideline_service
    from al_medlit.guideline.schemas import GuidelineVersionCreate

    project, document = _make_project_and_document(db, "iaa-ambiguous-rounds")
    structure = corpus_service.rebuild_document_structure(db, document.id)
    guideline = guideline_service.create_guideline_version(
        db,
        GuidelineVersionCreate(project_id=project.id, version_label="v2"),
    )
    for annotator in ("alice", "bob"):
        _add_entity_annotation(
            db,
            project.id,
            document.id,
            annotator,
            0,
            5,
            "Drug",
            structure_version_id=structure.id,
            guideline_version_id=guideline.id,
        )
        _add_entity_annotation(
            db,
            project.id,
            document.id,
            annotator,
            0,
            5,
            "Legacy",
        )

    with pytest.raises(ValidationError, match="cannot combine annotation rounds"):
        service.compute_iaa(
            db,
            project_id=project.id,
            annotation_type="entity",
        )

    legacy_report = service.compute_iaa(
        db,
        project_id=project.id,
        annotation_type="entity",
        filter_structure_version=True,
        guideline_version_id=None,
        filter_guideline_version=True,
    )
    assert legacy_report.status == "ok"
    assert legacy_report.item_count == 1
    assert legacy_report.percent_agreement == pytest.approx(1.0)


def test_compute_iaa_multi_label_span_counts_as_disagreement(db):
    project, document = _make_project_and_document(db, "iaa-multi-label-conflict")
    _add_entity_annotation(db, project.id, document.id, "alice", 0, 5, "Gene")
    _add_entity_annotation(db, project.id, document.id, "alice", 0, 5, "Drug")
    _add_entity_annotation(db, project.id, document.id, "bob", 0, 5, "Drug")

    report = service.compute_iaa(db, project_id=project.id, annotation_type="entity")

    assert report.status == "ok"
    assert report.item_count == 1
    assert report.percent_agreement == pytest.approx(0.0)
    assert report.cohens_kappa == pytest.approx(0.0)
    assert report.fleiss_kappa == pytest.approx(-1.0)
    assert report.span_detection_f1 == pytest.approx(1.0)


def test_compute_iaa_span_f1_reflects_partial_overlap(db):
    project, document = _make_project_and_document(db, "iaa-partial")
    _add_entity_annotation(db, project.id, document.id, "alice", 0, 5, "Drug")
    _add_entity_annotation(db, project.id, document.id, "alice", 6, 9, "Gene")
    _add_entity_annotation(db, project.id, document.id, "bob", 0, 5, "Drug")

    report = service.compute_iaa(db, project_id=project.id, annotation_type="entity")
    assert report.status == "ok"
    assert report.item_count == 2
    assert report.span_detection_f1 == pytest.approx(2 / 3)


def test_compute_iaa_doc_label_has_no_span_f1(db):
    project, document = _make_project_and_document(db, "iaa-doc-label")
    for annotator, label in (("alice", "ClinicalTrial"), ("bob", "ClinicalTrial")):
        db.add(
            Annotation(
                project_id=project.id,
                document_id=document.id,
                annotation_type="doc_label",
                label=label,
                annotator_id=annotator,
            )
        )
    db.commit()

    report = service.compute_iaa(db, project_id=project.id, annotation_type="doc_label")
    assert report.status == "ok"
    assert report.span_detection_f1 is None
    assert report.cohens_kappa == pytest.approx(1.0)


def test_iaa_endpoint_requires_authentication(client):
    response = client.get(
        "/api/projects/1/iaa",
        params={"annotation_type": "entity"},
        headers={"Authorization": ""},
    )
    assert response.status_code == 401


def test_iaa_endpoint_returns_report(auth_client, auth_user, db):
    from al_medlit.auth.models import User
    from al_medlit.workspace import capability_service
    from al_medlit.workspace import service as workspace_service

    user = db.get(User, auth_user["id"])
    workspace = workspace_service.create_team_workspace(
        db,
        user,
        name="IAA endpoint team",
    )
    capability_service.set_capability(
        db,
        workspace.id,
        preset="full",
        actor_user_id=user.id,
    )
    project, document = _make_project_and_document(
        db,
        "iaa-endpoint",
        workspace_id=workspace.id,
    )
    _add_entity_annotation(db, project.id, document.id, "alice", 0, 5, "Drug")
    _add_entity_annotation(db, project.id, document.id, "bob", 0, 5, "Drug")

    response = auth_client.get(
        f"/api/projects/{project.id}/iaa",
        params={"annotation_type": "entity"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["annotator_ids"] == ["alice", "bob"]
    assert body["cohens_kappa"] == pytest.approx(1.0)
    assert body["span_detection_f1"] == pytest.approx(1.0)


def test_iaa_endpoint_requires_multi_annotator_capability(auth_client, auth_user, db):
    project, _document = _make_project_and_document(
        db,
        "iaa-capability-off",
        workspace_id=auth_user["workspace_id"],
    )
    response = auth_client.get(
        f"/api/projects/{project.id}/iaa",
        params={"annotation_type": "entity"},
    )
    assert response.status_code == 403


def test_iaa_endpoint_rejects_unknown_annotation_type(auth_client, auth_user, db):
    project, _document = _make_project_and_document(
        db,
        "iaa-bad-type",
        workspace_id=auth_user["workspace_id"],
    )
    response = auth_client.get(
        f"/api/projects/{project.id}/iaa",
        params={"annotation_type": "bogus"},
    )
    assert response.status_code == 422
