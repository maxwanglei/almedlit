import pytest

from al_medlit.auth.models import User
from al_medlit.auth.security import hash_password
from al_medlit.core.exceptions import ConflictError
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import (
    Document,
    DocumentParagraph,
    DocumentSection,
    DocumentSentence,
    DocumentStructureVersion,
    ImmutableDocumentStructureError,
)
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.corpus.segmentation import segment_document, segment_sentences
from al_medlit.importers.pubmed import parse_pmc_xml_documents
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.workspace import service as workspace_service


def test_sentence_segmentation_is_unicode_safe_and_abbreviation_aware():
    text = "Dr. García measured 3.5 mg. Café 🧪 improved.\n\nNext cohort?"

    spans = segment_sentences(text)

    assert [text[start:end] for start, end in spans] == [
        "Dr. García measured 3.5 mg.",
        "Café 🧪 improved.",
        "Next cohort?",
    ]
    assert spans[-1][1] == len(text)


def test_plain_segmentation_is_deterministic_and_uses_zero_based_ordinals():
    text = "First sentence.\n\nSecond paragraph has one. And two."

    first = segment_document(text)
    second = segment_document(text)

    assert first == second
    assert len(first.sections) == 1
    assert first.sections[0].kind == "unknown"
    assert [paragraph.ordinal for paragraph in first.sections[0].paragraphs] == [0, 1]
    assert [
        sentence.ordinal
        for paragraph in first.sections[0].paragraphs
        for sentence in paragraph.sentences
    ] == [0, 1, 2]


def _project(db, name: str) -> Project:
    workspace = workspace_service.ensure_default_workspace(db)
    project = Project(name=name, workspace_id=workspace.id)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def test_document_creation_persists_and_activates_initial_structure(db):
    project = _project(db, "structure-create")
    text = "One sentence.\n\nTwo. Three."

    document = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project.id, text=text, source="manual"),
    )

    assert document.active_structure_version_id is not None
    structure = db.get(DocumentStructureVersion, document.active_structure_version_id)
    assert structure is not None
    assert structure.version == 1
    assert structure.text_length == len(text)
    assert db.query(DocumentSection).filter_by(structure_version_id=structure.id).count() == 1
    assert db.query(DocumentParagraph).filter_by(structure_version_id=structure.id).count() == 2
    sentences = (
        db.query(DocumentSentence)
        .filter_by(structure_version_id=structure.id)
        .order_by(DocumentSentence.ordinal)
        .all()
    )
    assert [text[item.start_offset : item.end_offset] for item in sentences] == [
        "One sentence.",
        "Two.",
        "Three.",
    ]
    assert document.sentences == [
        [item.start_offset, item.end_offset] for item in sentences
    ]

    structure.status = "changed"
    with pytest.raises(ImmutableDocumentStructureError, match="immutable"):
        db.commit()
    db.rollback()


def test_jats_import_metadata_preserves_section_paths_and_locators(db):
    xml = """<pmc-articleset><article>
      <front><article-meta><article-id pub-id-type="pmc">42</article-id></article-meta></front>
      <body>
        <sec><title>Methods</title><p>We enrolled patients.</p>
          <sec><title>Analysis</title><p>Models were fitted.</p></sec>
        </sec>
      </body>
    </article></pmc-articleset>"""
    body = parse_pmc_xml_documents(xml)["PMC42"]
    project = _project(db, "structure-jats")

    document = corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            text=body.text,
            source="pmc",
            metadata_={"structure_source": body.structure_source},
        ),
        structure_source_metadata=body.structure_source,
    )

    sections = (
        db.query(DocumentSection)
        .filter_by(structure_version_id=document.active_structure_version_id)
        .order_by(DocumentSection.ordinal)
        .all()
    )
    assert [section.path for section in sections] == [
        ["Methods"],
        ["Methods", "Analysis"],
    ]
    paragraphs = (
        db.query(DocumentParagraph)
        .filter_by(structure_version_id=document.active_structure_version_id)
        .order_by(DocumentParagraph.ordinal)
        .all()
    )
    assert paragraphs[0].locator["jats_path"] == "/body/sec[1]/title[1]"
    assert all(
        body.text[paragraph.start_offset : paragraph.end_offset]
        for paragraph in paragraphs
    )


def test_backfill_is_idempotent_and_reuses_ready_inactive_version(db):
    project = _project(db, "structure-backfill")
    document = Document(project_id=project.id, text="Legacy document.", source="legacy")
    db.add(document)
    db.commit()

    first = corpus_service.backfill_document_structures(db)
    second = corpus_service.backfill_document_structures(db)

    assert first.created == 1
    assert first.failures == {}
    assert second.skipped == 1
    assert (
        db.query(DocumentStructureVersion)
        .filter_by(document_id=document.id)
        .count()
        == 1
    )


def test_backfill_records_a_document_shaped_failure_and_keeps_going(db, monkeypatch):
    project = _project(db, "structure-backfill-partial")
    first_document = Document(project_id=project.id, text="First.", source="legacy")
    second_document = Document(project_id=project.id, text="Second.", source="legacy")
    db.add_all([first_document, second_document])
    db.commit()

    original_segment = corpus_service.segment_document

    def fail_on_first(text, *, source_metadata=None):
        if text == "First.":
            raise ValueError("unparseable legacy text")
        return original_segment(text, source_metadata=source_metadata)

    monkeypatch.setattr(corpus_service, "segment_document", fail_on_first)
    result = corpus_service.backfill_document_structures(db)

    assert result.created == 1
    assert result.failures == {first_document.id: "unparseable legacy text"}
    db.refresh(second_document)
    assert second_document.active_structure_version_id is not None


def test_backfill_aborts_instead_of_walking_the_corpus_on_a_dead_session(db, monkeypatch):
    from sqlalchemy.exc import OperationalError

    project = _project(db, "structure-backfill-abort")
    documents = [
        Document(project_id=project.id, text=f"Legacy {index}.", source="legacy")
        for index in range(3)
    ]
    db.add_all(documents)
    db.commit()

    attempted: list[str] = []

    def fail_with_dead_connection(text, *, source_metadata=None):
        attempted.append(text)
        raise OperationalError("SELECT 1", {}, Exception("server closed the connection"))

    monkeypatch.setattr(corpus_service, "segment_document", fail_with_dead_connection)
    with pytest.raises(OperationalError):
        corpus_service.backfill_document_structures(db)

    # The run stops on the first document rather than reporting the same
    # infrastructure failure once per remaining document.
    assert len(attempted) == 1


def test_structure_api_supports_ranges_rebuild_and_activation(client):
    project_response = client.post(
        "/api/projects",
        json={"name": "structure-api", "annotation_schema": {}, "settings": {}},
    )
    assert project_response.status_code == 200
    project_id = project_response.json()["id"]
    document_response = client.post(
        "/api/documents",
        json={
            "project_id": project_id,
            "text": "One.\n\nTwo. Three.",
            "source": "manual",
            "metadata_": {},
        },
    )
    assert document_response.status_code == 200
    document = document_response.json()

    first_page_response = client.get(
        f"/api/documents/{document['id']}/structure",
        params={"sentence_start": 0, "sentence_limit": 1},
    )
    assert first_page_response.status_code == 200
    first_page = first_page_response.json()
    assert first_page["range"] == {
        "start_ordinal": 0,
        "end_ordinal": 1,
        "total_sentences": 3,
        "has_more": True,
    }
    assert [sentence["text"] for sentence in first_page["sentences"]] == ["One."]
    original_version_id = first_page["structure_version"]["id"]

    rebuild_response = client.post(
        f"/api/documents/{document['id']}/structure/rebuild",
        json={"activate": False},
    )
    assert rebuild_response.status_code == 200
    rebuilt = rebuild_response.json()
    assert rebuilt["structure_version"]["version"] == 2
    rebuilt_version_id = rebuilt["structure_version"]["id"]

    still_active = client.get(f"/api/documents/{document['id']}").json()
    assert still_active["active_structure_version_id"] == original_version_id

    activate_response = client.post(
        f"/api/documents/{document['id']}/structure/{rebuilt_version_id}/activate"
    )
    assert activate_response.status_code == 200
    now_active = client.get(f"/api/documents/{document['id']}").json()
    assert now_active["active_structure_version_id"] == rebuilt_version_id

    invalid_range = client.get(
        f"/api/documents/{document['id']}/structure",
        params={"sentence_start": 99},
    )
    assert invalid_range.status_code == 422


def test_activation_is_blocked_by_open_evidence_assignment(db):
    project = _project(db, "structure-activation-guard")
    document = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project.id, text="Evidence sentence."),
    )
    user = User(
        username="structure-annotator",
        password_hash=hash_password("pw"),
        is_active=True,
    )
    task = ProjectTask(
        project_id=project.id,
        annotation_type="evidence_block",
        display_name="Evidence blocks",
    )
    db.add_all([user, task])
    db.flush()
    assignment = TaskAssignment(
        project_id=project.id,
        task_id=task.id,
        document_id=document.id,
        assignee_user_id=user.id,
        structure_version_id=document.active_structure_version_id,
        assignment_scope_key="target:test",
        annotator_id=user.username,
        status="in_progress",
    )
    db.add(assignment)
    db.commit()
    candidate = corpus_service.rebuild_document_structure(
        db,
        document.id,
        activate=False,
    )

    with pytest.raises(ConflictError, match="evidence assignments are open"):
        corpus_service.activate_document_structure(db, document.id, candidate.id)

    db.rollback()
    assignment.status = "submitted"
    db.commit()
    corpus_service.activate_document_structure(db, document.id, candidate.id)
    db.refresh(document)
    assert document.active_structure_version_id == candidate.id
