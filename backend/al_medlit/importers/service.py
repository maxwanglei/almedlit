"""Orchestration for the PubMed/PMC importer: preview and import.

Preview classifies PMIDs without writes; import persists the user's selection.
Previously imported papers can be reused without fetching them again.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_project_member, lock_project_member_for_mutation
from al_medlit.core.exceptions import UnauthorizedError
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import Document
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.importers import pubmed
from al_medlit.importers.pubmed import JATSBody, PubMedMeta
from al_medlit.importers.schemas import (
    ImportOutcome,
    ImportPreviewItem,
    ImportResponse,
)
from al_medlit.project import service as project_service


def normalize_pmids(pmids: list[str]) -> list[str]:
    """Strip whitespace and ``PMID:`` prefixes, keep digit-only tokens, dedupe."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in pmids:
        token = raw.strip().removeprefix("PMID:").removeprefix("pmid:").strip()
        token = "".join(ch for ch in token if ch.isdigit())
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _gather(
    client: httpx.Client, pmids: list[str]
) -> tuple[dict[str, PubMedMeta], dict[str, str], dict[str, JATSBody]]:
    """Fetch metadata, PMID->PMCID, and PMCID->body text for the PMIDs."""
    metadata = pubmed.fetch_pubmed_metadata(client, pmids)
    pmcids = pubmed.resolve_pmcids(client, pmids)
    fulltext = pubmed.fetch_pmc_documents(client, list(pmcids.values()))
    return metadata, pmcids, fulltext


@dataclass(frozen=True)
class ImportActor:
    user_id: int
    session_version: int


@dataclass(frozen=True)
class ProjectPaper:
    id: int
    title: str
    source: str | None
    journal: str
    year: str
    pmcid: str | None


@dataclass(frozen=True)
class FetchedImport:
    pmids: list[str]
    metadata: dict[str, PubMedMeta]
    pmcids: dict[str, str]
    fulltext: dict[str, JATSBody]


def project_documents_by_pmid(
    db: Session, project_id: int, pmids: list[str],
) -> dict[str, ProjectPaper]:
    if not pmids:
        return {}
    rows = db.execute(
        select(
            Document.external_id,
            Document.id,
            Document.title,
            Document.source,
            Document.metadata_["journal"].as_string(),
            Document.metadata_["year"].as_string(),
            Document.metadata_["pmcid"].as_string(),
        )
        .where(Document.project_id == project_id, Document.external_id.in_(pmids))
        .order_by(Document.id)
    )
    return {
        pmid: ProjectPaper(
            id=document_id, title=title or "", source=source,
            journal=str(journal or ""), year=str(year or ""), pmcid=pmcid,
        )
        for pmid, document_id, title, source, journal, year, pmcid in rows
    }


def fetch_import(
    client: httpx.Client, pmids: list[str], *, existing: dict[str, ProjectPaper],
) -> FetchedImport:
    normalized = normalize_pmids(pmids)
    missing = [pmid for pmid in normalized if pmid not in existing]
    metadata, pmcids, fulltext = _gather(client, missing) if missing else ({}, {}, {})
    return FetchedImport(normalized, metadata, pmcids, fulltext)


def revalidate_import_actor(db: Session, project_id: int, actor: ImportActor) -> User:
    user = db.query(User).filter(User.id == actor.user_id).populate_existing().first()
    if user is None or not user.is_active or user.session_version != actor.session_version:
        raise UnauthorizedError("Invalid or expired token")
    assert_project_member(db, user, project_id, min_role="manager")
    return user


def _lock_import(db: Session, project_id: int) -> None:
    # This namespace serializes import deduplication without locking annotation policy.
    if db.get_bind().dialect.name == "postgresql":
        db.execute(
            text("SELECT pg_advisory_xact_lock(:namespace, :project_id)"),
            {"namespace": 0x414C494D, "project_id": project_id},
        )


def preview_import(
    client: httpx.Client, pmids: list[str], *, existing: dict[str, ProjectPaper] | None = None,
) -> list[ImportPreviewItem]:
    normalized = normalize_pmids(pmids)
    existing = existing or {}
    missing = [pmid for pmid in normalized if pmid not in existing]
    metadata, pmcids, fulltext = _gather(client, missing) if missing else ({}, {}, {})

    items: list[ImportPreviewItem] = []
    for pmid in normalized:
        if pmid in existing:
            document = existing[pmid]
            full_text = document.source == "pmc"
            items.append(ImportPreviewItem(
                pmid=pmid, title=document.title or "",
                status="full_text" if full_text else "abstract_only",
                has_full_text=full_text, has_abstract=not full_text,
                journal=document.journal,
                year=document.year,
                pmcid=document.pmcid,
            ))
            continue
        meta = metadata.get(pmid)
        if meta is None:
            items.append(ImportPreviewItem(pmid=pmid, status="not_found"))
            continue
        pmcid = pmcids.get(pmid)
        has_full_text = bool(pmcid and pmcid in fulltext)
        has_abstract = bool(meta.abstract)
        status = "full_text" if has_full_text else "abstract_only" if has_abstract else "error"
        items.append(
            ImportPreviewItem(
                pmid=pmid,
                title=meta.title,
                journal=meta.journal,
                year=meta.year,
                pmcid=pmcid,
                status=status,
                has_full_text=has_full_text,
                has_abstract=has_abstract,
            )
        )
    return items


def run_import(
    db: Session,
    project_id: int,
    fetched: FetchedImport,
    include_abstract_only: bool,
    *,
    actor: ImportActor,
) -> ImportResponse:
    """Persist fetched papers and personal assignments in the caller's transaction."""
    user = revalidate_import_actor(db, project_id, actor)
    lock_project_member_for_mutation(db, user, project_id, min_role="manager")
    if user.session_version != actor.session_version:
        raise UnauthorizedError("Invalid or expired token")
    _lock_import(db, project_id)
    existing = project_documents_by_pmid(db, project_id, fetched.pmids)
    normalized = fetched.pmids
    metadata, pmcids, fulltext = fetched.metadata, fetched.pmcids, fetched.fulltext

    response = ImportResponse()
    for pmid in normalized:
        if pmid in existing:
            document = existing[pmid]
            response.skipped.append(
                ImportOutcome(
                    pmid=pmid,
                    status="full_text" if document.source == "pmc" else "abstract_only",
                    title=document.title or "",
                    document_id=document.id,
                    reason="Duplicate: already in this project; available to reuse",
                )
            )
            continue
        meta = metadata.get(pmid)
        if meta is None:
            response.skipped.append(
                ImportOutcome(pmid=pmid, status="not_found", reason="No PubMed record")
            )
            continue

        pmcid = pmcids.get(pmid)
        body = fulltext.get(pmcid) if pmcid else None

        if body:
            doc = _create_document(
                db,
                project_id,
                pmid,
                title=meta.title,
                text=body.text,
                source="pmc",
                metadata={
                    "has_full_text": True,
                    "pmcid": pmcid,
                    "journal": meta.journal,
                    "year": meta.year,
                    "structure_source": body.structure_source,
                },
                structure_source_metadata=body.structure_source,
            )
            response.created.append(
                ImportOutcome(
                    pmid=pmid,
                    status="full_text",
                    title=meta.title,
                    document_id=doc.id,
                )
            )
            continue

        if meta.abstract and include_abstract_only:
            doc = _create_document(
                db,
                project_id,
                pmid,
                title=meta.title,
                text=meta.abstract,
                source="pubmed_abstract",
                metadata={
                    "has_full_text": False,
                    "journal": meta.journal,
                    "year": meta.year,
                },
            )
            response.created.append(
                ImportOutcome(
                    pmid=pmid,
                    status="abstract_only",
                    title=meta.title,
                    document_id=doc.id,
                )
            )
            continue

        reason = (
            "Abstract-only, not selected for import"
            if meta.abstract
            else "No full text or abstract available"
        )
        response.skipped.append(
            ImportOutcome(
                pmid=pmid, status="abstract_only", title=meta.title, reason=reason
            )
        )

    project_service.ensure_personal_project_self_assignments(
        db,
        project_id,
        user,
        document_ids=[
            item.document_id for item in response.created if item.document_id is not None
        ],
        assigned_by_user=user,
        commit=False,
    )
    return response


def _create_document(
    db: Session,
    project_id: int,
    pmid: str,
    *,
    title: str,
    text: str,
    source: str,
    metadata: dict,
    structure_source_metadata: dict | None = None,
) -> Document:
    return corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=project_id,
            external_id=pmid,
            title=title or None,
            text=text,
            source=source,
            metadata_=metadata,
        ),
        structure_source_metadata=structure_source_metadata,
        commit=False,
    )
