"""Orchestration for the PubMed/PMC importer: preview and import.

Two-phase and stateless: ``preview_import`` classifies PMIDs without touching
the database; ``run_import`` re-fetches and persists the user's selection.
"""

from __future__ import annotations

import httpx
from sqlalchemy.orm import Session

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
from al_medlit.project.models import Project


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


def preview_import(client: httpx.Client, pmids: list[str]) -> list[ImportPreviewItem]:
    normalized = normalize_pmids(pmids)
    metadata, pmcids, fulltext = _gather(client, normalized)

    items: list[ImportPreviewItem] = []
    for pmid in normalized:
        meta = metadata.get(pmid)
        if meta is None:
            items.append(ImportPreviewItem(pmid=pmid, status="not_found"))
            continue
        pmcid = pmcids.get(pmid)
        has_full_text = bool(pmcid and pmcid in fulltext)
        has_abstract = bool(meta.abstract)
        status = "full_text" if has_full_text else "abstract_only"
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
    client: httpx.Client,
    project_id: int,
    pmids: list[str],
    include_abstract_only: bool,
) -> ImportResponse:
    normalized = normalize_pmids(pmids)
    metadata, pmcids, fulltext = _gather(client, normalized)

    existing = {
        external_id
        for (external_id,) in db.query(Document.external_id)
        .filter(Document.project_id == project_id)
        .all()
        if external_id is not None
    }

    response = ImportResponse()
    for pmid in normalized:
        meta = metadata.get(pmid)
        if meta is None:
            response.skipped.append(
                ImportOutcome(pmid=pmid, status="not_found", reason="No PubMed record")
            )
            continue

        if pmid in existing:
            response.skipped.append(
                ImportOutcome(
                    pmid=pmid,
                    status="full_text" if (pmcids.get(pmid) in fulltext) else "abstract_only",
                    title=meta.title,
                    reason="Duplicate: already imported in this project",
                )
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
            existing.add(pmid)
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
            existing.add(pmid)
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
    )


def ensure_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        from al_medlit.core.exceptions import NotFoundError

        raise NotFoundError(f"Project {project_id} not found")
    return project
