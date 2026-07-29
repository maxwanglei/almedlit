"""NCBI E-utilities client and PubMed/PMC XML parsing for the importer.

Network access is isolated behind an injectable ``httpx.Client`` so the parsing
and orchestration logic can be unit-tested with ``httpx.MockTransport`` and no
live NCBI calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import httpx
from defusedxml.ElementTree import fromstring as _safe_fromstring

from al_medlit.core.config import settings


class ImporterFetchError(Exception):
    """Raised when an NCBI request fails (network, timeout, or bad status)."""


@dataclass
class PubMedMeta:
    pmid: str
    title: str = ""
    journal: str = ""
    year: str = ""
    abstract: str = ""


@dataclass(frozen=True, slots=True)
class JATSBody:
    """Canonical full text plus locators used by document segmentation."""

    text: str
    structure_source: dict


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _parse_abstract(article: ET.Element) -> str:
    parts: list[str] = []
    for node in article.findall("./Abstract/AbstractText"):
        text = _text(node)
        if not text:
            continue
        label = node.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return "\n\n".join(parts)


def parse_pubmed_xml(xml: str) -> dict[str, PubMedMeta]:
    """Parse an efetch ``db=pubmed`` PubmedArticleSet into PMID -> metadata."""
    root = _safe_fromstring(xml)
    out: dict[str, PubMedMeta] = {}
    for citation in root.findall(".//MedlineCitation"):
        pmid = _text(citation.find("./PMID"))
        if not pmid:
            continue
        article = citation.find("./Article")
        if article is None:
            out[pmid] = PubMedMeta(pmid=pmid)
            continue
        out[pmid] = PubMedMeta(
            pmid=pmid,
            title=_text(article.find("./ArticleTitle")),
            journal=_text(article.find("./Journal/Title")),
            year=_text(article.find("./Journal/JournalIssue/PubDate/Year")),
            abstract=_parse_abstract(article),
        )
    return out


def _normalize_pmcid(raw: str) -> str:
    digits = "".join(ch for ch in raw if ch.isdigit())
    return f"PMC{digits}" if digits else ""


PMC_ID_TYPES = {"pmc", "pmcid", "pmcaid", "pmcaiid"}


def _article_pmcid(article: ET.Element) -> str:
    for node in article.findall(".//article-id"):
        if node.get("pub-id-type") not in PMC_ID_TYPES:
            continue
        pmcid = _normalize_pmcid(_text(node))
        if pmcid:
            return pmcid
    return ""


def _local_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _element_locator(
    element: ET.Element,
    body: ET.Element,
    parents: dict[ET.Element, ET.Element],
) -> str:
    parts: list[str] = []
    current = element
    while current is not body:
        parent = parents[current]
        tag = _local_tag(current)
        siblings = [child for child in parent if _local_tag(child) == tag]
        parts.append(f"{tag}[{siblings.index(current) + 1}]")
        current = parent
    return "/body/" + "/".join(reversed(parts))


def _section_path(
    element: ET.Element,
    body: ET.Element,
    parents: dict[ET.Element, ET.Element],
) -> list[str]:
    sections: list[ET.Element] = []
    current = parents.get(element)
    while current is not None and current is not body:
        if _local_tag(current) == "sec":
            sections.append(current)
        current = parents.get(current)

    path: list[str] = []
    for section in reversed(sections):
        title = next(
            (child for child in section if _local_tag(child) == "title"),
            None,
        )
        title_text = _text(title)
        if title_text:
            path.append(title_text)
    return path


def extract_jats_body(article: ET.Element) -> JATSBody | None:
    """Extract canonical text and deterministic JATS block locators."""
    body = article.find("./body")
    if body is None:
        return None

    parents = {child: parent for parent in body.iter() for child in parent}
    raw_blocks: list[dict] = []
    block_texts: list[str] = []
    for node in body.iter():
        tag = _local_tag(node)
        if tag not in {"title", "p"}:
            continue
        text = _text(node)
        if not text:
            continue
        section_path = _section_path(node, body, parents)
        block_texts.append(text)
        raw_blocks.append(
            {
                "section_path": section_path,
                "section_title": section_path[-1] if section_path else None,
                "section_kind": "jats",
                "locator": {
                    "jats_path": _element_locator(node, body, parents),
                    "block_kind": "section_title" if tag == "title" else "paragraph",
                },
            }
        )

    if not block_texts:
        return None

    canonical_text = "\n\n".join(block_texts)
    blocks: list[dict] = []
    offset = 0
    for index, (text, raw) in enumerate(zip(block_texts, raw_blocks, strict=True)):
        start = offset
        end = start + len(text)
        blocks.append(
            {
                **raw,
                "start_offset": start,
                "end_offset": end,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
        offset = end + (2 if index < len(block_texts) - 1 else 0)

    return JATSBody(
        text=canonical_text,
        structure_source={"format": "jats-v1", "blocks": blocks},
    )


def extract_jats_body_text(article: ET.Element) -> str | None:
    """Extract plain text from a JATS article ``<body>``.

    Section titles and paragraph text are concatenated; tables and figures are
    reduced to their captions. Returns ``None`` when the article has no body.
    """
    extracted = extract_jats_body(article)
    return extracted.text if extracted is not None else None


def parse_pmc_xml_documents(xml: str) -> dict[str, JATSBody]:
    """Parse PMC XML into canonical text documents with JATS locators."""
    root = _safe_fromstring(xml)
    out: dict[str, JATSBody] = {}
    for article in root.findall(".//article"):
        pmcid = _article_pmcid(article)
        if not pmcid:
            continue
        body = extract_jats_body(article)
        if body is not None:
            out[pmcid] = body
    return out


def parse_pmc_xml(xml: str) -> dict[str, str]:
    """Parse an efetch ``db=pmc`` article set into PMCID -> body text.

    Only articles that expose a retrievable ``<body>`` (the open-access
    full-text signal) appear in the result.
    """
    return {pmcid: body.text for pmcid, body in parse_pmc_xml_documents(xml).items()}


def parse_idconv_json(data: dict) -> dict[str, str]:
    """Map PMID -> PMCID from a PMC ID Converter response, skipping records
    that have no PMCID (errors or PubMed-only articles)."""
    out: dict[str, str] = {}
    for record in data.get("records", []):
        pmid = record.get("pmid")
        pmcid = record.get("pmcid")
        if pmid and pmcid:
            out[str(pmid)] = pmcid
    return out


# --- HTTP layer (NCBI E-utilities) ------------------------------------------


def _auth_params() -> dict[str, str]:
    params = {"tool": settings.ncbi_tool}
    if settings.ncbi_email:
        params["email"] = settings.ncbi_email
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    return params


def _get(client: httpx.Client, url: str, params: dict[str, str]) -> httpx.Response:
    try:
        response = client.get(url, params=params, timeout=settings.ncbi_request_timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:  # network, timeout, or non-2xx status
        raise ImporterFetchError(f"NCBI request failed: {exc}") from exc
    return response


def fetch_pubmed_metadata(
    client: httpx.Client, pmids: list[str]
) -> dict[str, PubMedMeta]:
    """Batched efetch ``db=pubmed`` -> PMID -> metadata."""
    if not pmids:
        return {}
    params = {**_auth_params(), "db": "pubmed", "retmode": "xml", "id": ",".join(pmids)}
    response = _get(client, f"{settings.ncbi_eutils_base}/efetch.fcgi", params)
    return parse_pubmed_xml(response.text)


def resolve_pmcids(client: httpx.Client, pmids: list[str]) -> dict[str, str]:
    """Batched PMC ID Converter -> PMID -> PMCID (only those with a PMCID)."""
    if not pmids:
        return {}
    params = {**_auth_params(), "format": "json", "idtype": "pmid", "ids": ",".join(pmids)}
    response = _get(client, settings.ncbi_idconv_base, params)
    return parse_idconv_json(response.json())


def fetch_pmc_fulltext(client: httpx.Client, pmcids: list[str]) -> dict[str, str]:
    """Batched efetch ``db=pmc`` -> PMCID -> body text (open-access only)."""
    if not pmcids:
        return {}
    ids = ",".join(pmcid.removeprefix("PMC") for pmcid in pmcids)
    params = {**_auth_params(), "db": "pmc", "retmode": "xml", "id": ids}
    response = _get(client, f"{settings.ncbi_eutils_base}/efetch.fcgi", params)
    return parse_pmc_xml(response.text)


def fetch_pmc_documents(
    client: httpx.Client,
    pmcids: list[str],
) -> dict[str, JATSBody]:
    """Batched PMC fetch retaining canonical JATS section/paragraph locators."""
    if not pmcids:
        return {}
    ids = ",".join(pmcid.removeprefix("PMC") for pmcid in pmcids)
    params = {**_auth_params(), "db": "pmc", "retmode": "xml", "id": ids}
    response = _get(client, f"{settings.ncbi_eutils_base}/efetch.fcgi", params)
    return parse_pmc_xml_documents(response.text)
