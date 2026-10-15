import hashlib

import pytest

from al_medlit.importers.pubmed import parse_pmc_xml_documents, parse_pubmed_xml


def _extract(body):
    xml = (
        "<pmc-articleset><article><front><article-meta>"
        '<article-id pub-id-type="pmc">42</article-id>'
        "</article-meta></front><body>" + body + "</body></article></pmc-articleset>"
    )
    return parse_pmc_xml_documents(xml)["PMC42"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "<p><list><list-item><p>First.</p></list-item>"
            "<list-item><p>Second.</p></list-item></list></p>",
            ["First.", "Second."],
        ),
        (
            "<p><fig><caption><title>Figure title</title>"
            "<p>Figure caption.</p></caption></fig></p>",
            ["Figure title", "Figure caption."],
        ),
        (
            "<p><table-wrap><caption><title>Table title</title>"
            "<p>Table caption.</p></caption></table-wrap></p>",
            ["Table title", "Table caption."],
        ),
        (
            "<p>Intro.<list><list-item><p>First.</p></list-item>"
            "<list-item><p>Second.</p></list-item></list>Tail.</p>",
            ["Intro.", "First.", "Second.", "Tail."],
        ),
        (
            "<p>Before.<boxed-text><p>Inside.<list><list-item><p>Nested.</p>"
            "</list-item></list>After inside.</p></boxed-text>After.</p>",
            ["Before.", "Inside.", "Nested.", "After inside.", "After."],
        ),
    ],
    ids=["list", "figure-caption", "table-caption", "surrounding-prose", "deep-nesting"],
)
def test_nested_jats_blocks_are_emitted_once_in_source_order(body, expected):
    document = _extract(body)

    assert document.text == "\n\n".join(expected)
    blocks = document.structure_source["blocks"]
    assert len(blocks) == len(expected)
    for block, text in zip(blocks, expected, strict=True):
        assert document.text[block["start_offset"] : block["end_offset"]] == text
        assert block["text_hash"] == hashlib.sha256(text.encode()).hexdigest()
    assert _extract(body) == document


@pytest.mark.parametrize(
    ("paragraph", "expected"),
    [
        ("<p>Alpha<break/>Beta.</p>", "Alpha Beta."),
        (
            "<p><list><list-item>Alpha</list-item><list-item>Beta.</list-item></list></p>",
            "Alpha Beta.",
        ),
        (
            "<p>Before<boxed-text>Inside</boxed-text>After.</p>",
            "Before Inside After.",
        ),
        (
            "<p>Before<disp-formula><math>x=1</math></disp-formula>After.</p>",
            "Before x=1 After.",
        ),
        (
            "<p><def-list><def-item><term>Alpha</term><def>First.</def></def-item>"
            "<def-item><term>Beta</term><def>Second.</def></def-item></def-list></p>",
            "Alpha First. Beta Second.",
        ),
        (
            "<p>TNF-<italic>α</italic>, CO<sub>2</sub>, 10<sup>3</sup> cells.</p>",
            "TNF-α, CO2, 103 cells.",
        ),
        (
            "<p>We found <bold>strong</bold> evidence (<xref>1</xref>).</p>",
            "We found strong evidence (1).",
        ),
    ],
    ids=[
        "line-break",
        "list-items",
        "boxed-text",
        "display-formula",
        "definition-list",
        "scientific-tokens",
        "inline-punctuation",
    ],
)
def test_jats_text_separates_blocks_and_preserves_inline_content(paragraph, expected):
    document = _extract(paragraph)

    assert document.text == expected
    assert document.structure_source["blocks"][0]["locator"]["jats_path"] == "/body/p[1]"


def test_pubmed_abstract_separates_nested_paragraphs():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article>
        <ArticleTitle>CO<sub>2</sub> study</ArticleTitle>
        <Abstract><AbstractText><p>First.</p><p>Second.</p></AbstractText></Abstract>
        </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""

    metadata = parse_pubmed_xml(xml)["42"]

    assert metadata.abstract == "First. Second."
    assert metadata.title == "CO2 study"
