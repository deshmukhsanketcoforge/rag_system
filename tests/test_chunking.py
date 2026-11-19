"""Chunking tests: size cap, overlap, metadata, no empty chunks. No model or API needed."""

from collections import Counter
from pathlib import Path

from rag.chunking import HEADER_SECTION, chunk_corpus, chunk_document, split_body
from rag.config import CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS, RAW_DIR

SHORT_DOC = (
    """# Demo Policy

Effective date: 2024-01-01
Owner: Tests

## 1. Purpose

Short section.

## 2. Rules

### 2.1 First

"""
    + " ".join(f"a{i}" for i in range(80))
    + """

### 2.2 Second

"""
    + " ".join(f"b{i}" for i in range(80))
    + """
"""
)


def _body(chunk):
    """Chunk text without the 'title - section' header line."""
    return chunk.text.split("\n\n", 1)[1]


def test_corpus_chunks_every_document():
    chunks = chunk_corpus(RAW_DIR)
    docs = {c.doc_name for c in chunks}
    assert docs == {p.stem for p in RAW_DIR.glob("*.md")}
    assert len(chunks) > len(docs)


def test_no_empty_chunks_and_metadata_populated():
    for c in chunk_corpus(RAW_DIR):
        assert c.text.strip()
        assert c.doc_name and c.section
        assert c.metadata == {
            "doc_name": c.doc_name,
            "section": c.section,
            "chunk_index": c.chunk_index,
        }


def test_ids_are_deterministic_unique_and_sequential():
    first = chunk_corpus(RAW_DIR)
    assert [c.id for c in first] == [c.id for c in chunk_corpus(RAW_DIR)]
    assert len({c.id for c in first}) == len(first)
    per_doc = Counter(c.doc_name for c in first)
    for c in first:
        assert c.id == f"{c.doc_name}::{c.chunk_index}"
        assert 0 <= c.chunk_index < per_doc[c.doc_name]


def test_chunks_respect_max_size():
    for c in chunk_corpus(RAW_DIR):
        if c.section != HEADER_SECTION:
            assert len(_body(c).split()) <= CHUNK_MAX_WORDS, c.id


def test_every_section_heading_is_covered():
    text = (RAW_DIR / "pto_policy.md").read_text()
    headings = {line[3:].strip() for line in text.splitlines() if line.startswith("## ")}
    sections = {c.section for c in chunk_corpus(RAW_DIR) if c.doc_name == "pto_policy"}
    assert headings <= sections


def test_section_that_fits_stays_whole():
    chunks = chunk_document("demo", SHORT_DOC, max_words=100, overlap=10)
    purpose = [c for c in chunks if c.section == "1. Purpose"]
    assert len(purpose) == 1 and "Short section." in purpose[0].text


def test_long_section_splits_on_subsection_boundaries():
    chunks = chunk_document("demo", SHORT_DOC, max_words=100, overlap=10)
    rules = [c for c in chunks if c.section == "2. Rules"]
    assert len(rules) == 2
    assert "### 2.1 First" in rules[0].text and "### 2.2 Second" in rules[1].text
    assert "a0" not in rules[1].text  # no cut inside a subsection


def test_oversized_block_is_windowed_with_overlap():
    long_doc = "# Long\n\n## 1. Big\n\n" + " ".join(f"w{i}" for i in range(250))
    chunks = chunk_document("long", long_doc, max_words=100, overlap=20)
    pieces = [_body(c).split() for c in chunks]
    assert len(pieces) == 3
    assert all(len(p) <= 100 for p in pieces)
    for prev, nxt in zip(pieces, pieces[1:], strict=False):
        assert prev[-20:] == nxt[:20]
    assert {w for p in pieces for w in p} == {f"w{i}" for i in range(250)}  # nothing lost


def test_default_overlap_is_a_sane_fraction_of_the_cap():
    assert 0.1 <= CHUNK_OVERLAP_WORDS / CHUNK_MAX_WORDS <= 0.25


def test_each_document_has_a_header_chunk_with_date_and_owner():
    chunks = chunk_corpus(RAW_DIR)
    for doc in {c.doc_name for c in chunks}:
        first = next(c for c in chunks if c.doc_name == doc and c.chunk_index == 0)
        assert first.section == HEADER_SECTION, doc
        assert "Effective date" in first.text and "Owner" in first.text, doc


# --- review fixes ---


def test_long_preamble_is_split_and_respects_the_cap():
    preamble = " ".join(f"p{i}" for i in range(250))
    doc = f"# Long Intro\n\n{preamble}\n\n## 1. Purpose\n\nShort."
    chunks = chunk_document("intro", doc, max_words=100, overlap=20)
    header = [c for c in chunks if c.section == HEADER_SECTION]
    assert len(header) == 3
    assert all(len(_body(c).split()) <= 100 for c in header)
    assert {w for c in header for w in _body(c).split()} == {f"p{i}" for i in range(250)}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_short_preamble_stays_one_header_chunk():
    chunks = chunk_document("d", "# T\n\nEffective date: 2024\nOwner: X\n\n## 1. A\n\nBody.")
    assert [c.section for c in chunks] == [HEADER_SECTION, "1. A"]


def test_files_are_read_as_utf8_whatever_the_locale(tmp_path, monkeypatch):
    (tmp_path / "d.md").write_text("# T\n\n## 1. A\n\nIt\u2019s a \u2014 test.", encoding="utf-8")
    real = Path.read_text

    def strict(self, encoding=None, errors=None):
        assert encoding == "utf-8", "read_text must pass encoding='utf-8'"
        return real(self, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", strict)
    (chunk,) = [c for c in chunk_corpus(tmp_path) if c.section == "1. A"]
    assert "It\u2019s a \u2014 test." in chunk.text


FENCED = (
    "# Code Doc\n\n## 1. Example\n\nRun this:\n\n```\n## not a heading\n### nor this\n```\n\n"
    "Done.\n\n## 2. Next\n\nText."
)


def test_headings_inside_code_fences_do_not_split_sections():
    chunks = chunk_document("c", FENCED)
    assert [c.section for c in chunks] == ["1. Example", "2. Next"]
    assert "## not a heading" in chunks[0].text and "Done." in chunks[0].text


def test_subsection_markers_inside_fences_are_not_split_points():
    def words(tag, n):
        return " ".join(f"{tag}{i}" for i in range(n))

    # The fenced "### " line sits just past the 100-word mark. If it were treated as a heading,
    # a piece would start there.
    body = (
        f"### A\n{words('a', 90)}\n```\n### not a heading\n{words('f', 10)}\n```\n"
        f"### B\n{words('b', 40)}"
    )
    pieces = split_body(body, 100, 20)
    assert not any(p.startswith("### not a heading") for p in pieces)
    assert sum("### not a heading" in p for p in pieces) >= 1
    assert any(p.startswith("### B") for p in pieces)


def test_title_keeps_hash_characters_that_are_content():
    chunks = chunk_document("d", "# #1 Priorities\n\n## 1. A\n\nBody.")
    assert chunks[0].text.startswith("#1 Priorities - 1. A")


def test_document_without_a_title_block_still_chunks():
    chunks = chunk_document("bare", "## 1. A\n\nBody one.\n\n## 2. B\n\nBody two.")
    assert [c.section for c in chunks] == ["1. A", "2. B"]
    assert chunks[0].text.startswith("bare - 1. A")


def test_empty_document_gives_no_chunks():
    assert chunk_document("empty", "  \n") == []
