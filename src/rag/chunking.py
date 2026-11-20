"""Section-aware chunking of the markdown policy documents.

Strategy (see the rationale next to CHUNK_MAX_WORDS in config.py):
1. Split each document on '## ' headings. A section that fits in CHUNK_MAX_WORDS is one chunk.
2. A longer section is packed greedily from its '### ' subsections, never cutting inside one.
3. Only a single block that is still too long is cut into word windows with
   CHUNK_OVERLAP_WORDS of overlap, so a fact on a boundary appears in both neighbours.

Every chunk starts with a header line ("<doc title> - <section>") so the text is understandable
and keyword-searchable on its own. Metadata carries doc_name, section and chunk_index; the doc
title is the same for both PTO documents on purpose, nothing here marks one as newer.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from rag.config import CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS

HEADER_SECTION = "Document header"

_FENCE = re.compile(r"^\s*(```|~~~)")
_TITLE = re.compile(r"^#\s+")


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_name>::<chunk_index>", deterministic so re-ingest overwrites, never duplicates
    text: str
    doc_name: str
    section: str
    chunk_index: int

    @property
    def metadata(self) -> dict:
        return {"doc_name": self.doc_name, "section": self.section, "chunk_index": self.chunk_index}


def _words(text: str) -> int:
    return len(text.split())


def _split_before(text: str, prefix: str) -> list[str]:
    """Split text into parts that each start at a line beginning with `prefix`.

    Lines inside fenced code blocks (``` or ~~~) never count as headings, so an example that
    contains "## ..." is not mistaken for a section. Text before the first match is its own part.
    """
    parts: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            in_fence = not in_fence
        if not in_fence and line.startswith(prefix) and current:
            parts.append("".join(current))
            current = []
        current.append(line)
    if current:
        parts.append("".join(current))
    return parts


def _window(text: str, max_words: int, overlap: int) -> list[str]:
    """Cut text into windows of at most max_words that share `overlap` words."""
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    words = text.split()
    step = max_words - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return windows


def split_body(body: str, max_words: int, overlap: int) -> list[str]:
    """Split one section body into pieces of at most max_words."""
    body = body.strip()
    if _words(body) <= max_words:
        return [body]

    blocks = [b.strip() for b in _split_before(body, "### ") if b.strip()]
    pieces: list[str] = []
    current = ""
    for block in blocks:
        if _words(block) > max_words:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_window(block, max_words, overlap))
        elif current and _words(current) + _words(block) > max_words:
            pieces.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    doc_name: str,
    text: str,
    max_words: int = CHUNK_MAX_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Chunk one markdown document. doc_name is the file stem, e.g. 'pto_policy'."""
    parts = _split_before(text.strip(), "## ")
    if not parts:
        return []
    # A document that opens straight with a '## ' section has no title block.
    head, sections = ("", parts) if parts[0].startswith("## ") else (parts[0].strip(), parts[1:])
    title = _TITLE.sub("", head.splitlines()[0]).strip() if head else doc_name
    preamble = "\n".join(head.splitlines()[1:]).strip()

    raw: list[tuple[str, str]] = []  # (section, text)
    # The preamble (title block: effective date, owner, intro) obeys the same size cap.
    for piece in split_body(preamble, max_words, overlap) if preamble else []:
        raw.append((HEADER_SECTION, f"{title}\n\n{piece}"))
    for part in sections:
        heading, _, body = part.partition("\n")
        section = heading.removeprefix("## ").strip()
        for piece in split_body(body, max_words, overlap):
            if piece:
                raw.append((section, f"{title} - {section}\n\n{piece}"))

    return [
        Chunk(f"{doc_name}::{i}", chunk_text, doc_name, section, i)
        for i, (section, chunk_text) in enumerate(raw)
    ]


def chunk_corpus(raw_dir: Path) -> list[Chunk]:
    """Chunk every .md file in raw_dir, in a stable (sorted) order."""
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.glob("*.md")):
        chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8")))
    return chunks
