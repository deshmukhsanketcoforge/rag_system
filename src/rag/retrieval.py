"""Keyword search, Reciprocal Rank Fusion and hybrid search (RAG-6).

Vector search finds chunks that *mean* the same as the question, but it can rank a chunk low when
the question hinges on an exact token it barely encodes: a form code (EX-14), a section number
(7.3), a rare name. Keyword search is the opposite: exact tokens, no sense of meaning. Hybrid
search runs both and merges the two ranked lists with Reciprocal Rank Fusion (RRF), which only
uses ranks, so the cosine distances and keyword scores never need to be put on one scale.

Keyword scoring (all matching is case-insensitive):
- the question is split into tokens and stopwords are dropped;
- each distinct token found in a chunk scores 1; a "code" token (EX-14, 7.3) scores CODE_WEIGHT;
- each pair of words that sit next to each other in the question (a stopword between them
  breaks the pair) and are found next to each other in the chunk scores PHRASE_WEIGHT, so
  "meal per-diem" prefers the chunk that has the phrase;
- a word matches from its start, never from the middle ("pto" is not in "laptop"), in single
  words and in phrases alike. "reimburse" matches "reimbursed", the cheap stand-in for stemming.
  A code must match whole: "EX-14" is not in "EX-145", and "7.3" is not in "7.3.1".
"""

import re
from collections.abc import Callable

from rag import store
from rag.config import CANDIDATE_POOL, RRF_K

CODE_WEIGHT = 3
PHRASE_WEIGHT = 2

# Words that carry no topic. Deliberately short: a long list starts to remove real words.
STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it its me my
    of on or our so than that the their them then there these they this to was we were what when
    where which who why will with would you your""".split()
)

_TOKEN = re.compile(r"[a-z]+-\d+|\d+(?:\.\d+)+|[a-z0-9]+")
_CODE = re.compile(r"[a-z]+-\d+|\d+(?:\.\d+)+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens without stopwords. Form codes and section numbers stay in one piece."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _is_code(token: str) -> bool:
    return _CODE.fullmatch(token) is not None


def _squash(text: str) -> str:
    """Lowercase with every run of punctuation as one space, for phrase matching."""
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def _found(token: str, text: str) -> bool:
    start = r"(?<![a-z0-9])" + re.escape(token)
    return re.search(start + (r"(?![a-z0-9]|\.\d)" if _is_code(token) else ""), text) is not None


def _phrase_found(phrase: tuple[str, str], squashed: str) -> bool:
    """Both words, in order, next to each other; each matches from its start like a single word."""
    words = [w for part in phrase for w in _squash(part).split()]
    pattern = r"(?<![a-z0-9])" + r"[a-z0-9]* ".join(re.escape(w) for w in words)
    return re.search(pattern, squashed) is not None


def _score(tokens: list[str], phrases: list[tuple], text: str, squashed: str) -> int:
    score = sum(CODE_WEIGHT if _is_code(t) else 1 for t in tokens if _found(t, text))
    return score + PHRASE_WEIGHT * sum(1 for p in phrases if _phrase_found(p, squashed))


def keyword_search(query: str, chunks: list[dict], k: int) -> list[dict]:
    """Top-k chunks by keyword score, best first; chunks with no match are left out.

    Ties keep the order of `chunks`. Returns copies with a `score` added.
    """
    tokens = list(dict.fromkeys(tokenize(query)))
    words = _TOKEN.findall(query.lower())  # stopwords kept, so they break a phrase
    phrases = list(
        dict.fromkeys(
            (a, b)
            for a, b in zip(words, words[1:], strict=False)
            if a not in STOPWORDS and b not in STOPWORDS
        )
    )
    scored = []
    for chunk in chunks:
        text = chunk["text"].lower()
        score = _score(tokens, phrases, text, _squash(text))
        if score > 0:
            scored.append({**chunk, "score": score})
    scored.sort(key=lambda c: -c["score"])  # stable, so ties keep their input order
    return scored[:k]


def rrf(rankings: list[list[dict]], k: int, rrf_k: int = RRF_K) -> list[dict]:
    """Fuse ranked lists: each chunk scores the sum of 1 / (rrf_k + rank) over the lists holding it.

    A chunk appearing in several lists is returned once. Ties go to the earlier list. The vector
    `distance` is kept when any list supplied one, otherwise it is None.
    """
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            entry = fused.setdefault(chunk["id"], {**chunk, "score": 0.0, "distance": None})
            entry["score"] += 1 / (rrf_k + rank)
            if entry["distance"] is None:
                entry["distance"] = chunk.get("distance")
    return sorted(fused.values(), key=lambda c: -c["score"])[:k]


def all_chunks(collection) -> list[dict]:
    """Every stored chunk as {id, text, metadata}, in document order."""
    res = collection.get(include=["documents", "metadatas"])
    chunks = [
        {"id": i, "text": t, "metadata": m}
        for i, t, m in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
    ]
    return sorted(chunks, key=lambda c: (c["metadata"]["doc_name"], c["metadata"]["chunk_index"]))


def hybrid_search(
    query: str, collection, embedder: Callable, k: int, pool: int = CANDIDATE_POOL
) -> list[dict]:
    """Vector search and keyword search, fused with RRF. Returns up to k chunks.

    Each search contributes `pool` candidates, or k if that is larger, so asking for more chunks
    than the pool still returns k, as vector mode does. Every chunk has id, text, metadata and
    `score` (the fused RRF score). `distance` is the cosine distance, or None for a chunk only the
    keyword search found: there is no honest distance for it, so code that reads `distance`
    must allow None.
    """
    pool = max(pool, k)
    vector_hits = store.query(collection, embedder([query])[0], k=pool)
    keyword_hits = keyword_search(query, all_chunks(collection), k=pool)
    return rrf([vector_hits, keyword_hits], k=k)
