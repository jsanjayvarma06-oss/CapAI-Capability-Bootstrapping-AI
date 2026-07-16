"""
capai.capability_matcher
==========================
Capability Generalization (Section: research extension #1) and
Capability Compression (research extension #5) — both are the same
underlying mechanism: detecting when two differently-worded requests
refer to the same underlying capability, so CapAI can reuse an existing
verified artifact instead of building a near-duplicate.

IMPORTANT — what this is and is not:
This is a LEXICAL/STRUCTURAL similarity matcher (token overlap +
sequence similarity), not a deep-learning semantic embedding model. It
will correctly cluster things like:
    "calculate GST"  ~  "compute GST"  ~  "18% GST on 5000"
because they share content words (gst) and structural pattern, but it
will NOT catch cases with zero lexical overlap but identical meaning
(e.g. "figure out the tax on this" vs "calculate GST" — no shared
tokens). A production-grade version of this feature would use sentence
embeddings (e.g. a small local sentence-transformer) and cosine
similarity in vector space; that is flagged explicitly as future work
rather than silently presented as already solved, since embedding-based
semantic matching is a materially different (and stronger) claim than
what is implemented here.

Design decision — matches are SUGGESTED, never auto-substituted:
Silently executing capability B when the user asked for capability A,
on the basis of a similarity score, risks exactly the kind of silent
correctness failure documented in the CapAI evaluation (Section V-A of
the paper). This module therefore never auto-executes a matched
capability; it only ever returns ranked suggestions, which a caller
(human or agent) can choose to accept.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Optional

STOPWORDS = {
    "a", "an", "the", "is", "are", "if", "of", "to", "for", "and", "or",
    "given", "from", "with", "in", "on", "this", "that", "it", "return",
    "returns", "check", "calculate", "compute", "get", "find", "make",
}


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _sequence_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


@dataclass
class MatchResult:
    name: str
    description: str
    similarity: float
    token_overlap: float
    sequence_ratio: float


def similarity_score(description_a: str, description_b: str) -> float:
    """
    Combined similarity score in [0, 1]. Weighted average of token-set
    Jaccard overlap (captures shared vocabulary regardless of word
    order) and character-sequence ratio (captures structural/phrasing
    similarity). Weights were chosen empirically during development to
    balance false positives (unrelated requests sharing a common word
    like "calculate") against false negatives (paraphrases with low
    lexical overlap); they are a tunable hyperparameter, not derived
    from a validated optimization.
    """
    tokens_a, tokens_b = _tokenize(description_a), _tokenize(description_b)
    jaccard = _jaccard(tokens_a, tokens_b)
    seq_ratio = _sequence_ratio(description_a, description_b)
    return round(0.7 * jaccard + 0.3 * seq_ratio, 4)


def find_similar_capabilities(
    query_description: str,
    existing_capabilities: list[dict],
    threshold: float = 0.35,
    top_k: int = 5,
) -> list[MatchResult]:
    """
    Given a new capability request and a list of existing capabilities
    (each a dict with at least 'name' and 'description'), return the
    top_k most similar existing capabilities scoring above `threshold`,
    ranked highest-similarity first. Returns an empty list if nothing
    clears the threshold — this is the expected, common case, and
    should not be treated as an error.
    """
    scored = []
    for cap in existing_capabilities:
        desc = cap.get("description", "")
        if not desc:
            continue
        tokens_a = _tokenize(query_description)
        tokens_b = _tokenize(desc)
        jaccard = _jaccard(tokens_a, tokens_b)
        seq_ratio = _sequence_ratio(query_description, desc)
        combined = round(0.7 * jaccard + 0.3 * seq_ratio, 4)
        if combined >= threshold:
            scored.append(MatchResult(
                name=cap.get("name", "?"),
                description=desc,
                similarity=combined,
                token_overlap=round(jaccard, 4),
                sequence_ratio=round(seq_ratio, 4),
            ))

    scored.sort(key=lambda m: m.similarity, reverse=True)
    return scored[:top_k]


def cluster_capabilities(
    capabilities: list[dict],
    threshold: float = 0.35,
) -> list[list[dict]]:
    """
    Capability Compression (research extension #5): partitions a full
    capability list into clusters of mutually-similar entries using a
    simple greedy single-linkage approach — each capability is compared
    against existing cluster representatives (the first member added to
    each cluster) and joins the first cluster it matches, or starts a
    new one. This is a coarse, O(n * clusters) heuristic, not a
    principled clustering algorithm (e.g. no attempt at optimal k,
    no hierarchical merge/split); it is intended to demonstrate the
    concept and flag likely duplicate groups for human review, not to
    automatically merge implementations without oversight.
    """
    clusters: list[list[dict]] = []

    for cap in capabilities:
        desc = cap.get("description", "")
        placed = False
        for cluster in clusters:
            representative = cluster[0]
            score = similarity_score(desc, representative.get("description", ""))
            if score >= threshold:
                cluster.append(cap)
                placed = True
                break
        if not placed:
            clusters.append([cap])

    return clusters
