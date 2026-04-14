"""Reciprocal Rank Fusion (RRF) for combining dense and BM25 results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FusedDocument:
    """A document with provenance and scores from both retrievers."""

    document: str
    metadata: dict[str, Any]
    doc_id: str
    fused_score: float
    in_dense: bool
    in_bm25: bool
    raw_dense_score: float | None
    dense_rank: int | None
    bm25_rank: int | None


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, dict[str, Any], str, float]],
    bm25_results: list[tuple[str, dict[str, Any], str, float]],
    k: int = 60,
) -> list[FusedDocument]:
    """Combine dense and BM25 results using RRF.

    Each input is a list of (document, metadata, doc_id, original_score).
    Returns fused list sorted by RRF score descending.

    Args:
        dense_results: Ranked results from embedding search.
        bm25_results: Ranked results from BM25 keyword search.
        k: RRF constant (default 60). Higher values reduce the influence of rank.

    Returns:
        Fused list of FusedDocument with retriever participation tracked.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, tuple[str, dict[str, Any]]] = {}
    dense_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}
    bm25_ranks: dict[str, int] = {}

    for rank, (doc, meta, doc_id, raw_score) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        doc_map[doc_id] = (doc, meta)
        dense_ranks[doc_id] = rank
        dense_scores[doc_id] = raw_score

    for rank, (doc, meta, doc_id, _score) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        doc_map[doc_id] = (doc, meta)
        bm25_ranks[doc_id] = rank

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        FusedDocument(
            document=doc_map[doc_id][0],
            metadata=doc_map[doc_id][1],
            doc_id=doc_id,
            fused_score=fused_score,
            in_dense=doc_id in dense_ranks,
            in_bm25=doc_id in bm25_ranks,
            raw_dense_score=dense_scores.get(doc_id),
            dense_rank=dense_ranks.get(doc_id),
            bm25_rank=bm25_ranks.get(doc_id),
        )
        for doc_id, fused_score in ranked
    ]
