"""Retrieve relevant chunks using hybrid search (dense + BM25 + RRF)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_exercise.constants import SETTINGS
from ai_exercise.retrieval.hybrid import FusedDocument, reciprocal_rank_fusion

if TYPE_CHECKING:
    import chromadb

    from ai_exercise.retrieval.bm25 import BM25Index


# Refusal message returned when evidence is too weak.
REFUSAL_MESSAGE = (
    "I couldn't find sufficient information in the indexed StackOne API "
    "specifications to answer this confidently."
)


def _dense_search(
    collection: chromadb.Collection, query: str, k: int
) -> list[tuple[str, dict[str, Any], str, float]]:
    """Run embedding-based dense search via Chroma."""
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],  # type: ignore[list-item]
    )

    docs: list[str] = (results["documents"] or [[]])[0]
    metas: list[dict[str, Any]] = (results["metadatas"] or [[]])[0]  # type: ignore[assignment]
    ids: list[str] = results["ids"][0]
    distances: list[float] = (results["distances"] or [[]])[0]

    return [
        (doc, meta, doc_id, float(dist))
        for doc, meta, doc_id, dist in zip(
            docs, metas, ids, distances, strict=True
        )
    ]


def _hybrid_search(
    collection: chromadb.Collection,
    query: str,
    k: int,
    bm25_index: BM25Index | None = None,
) -> list[FusedDocument]:
    """Run hybrid search returning FusedDocument list.

    When no BM25 index is available, wraps dense-only results as
    FusedDocument with in_bm25=False.
    """
    dense_results = _dense_search(collection, query, k)

    if bm25_index is None:
        return [
            FusedDocument(
                document=doc,
                metadata=meta,
                doc_id=doc_id,
                fused_score=score,
                in_dense=True,
                in_bm25=False,
                raw_dense_score=score,
                dense_rank=rank,
                bm25_rank=None,
            )
            for rank, (doc, meta, doc_id, score) in enumerate(
                dense_results
            )
        ]

    bm25_results = bm25_index.query(query, n_results=k)
    return reciprocal_rank_fusion(dense_results, bm25_results)[:k]


def get_relevant_chunks(
    collection: chromadb.Collection,
    query: str,
    k: int,
    bm25_index: BM25Index | None = None,
) -> list[str]:
    """Retrieve k most relevant chunks for the query.

    When a BM25 index is provided, uses hybrid search (dense + BM25 + RRF).
    Falls back to dense-only when no BM25 index is available.

    Returns the document texts of the top-k results.
    """
    result = get_relevant_chunks_with_scores(
        collection=collection,
        query=query,
        k=k,
        bm25_index=bm25_index,
    )
    return [doc for doc, _meta, _id, _score in result]


def get_relevant_chunks_with_scores(
    collection: chromadb.Collection,
    query: str,
    k: int,
    bm25_index: BM25Index | None = None,
) -> list[tuple[str, dict[str, Any], str, float]]:
    """Retrieve k most relevant chunks with scores.

    Returns list of (document, metadata, doc_id, score).
    When hybrid, the score is the RRF score.
    """
    fused = _hybrid_search(collection, query, k, bm25_index)
    return [
        (f.document, f.metadata, f.doc_id, f.fused_score)
        for f in fused
    ]


def should_refuse_multi(top: FusedDocument) -> bool:
    """Multi-signal refusal check.

    Rule 1: fused score below minimum → refuse
    Rule 2: present in both retrievers → proceed
    Rule 3: dense-only with strong raw score → proceed
    Rule 4: otherwise → refuse
    """
    if top.fused_score < SETTINGS.refusal_min_fused_score:
        return True
    if top.in_dense and top.in_bm25:
        return False
    return not (
        top.in_dense
        and top.raw_dense_score is not None
        and top.raw_dense_score >= SETTINGS.refusal_min_dense_score
    )


def retrieve_or_refuse(
    collection: chromadb.Collection,
    query: str,
    k: int,
    bm25_index: BM25Index | None = None,
) -> tuple[list[str], bool, str]:
    """Retrieve chunks and decide whether to refuse.

    Returns:
        (chunks, refused, retrieval_strategy) where:
        - chunks: list of document texts (empty if refused)
        - refused: True if the evidence was too weak
        - retrieval_strategy: "hybrid" or "dense"
    """
    fused = _hybrid_search(collection, query, k, bm25_index)
    strategy = "hybrid" if bm25_index is not None else "dense"

    if not fused:
        return [], True, strategy

    top = fused[0]
    refused = should_refuse_multi(top)

    if refused:
        return [], True, strategy

    chunks = [f.document for f in fused]
    return chunks, False, strategy
