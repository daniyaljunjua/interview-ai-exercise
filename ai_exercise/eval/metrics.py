"""Evaluation metrics for the StackOne RAG system."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionResult:
    """Result of evaluating a single question."""

    question_id: str
    answerable: bool
    retrieval_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    total_time_ms: float = 0.0
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    refused: bool = False
    has_citations: bool = False


def hit_at_k(
    retrieved: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    k: int = 5,
) -> float:
    """Check if any expected chunk appears in the top-k retrieved chunks.

    Returns 1.0 if hit, 0.0 if miss. Matching uses the fields present in
    each expected chunk dict (spec, chunk_type, and optionally path/method/
    schema_name).
    """
    if not expected:
        return 1.0  # no expected chunks means retrieval isn't scored

    top_k = retrieved[:k]
    for exp in expected:
        for ret in top_k:
            if _chunk_matches(ret, exp):
                return 1.0
    return 0.0


def reciprocal_rank(
    retrieved: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> float:
    """Reciprocal rank of the first matching chunk.

    Returns 1/rank for the first retrieved chunk matching any expected chunk,
    or 0.0 if none match.
    """
    if not expected:
        return 1.0

    for rank, ret in enumerate(retrieved, start=1):
        for exp in expected:
            if _chunk_matches(ret, exp):
                return 1.0 / rank
    return 0.0


def refusal_accuracy(results: list[QuestionResult]) -> float:
    """Fraction of questions where the refusal decision was correct.

    Correct means: refused on unanswerable questions AND did not refuse
    on answerable questions.
    """
    if not results:
        return 0.0
    correct = sum(
        1
        for r in results
        if (not r.answerable and r.refused) or (r.answerable and not r.refused)
    )
    return correct / len(results)


def keyword_match_score(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    lower_answer = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower_answer)
    return hits / len(expected_keywords)


def latency_percentiles(
    times_ms: list[float],
) -> dict[str, float]:
    """Compute p50 and p95 latency from a list of times in milliseconds."""
    if not times_ms:
        return {"p50": 0.0, "p95": 0.0}
    sorted_times = sorted(times_ms)
    n = len(sorted_times)
    p50 = sorted_times[int(n * 0.5)]
    p95 = sorted_times[min(int(n * 0.95), n - 1)]
    return {"p50": round(p50, 1), "p95": round(p95, 1)}


def aggregate_metrics(
    results: list[QuestionResult],
    gold: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    """Compute all metrics across a full evaluation run.

    Args:
        results: list of QuestionResult from the eval run.
        gold: list of gold-standard question dicts (from gold.jsonl).
        k: top-k for Hit@k.

    Returns a dict with hit_at_k, mrr, refusal_accuracy, keyword_match,
    latency_p50_ms, latency_p95_ms.
    """
    gold_by_id = {g["id"]: g for g in gold}

    hit_scores: list[float] = []
    mrr_scores: list[float] = []
    kw_scores: list[float] = []
    total_times: list[float] = []

    for r in results:
        g = gold_by_id.get(r.question_id)
        if g is None:
            continue

        expected_chunks = g.get("expected_chunks", [])
        expected_keywords = g.get("expected_keywords", [])

        hit_scores.append(hit_at_k(r.retrieved_chunks, expected_chunks, k))
        mrr_scores.append(reciprocal_rank(r.retrieved_chunks, expected_chunks))

        if r.answerable:
            kw_scores.append(keyword_match_score(r.answer, expected_keywords))

        total_times.append(r.total_time_ms)

    latencies = latency_percentiles(total_times)

    # Citation rate: among answerable questions that weren't refused
    answerable_answered = [r for r in results if r.answerable and not r.refused]
    citation_count = sum(1 for r in answerable_answered if r.has_citations)
    citation_rate = (
        citation_count / len(answerable_answered)
        if answerable_answered
        else 0.0
    )

    return {
        "hit_at_k": round(_safe_mean(hit_scores), 3),
        "mrr": round(_safe_mean(mrr_scores), 3),
        "refusal_accuracy": round(refusal_accuracy(results), 3),
        "keyword_match": round(_safe_mean(kw_scores), 3),
        "citation_rate": round(citation_rate, 3),
        "latency_p50_ms": latencies["p50"],
        "latency_p95_ms": latencies["p95"],
    }


class Timer:
    """Simple context-manager timer returning elapsed milliseconds."""

    def __init__(self) -> None:  # noqa: D107
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:  # noqa: D105
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:  # noqa: D105
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


def _chunk_matches(retrieved: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check if a retrieved chunk's metadata matches an expected chunk spec."""
    return all(retrieved.get(key) == value for key, value in expected.items())


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
