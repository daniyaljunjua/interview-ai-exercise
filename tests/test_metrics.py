"""Tests for ai_exercise/eval/metrics.py and eval/run.py helpers."""

import pytest

from ai_exercise.eval.metrics import (
    QuestionResult,
    Timer,
    aggregate_metrics,
    hit_at_k,
    keyword_match_score,
    latency_percentiles,
    reciprocal_rank,
    refusal_accuracy,
)
from ai_exercise.eval.run import _build_bm25_from_collection

# --- hit_at_k ---


def test_hit_when_expected_chunk_is_first() -> None:
    retrieved = [{"spec": "stackone", "chunk_type": "auth"}]
    expected = [{"spec": "stackone", "chunk_type": "auth"}]
    assert hit_at_k(retrieved, expected, k=5) == 1.0


def test_hit_when_expected_chunk_is_at_rank_k() -> None:
    retrieved = [
        {"spec": "hris", "chunk_type": "operation"},
        {"spec": "ats", "chunk_type": "operation"},
        {"spec": "stackone", "chunk_type": "auth"},
    ]
    expected = [{"spec": "stackone", "chunk_type": "auth"}]
    assert hit_at_k(retrieved, expected, k=3) == 1.0


def test_hit_miss_when_expected_is_beyond_k() -> None:
    retrieved = [
        {"spec": "hris", "chunk_type": "operation"},
        {"spec": "ats", "chunk_type": "operation"},
        {"spec": "stackone", "chunk_type": "auth"},
    ]
    expected = [{"spec": "stackone", "chunk_type": "auth"}]
    assert hit_at_k(retrieved, expected, k=2) == 0.0


def test_hit_miss_when_no_match() -> None:
    retrieved = [{"spec": "hris", "chunk_type": "operation"}]
    expected = [{"spec": "stackone", "chunk_type": "auth"}]
    assert hit_at_k(retrieved, expected, k=5) == 0.0


def test_hit_empty_expected_returns_1() -> None:
    retrieved = [{"spec": "hris", "chunk_type": "operation"}]
    assert hit_at_k(retrieved, [], k=5) == 1.0


def test_hit_partial_metadata_match() -> None:
    """Expected chunk only specifies spec and chunk_type; retrieved has more."""
    retrieved = [
        {
            "spec": "stackone",
            "chunk_type": "operation",
            "path": "/accounts",
            "method": "GET",
        }
    ]
    expected = [{"spec": "stackone", "chunk_type": "operation"}]
    assert hit_at_k(retrieved, expected, k=5) == 1.0


def test_hit_extra_expected_fields_must_match() -> None:
    """If expected specifies path, retrieved must have it too."""
    retrieved = [
        {
            "spec": "stackone",
            "chunk_type": "operation",
            "path": "/connectors",
            "method": "GET",
        }
    ]
    expected = [
        {"spec": "stackone", "chunk_type": "operation", "path": "/accounts"}
    ]
    assert hit_at_k(retrieved, expected, k=5) == 0.0


# --- reciprocal_rank ---


def test_rr_rank_1() -> None:
    retrieved = [{"spec": "stackone", "chunk_type": "auth"}]
    expected = [{"spec": "stackone", "chunk_type": "auth"}]
    assert reciprocal_rank(retrieved, expected) == 1.0


def test_rr_rank_3() -> None:
    retrieved = [
        {"spec": "hris", "chunk_type": "operation"},
        {"spec": "ats", "chunk_type": "schema"},
        {"spec": "stackone", "chunk_type": "auth"},
    ]
    expected = [{"spec": "stackone", "chunk_type": "auth"}]
    assert reciprocal_rank(retrieved, expected) == pytest.approx(1 / 3)


def test_rr_no_match() -> None:
    retrieved = [{"spec": "hris", "chunk_type": "operation"}]
    expected = [{"spec": "stackone", "chunk_type": "auth"}]
    assert reciprocal_rank(retrieved, expected) == 0.0


def test_rr_empty_expected() -> None:
    retrieved = [{"spec": "hris", "chunk_type": "operation"}]
    assert reciprocal_rank(retrieved, []) == 1.0


# --- refusal_accuracy ---


def test_refusal_perfect() -> None:
    results = [
        QuestionResult(question_id="q1", answerable=True, refused=False),
        QuestionResult(question_id="q2", answerable=False, refused=True),
    ]
    assert refusal_accuracy(results) == 1.0


def test_refusal_all_wrong() -> None:
    results = [
        QuestionResult(question_id="q1", answerable=True, refused=True),
        QuestionResult(question_id="q2", answerable=False, refused=False),
    ]
    assert refusal_accuracy(results) == 0.0


def test_refusal_partial() -> None:
    results = [
        QuestionResult(question_id="q1", answerable=True, refused=False),
        QuestionResult(question_id="q2", answerable=False, refused=False),
        QuestionResult(question_id="q3", answerable=True, refused=True),
        QuestionResult(question_id="q4", answerable=False, refused=True),
    ]
    assert refusal_accuracy(results) == 0.5


def test_refusal_empty() -> None:
    assert refusal_accuracy([]) == 0.0


# --- keyword_match_score ---


def test_kw_all_present() -> None:
    answer = "Use HTTP Basic Auth with your API key as the username"
    keywords = ["basic", "api key", "username"]
    assert keyword_match_score(answer, keywords) == 1.0


def test_kw_some_present() -> None:
    answer = "Use Basic Auth to authenticate"
    keywords = ["basic", "api key", "username"]
    assert keyword_match_score(answer, keywords) == pytest.approx(1 / 3)


def test_kw_none_present() -> None:
    answer = "I don't know"
    keywords = ["basic", "api key"]
    assert keyword_match_score(answer, keywords) == 0.0


def test_kw_case_insensitive() -> None:
    answer = "BASIC AUTH with API KEY"
    keywords = ["basic", "api key"]
    assert keyword_match_score(answer, keywords) == 1.0


def test_kw_empty_keywords() -> None:
    assert keyword_match_score("anything", []) == 1.0


# --- latency_percentiles ---


def test_latency_basic() -> None:
    times = list(range(1, 101))  # 1..100
    result = latency_percentiles(times)
    assert result["p50"] == 51.0  # int(100 * 0.5) = index 50 → value 51
    assert result["p95"] == 96.0  # int(100 * 0.95) = index 95 → value 96


def test_latency_single() -> None:
    result = latency_percentiles([42.0])
    assert result["p50"] == 42.0
    assert result["p95"] == 42.0


def test_latency_empty() -> None:
    result = latency_percentiles([])
    assert result["p50"] == 0.0
    assert result["p95"] == 0.0


# --- aggregate_metrics ---


def test_aggregate_full() -> None:
    gold = [
        {
            "id": "q1",
            "question": "How to auth?",
            "answerable": True,
            "expected_keywords": ["basic", "key"],
            "expected_chunks": [{"spec": "stackone", "chunk_type": "auth"}],
        },
        {
            "id": "q2",
            "question": "Create LMS course?",
            "answerable": False,
            "expected_keywords": [],
            "expected_chunks": [],
        },
    ]
    results = [
        QuestionResult(
            question_id="q1",
            answerable=True,
            retrieved_chunks=[{"spec": "stackone", "chunk_type": "auth"}],
            answer="Use basic auth with your key",
            refused=False,
            total_time_ms=100.0,
            has_citations=True,
        ),
        QuestionResult(
            question_id="q2",
            answerable=False,
            retrieved_chunks=[{"spec": "lms", "chunk_type": "operation"}],
            answer="I couldn't find sufficient information",
            refused=True,
            total_time_ms=50.0,
        ),
    ]
    metrics = aggregate_metrics(results, gold, k=5)
    assert metrics["hit_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["keyword_match"] == 1.0
    assert metrics["citation_rate"] == 1.0


# --- Timer ---


def test_timer_records_time() -> None:
    with Timer() as t:
        _ = sum(range(10000))
    assert t.elapsed_ms > 0


# --- refusal flag interaction with metrics ---


def test_refusal_accuracy_with_no_refuse_mode() -> None:
    """When no_refuse is active, refused is always False.

    For answerable questions that's correct; for unanswerable questions it's wrong.
    """
    results = [
        QuestionResult(question_id="q1", answerable=True, refused=False),
        QuestionResult(question_id="q2", answerable=False, refused=False),  # wrong
        QuestionResult(question_id="q3", answerable=True, refused=False),
    ]
    # 2 correct (q1, q3), 1 wrong (q2)
    assert refusal_accuracy(results) == pytest.approx(2 / 3)


def test_refusal_accuracy_retrieval_level_refuse() -> None:
    """Retrieval-level refusal sets refused=True without LLM text parsing."""
    results = [
        QuestionResult(
            question_id="q1", answerable=True, refused=False, answer="Use basic auth"
        ),
        QuestionResult(
            question_id="q4", answerable=False, refused=True, answer=""
        ),
    ]
    assert refusal_accuracy(results) == 1.0


# --- _build_bm25_from_collection ---


class _FakeCollection:
    """Minimal fake Chroma collection for testing BM25 builder."""

    def __init__(
        self,
        docs: list[str],
        metas: list[dict[str, str]],
        ids: list[str],
    ) -> None:
        self._docs = docs
        self._metas = metas
        self._ids = ids

    def count(self) -> int:
        return len(self._docs)

    def get(self, include: list[str] | None = None) -> dict:
        return {
            "documents": self._docs,
            "metadatas": self._metas,
            "ids": self._ids,
        }


def test_build_bm25_from_empty_collection() -> None:
    coll = _FakeCollection([], [], [])
    index = _build_bm25_from_collection(coll)
    assert index._index is None


def test_build_bm25_from_populated_collection() -> None:
    coll = _FakeCollection(
        docs=["hello world", "foo bar baz"],
        metas=[{"spec": "a"}, {"spec": "b"}],
        ids=["id1", "id2"],
    )
    index = _build_bm25_from_collection(coll)
    assert index._index is not None
    assert len(index.documents) == 2

    results = index.query("hello", n_results=1)
    assert len(results) == 1
    assert results[0][0] == "hello world"
