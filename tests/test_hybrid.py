"""Tests for BM25 index, RRF fusion, and refusal logic."""

import pytest

from ai_exercise.retrieval.bm25 import BM25Index, _tokenize
from ai_exercise.retrieval.hybrid import FusedDocument, reciprocal_rank_fusion
from ai_exercise.retrieval.retrieval import REFUSAL_MESSAGE, should_refuse_multi

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for the BM25 tokenizer."""

    def test_lowercases(self) -> None:
        """Tokens are lowercased."""
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_preserves_hyphens_and_underscores(self) -> None:
        """Hyphens and underscores stay in tokens."""
        tokens = _tokenize("x-account-id my_field")
        assert "x-account-id" in tokens
        assert "my_field" in tokens

    def test_strips_punctuation(self) -> None:
        """Commas, periods, exclamation marks are stripped."""
        tokens = _tokenize("foo, bar. baz!")
        assert tokens == ["foo", "bar", "baz"]

    def test_preserves_paths(self) -> None:
        """URL paths with slashes are kept as single tokens."""
        tokens = _tokenize("GET /unified/hris/employees")
        assert "/unified/hris/employees" in tokens


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    "Authentication uses HTTP Basic Auth with API key as username",
    "GET /accounts returns a list of linked accounts with provider filter",
    "ConnectSessionCreate schema has expires_in default 1800 seconds",
    "GET /unified/hris/employees returns EmployeesPaginated",
    "LMS courses are read-only with GET /unified/lms/courses",
]
SAMPLE_META = [{"spec": "stackone", "chunk_type": "auth"}] + [
    {"spec": "stackone", "chunk_type": "operation"} for _ in range(4)
]
SAMPLE_IDS = [f"doc_{i}" for i in range(5)]


@pytest.fixture
def bm25() -> BM25Index:
    """Build a small BM25 index for testing."""
    index = BM25Index()
    index.build(SAMPLE_DOCS, SAMPLE_META, SAMPLE_IDS)
    return index


class TestBM25Index:
    """Tests for BM25Index."""

    def test_build_sets_documents(self, bm25: BM25Index) -> None:
        """Index stores all documents after build."""
        assert len(bm25.documents) == 5

    def test_query_returns_results(self, bm25: BM25Index) -> None:
        """Top result for auth query is the auth document."""
        results = bm25.query("authentication API key", n_results=3)
        assert len(results) == 3
        top_doc, top_meta, top_id, top_score = results[0]
        assert "Basic Auth" in top_doc
        assert top_meta["chunk_type"] == "auth"
        assert top_score > 0

    def test_query_exact_term(self, bm25: BM25Index) -> None:
        """Exact keyword match finds the right document."""
        results = bm25.query("expires_in 1800", n_results=2)
        top_doc = results[0][0]
        assert "expires_in" in top_doc
        assert "1800" in top_doc

    def test_query_empty_index(self) -> None:
        """Querying an unbuilt index returns empty list."""
        index = BM25Index()
        results = index.query("anything")
        assert results == []

    def test_query_respects_n_results(self, bm25: BM25Index) -> None:
        """Only n_results items are returned."""
        results = bm25.query("accounts", n_results=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


class TestRRF:
    """Tests for reciprocal rank fusion."""

    def test_basic_fusion(self) -> None:
        """Document appearing in both lists ranks highest."""
        dense = [
            ("doc_a", {"s": "1"}, "id_a", 0.9),
            ("doc_b", {"s": "1"}, "id_b", 0.8),
        ]
        bm25 = [
            ("doc_b", {"s": "1"}, "id_b", 5.0),
            ("doc_c", {"s": "1"}, "id_c", 3.0),
        ]
        fused = reciprocal_rank_fusion(dense, bm25, k=60)
        assert fused[0].doc_id == "id_b"

    def test_scores_are_rrf(self) -> None:
        """RRF score equals sum of 1/(k+rank+1) across both lists."""
        dense = [("doc_a", {}, "id_a", 0.9)]
        bm25 = [("doc_a", {}, "id_a", 5.0)]
        fused = reciprocal_rank_fusion(dense, bm25, k=60)
        expected = 2.0 / 61.0
        assert abs(fused[0].fused_score - expected) < 1e-9

    def test_empty_inputs(self) -> None:
        """Empty inputs produce empty output."""
        assert reciprocal_rank_fusion([], []) == []

    def test_one_side_empty(self) -> None:
        """Works when one result set is empty."""
        dense = [("doc_a", {}, "id_a", 0.9)]
        fused = reciprocal_rank_fusion(dense, [])
        assert len(fused) == 1
        assert fused[0].doc_id == "id_a"

    def test_preserves_all_unique_docs(self) -> None:
        """All unique documents from both lists appear in output."""
        dense = [("a", {}, "1", 0.5), ("b", {}, "2", 0.4)]
        bm25 = [("c", {}, "3", 3.0), ("d", {}, "4", 2.0)]
        fused = reciprocal_rank_fusion(dense, bm25)
        assert len(fused) == 4

    def test_tracks_retriever_participation(self) -> None:
        """FusedDocument records which retrievers contributed."""
        dense = [
            ("doc_a", {}, "id_a", 0.9),
            ("doc_b", {}, "id_b", 0.8),
        ]
        bm25 = [
            ("doc_b", {}, "id_b", 5.0),
            ("doc_c", {}, "id_c", 3.0),
        ]
        fused = reciprocal_rank_fusion(dense, bm25, k=60)
        by_id = {f.doc_id: f for f in fused}

        # doc_b in both
        assert by_id["id_b"].in_dense is True
        assert by_id["id_b"].in_bm25 is True
        # doc_a dense-only
        assert by_id["id_a"].in_dense is True
        assert by_id["id_a"].in_bm25 is False
        assert by_id["id_a"].raw_dense_score == 0.9
        # doc_c bm25-only
        assert by_id["id_c"].in_dense is False
        assert by_id["id_c"].in_bm25 is True
        assert by_id["id_c"].raw_dense_score is None


# ---------------------------------------------------------------------------
# Refusal logic
# ---------------------------------------------------------------------------


def _make_fused(
    *,
    fused_score: float = 0.03,
    in_dense: bool = True,
    in_bm25: bool = True,
    raw_dense_score: float | None = 0.5,
    dense_rank: int | None = 0,
    bm25_rank: int | None = 0,
) -> FusedDocument:
    """Helper to build a FusedDocument for refusal tests."""
    return FusedDocument(
        document="test doc",
        metadata={"spec": "test", "chunk_type": "operation"},
        doc_id="id_test",
        fused_score=fused_score,
        in_dense=in_dense,
        in_bm25=in_bm25,
        raw_dense_score=raw_dense_score,
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
    )


class TestMultiSignalRefusal:
    """Tests for multi-signal refusal logic."""

    def test_both_retrievers_proceed(self) -> None:
        """In both dense and BM25 → proceed."""
        top = _make_fused(in_dense=True, in_bm25=True, fused_score=0.03)
        assert should_refuse_multi(top) is False

    def test_dense_only_strong_score_proceed(self) -> None:
        """Dense-only with strong raw score → proceed."""
        top = _make_fused(
            in_dense=True, in_bm25=False, raw_dense_score=0.5,
        )
        assert should_refuse_multi(top) is False

    def test_dense_only_weak_score_refuse(self) -> None:
        """Dense-only with weak raw score → refuse."""
        top = _make_fused(
            in_dense=True, in_bm25=False, raw_dense_score=0.2,
        )
        assert should_refuse_multi(top) is True

    def test_fused_score_below_minimum_refuse(self) -> None:
        """Fused score below minimum → refuse regardless of participation."""
        top = _make_fused(
            fused_score=0.005, in_dense=True, in_bm25=True,
        )
        assert should_refuse_multi(top) is True

    def test_bm25_only_refuse(self) -> None:
        """BM25-only (not in dense) → refuse."""
        top = _make_fused(
            in_dense=False,
            in_bm25=True,
            raw_dense_score=None,
            dense_rank=None,
        )
        assert should_refuse_multi(top) is True

    def test_refusal_message_content(self) -> None:
        """Refusal message contains expected text."""
        assert "couldn't find sufficient information" in REFUSAL_MESSAGE
