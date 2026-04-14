"""CLI runner for the StackOne RAG evaluation harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.eval.metrics import (
    QuestionResult,
    Timer,
    aggregate_metrics,
)
from ai_exercise.llm.completions import create_prompt, extract_citations, get_completion
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.retrieval.bm25 import BM25Index
from ai_exercise.retrieval.retrieval import (
    get_relevant_chunks_with_scores,
    retrieve_or_refuse,
)
from ai_exercise.retrieval.vector_store import create_collection

GOLD_PATH = Path(__file__).parent / "gold.jsonl"


def load_gold(path: Path | None = None) -> list[dict[str, Any]]:
    """Load gold-standard questions from a JSONL file."""
    p = path or GOLD_PATH
    questions: list[dict[str, Any]] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _build_bm25_from_collection(
    collection: Any,
) -> BM25Index:
    """Build a BM25 index from all documents in a Chroma collection."""
    total = collection.count()
    if total == 0:
        return BM25Index()

    all_data = collection.get(include=["documents", "metadatas"])
    documents: list[str] = all_data.get("documents") or []
    metadatas: list[dict[str, str]] = all_data.get("metadatas") or []
    doc_ids: list[str] = all_data.get("ids") or []

    index = BM25Index()
    index.build(
        documents=documents,
        metadatas=[dict(m) if m else {} for m in metadatas],
        doc_ids=doc_ids,
    )
    return index


def run_eval(
    strategy: str,
    k: int = 5,
    gold_path: Path | None = None,
    skip_llm: bool = False,
    no_refuse: bool = False,
) -> tuple[list[QuestionResult], dict[str, Any]]:
    """Run the full evaluation against a given strategy's collection.

    Uses hybrid search (dense + BM25 + RRF) and retrieve_or_refuse()
    for retrieval-level refusal scoring.

    Args:
        strategy: collection strategy name (e.g. "naive", "structural").
        k: number of chunks to retrieve.
        gold_path: optional path to gold.jsonl.
        skip_llm: if True, only evaluate retrieval (no LLM calls).
        no_refuse: if True, disable refusal logic (hybrid search only).

    Returns (results, aggregated_metrics).
    """
    gold = load_gold(gold_path)
    collection_name = f"stackone_{strategy}"

    collection = create_collection(chroma_client, openai_ef, collection_name)

    if collection.count() == 0:
        print(
            f"Collection '{collection_name}' is empty. "
            f"Run /load?strategy={strategy} first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build BM25 index from the loaded collection
    bm25_index = _build_bm25_from_collection(collection)
    active_bm25 = bm25_index if bm25_index._index is not None else None

    results: list[QuestionResult] = []

    for q in gold:
        qr = QuestionResult(
            question_id=q["id"],
            answerable=q.get("answerable", True),
        )

        # --- Retrieval (hybrid + optional refusal) ---
        with Timer() as retrieval_timer:
            if no_refuse:
                # Hybrid search only, no refusal check
                scored = get_relevant_chunks_with_scores(
                    collection=collection,
                    query=q["question"],
                    k=k,
                    bm25_index=active_bm25,
                )
                docs = [doc for doc, _meta, _id, _score in scored]
                metas = [meta for _doc, meta, _id, _score in scored]
                refused = False
            else:
                # Hybrid search + refusal
                docs, refused, _retrieval_strategy = retrieve_or_refuse(
                    collection=collection,
                    query=q["question"],
                    k=k,
                    bm25_index=active_bm25,
                )
                # Always get scored results for chunk metadata (metrics need it
                # even when refused, since retrieve_or_refuse returns empty docs).
                scored = get_relevant_chunks_with_scores(
                    collection=collection,
                    query=q["question"],
                    k=k,
                    bm25_index=active_bm25,
                )
                metas = [meta for _doc, meta, _id, _score in scored]

        qr.retrieval_time_ms = retrieval_timer.elapsed_ms
        qr.retrieved_chunks = [dict(m) if m else {} for m in metas]
        qr.refused = refused

        if skip_llm or refused:
            qr.answer = ""
            qr.llm_time_ms = 0.0
            qr.has_citations = False
        else:
            # --- LLM ---
            prompt = create_prompt(query=q["question"], context=docs)
            with Timer() as llm_timer:
                qr.answer = get_completion(
                    client=openai_client,
                    prompt=prompt,
                    model=SETTINGS.openai_model,
                )
            qr.llm_time_ms = llm_timer.elapsed_ms
            qr.has_citations = len(extract_citations(qr.answer)) > 0

        qr.total_time_ms = qr.retrieval_time_ms + qr.llm_time_ms
        results.append(qr)

    metrics = aggregate_metrics(results, gold, k=k)
    return results, metrics


def print_table_row(strategy: str, metrics: dict[str, Any]) -> None:
    """Print a single row for the comparison table."""
    print(
        f"| {strategy:<30} "
        f"| {metrics['hit_at_k']:.3f} "
        f"| {metrics['mrr']:.3f} "
        f"| {metrics['refusal_accuracy']:.3f} "
        f"| {metrics['keyword_match']:.3f} "
        f"| {metrics['citation_rate']:.3f} "
        f"| {metrics['latency_p50_ms']:>7.0f} "
        f"| {metrics['latency_p95_ms']:>7.0f} |"
    )


def print_table_header() -> None:
    """Print the comparison table header."""
    print(
        f"| {'Version':<30} "
        f"| {'Hit@5':>5} "
        f"| {'MRR':>5} "
        f"| {'Ref.A':>5} "
        f"| {'KW':>5} "
        f"| {'Cite':>5} "
        f"| {'p50ms':>7} "
        f"| {'p95ms':>7} |"
    )
    sep = f"|{'-' * 32}" + (f"|{'-' * 7}" * 5) + (f"|{'-' * 9}" * 2) + "|"
    print(sep)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run StackOne RAG evaluation harness"
    )
    parser.add_argument(
        "strategies",
        nargs="+",
        help="Strategy names to evaluate (e.g. naive structural)",
    )
    parser.add_argument(
        "-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM calls; only evaluate retrieval metrics",
    )
    parser.add_argument(
        "--gold", type=str, default=None, help="Path to gold.jsonl file"
    )
    parser.add_argument(
        "--no-refuse",
        action="store_true",
        help="Disable refusal logic; measure hybrid search alone",
    )

    args = parser.parse_args()
    gold_path = Path(args.gold) if args.gold else None

    print()
    print_table_header()

    for strategy in args.strategies:
        _results, metrics = run_eval(
            strategy=strategy,
            k=args.k,
            gold_path=gold_path,
            skip_llm=args.skip_llm,
            no_refuse=args.no_refuse,
        )
        label = f"{strategy} (no-refuse)" if args.no_refuse else strategy
        print_table_row(label, metrics)

    print()


if __name__ == "__main__":
    main()
