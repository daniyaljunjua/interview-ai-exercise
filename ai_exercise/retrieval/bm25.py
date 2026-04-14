"""BM25 keyword search index for hybrid retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9_\-/]+", text.lower())


@dataclass
class BM25Index:
    """In-memory BM25 index over a corpus of text documents."""

    documents: list[str] = field(default_factory=list)
    metadatas: list[dict[str, str]] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    _index: BM25Okapi | None = field(default=None, repr=False)

    def build(
        self,
        documents: list[str],
        metadatas: list[dict[str, str]],
        doc_ids: list[str],
    ) -> None:
        """Build the BM25 index from a corpus."""
        self.documents = documents
        self.metadatas = metadatas
        self.doc_ids = doc_ids
        tokenized = [_tokenize(doc) for doc in documents]
        self._index = BM25Okapi(tokenized)

    def query(
        self, query_text: str, n_results: int = 5
    ) -> list[tuple[str, dict[str, str], str, float]]:
        """Search the index. Returns list of (document, metadata, doc_id, score)."""
        if self._index is None or len(self.documents) == 0:
            return []

        tokenized_query = _tokenize(query_text)
        scores = self._index.get_scores(tokenized_query)

        # Get top-n indices sorted by score descending
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_results]

        return [
            (
                self.documents[i],
                self.metadatas[i],
                self.doc_ids[i],
                float(scores[i]),
            )
            for i in ranked_indices
        ]
