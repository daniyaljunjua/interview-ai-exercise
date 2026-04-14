# StackOne RAG

A retrieval-augmented QA system over StackOne's seven OpenAPI specs (Platform, HRIS, ATS, LMS, IAM, CRM, Marketing). The starter scored 0/5 on the assignment's sample questions; this submission scores 5/5 with cited sources and informed refusal on unanswerable questions.

The main change is replacing character-based chunking with OpenAPI-aware chunking across operations, schemas, auth, and overview content. See [IMPROVEMENTS.md](./IMPROVEMENTS.md) for the design decisions, measured results, per-question before/after, and production roadmap.

## Setup

Requirements: Python 3.11+, [uv](https://github.com/astral-sh/uv), an OpenAI API key.

```bash
make install
cp .env_example .env
# edit .env and add: OPENAI_API_KEY=sk-...
make dev-api
```

Server runs on `http://localhost:80`.

## Try it

In a second terminal:

```bash
# Load all 7 specs (~2 min on first run, uses structural chunker by default)
curl http://localhost:80/load

# Ask one of the assignment's sample questions
curl -X POST http://localhost:80/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do you authenticate to the StackOne API?"}'
```

The `/load` endpoint accepts `?strategy=naive` for the baseline character-based chunker, used by the eval comparison below.

Sample response:

```json
{
  "message": "To authenticate to the StackOne API, you use HTTP Basic Authentication by sending your StackOne API key as the HTTP username and leaving the password empty [stackone: auth].",
  "citations": ["stackone: auth"],
  "refused": false,
  "retrieval_strategy": "hybrid"
}
```

## Run the eval

One command runs the 30-question eval and compares the baseline (`naive`) and improved (`structural`) systems:

```bash
make eval
```

This loads both collections and runs the comparison. Expected output:

```
| Version    | Hit@5 |   MRR | Ref.A |    KW |  Cite |   p50ms |   p95ms |
|------------|-------|-------|-------|-------|-------|---------|---------|
| naive      | 0.241 | 0.241 | 0.759 | 0.515 | 0.045 |    2305 |    3687 |
| structural | 0.897 | 0.738 | 0.759 | 0.909 | 0.955 |    2527 |    5521 |
```

The 30 gold questions live in `ai_exercise/eval/gold.jsonl` across 5 categories: 5 official assignment questions, 10 paraphrases, 5 cross-spec, 5 unanswerable (including the LMS create-course trap), and 5 edge cases.

Metrics:
- **Hit@5**: correct chunk in top 5 retrieved
- **MRR**: mean reciprocal rank of the correct chunk
- **Ref.A**: refusal accuracy (did it refuse on unanswerable questions and proceed on answerable ones)
- **KW**: keyword match in answer
- **Cite**: citation quality (correctly formatted citations)
- **p50/p95**: latency percentiles

## Tests

```bash
make test       # pytest
make lint       # ruff
make typecheck  # mypy
```

## Project structure

```
ai_exercise/
  loading/
    openapi_chunker.py    # Structural chunker (4 chunk types, $ref resolution)
    chunk_json.py         # JSON chunking helpers
    spec_loader.py        # OpenAPI spec fetching
    document_loader.py    # Embedding pipeline + Chroma persistence
  retrieval/
    retrieval.py          # Hybrid retrieve + multi-signal refusal
    hybrid.py             # RRF fusion
    bm25.py               # In-memory BM25 index
    vector_store.py       # Chroma collection setup
  llm/
    completions.py        # Grounded prompt + citation extractor
    embeddings.py         # OpenAI embedding wrapper
  eval/
    gold.jsonl            # 30 hand-curated questions
    metrics.py            # Hit@k, MRR, refusal accuracy, keyword match
    run.py                # CLI eval runner
  main.py                 # FastAPI: /load, /chat, /health
  constants.py            # Settings, OpenAI + Chroma clients
  models.py               # Pydantic models
demo/                     # Sample scripts from starter
docs/                     # Project notes
tests/                    # Unit tests
```

Tested on M1 MacBook Air 8GB. The first `/load` takes about 2 minutes because embeddings are created for all 7 specs; subsequent runs reuse the persisted Chroma collection in `.chroma_db/`.