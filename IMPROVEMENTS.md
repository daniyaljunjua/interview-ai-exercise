# Improvements & Design Decisions

A retrieval system over StackOne's seven OpenAPI specs. The starter failed the assignment’s sample questions; this submission handles all 5 correctly, with cited sources and informed refusal on unanswerable questions.

This document covers how the 5 sample questions improved, what I built, what I measured, what's still imperfect, and what I would build next.

## How the 5 sample questions behave

Both columns use the same end-to-end pipeline (hybrid retrieval, multi-signal refusal, grounded prompt, citation extraction). The only thing that differs between them is the chunker. Naive uses character-boundary splitting (the starter approach). Structural uses my chunker that respects OpenAPI structure. This isolates the impact of structural chunking specifically.

| # | Question | Naive (character-based chunking) | Structural (this submission) |
|---|---|---|---|
| 1 | How do you authenticate to the StackOne API? | "Authentication...is done via the `/connect_sessions/authenticate` endpoint. You need to send a `POST` request with the required parameters in the request body to authenticate." Cited as `[StackOne: POST /connect_sessions/authenticate]`. **Wrong endpoint**, this is a session creation flow, not API authentication. | "To authenticate to the StackOne API, you use HTTP Basic Authentication by sending your StackOne API key as the HTTP username and leaving the password empty." Cited as `[stackone: auth]`. **Correct.** |
| 2 | Can I retrieve all linked accounts with workday provider? | "I couldn't find sufficient information... The 'provider' parameter in the listing accounts endpoint is marked as deprecated `[1: GET /accounts]`." **Hedged answer** with malformed citation. | "Yes, you can retrieve all linked accounts with the Workday provider using the `GET /accounts` endpoint. You need to specify the provider 'workday' in the `providers` query parameter to filter the results by provider." Cited as `[stackone: GET /accounts]`. **Correct.** |
| 3 | What is the default expiry of the session token? | "The default expiry of the session token is 1800 seconds `[spec_name: SchemaName]`." **Right answer with broken citation**. The LLM literally returned the prompt's placeholder text and the citation extractor rejected it. | "The default expiry of the session token is 1800 seconds (30 minutes)." Cited as `[stackone: POST /connect_sessions]`. **Correct.** |
| 4 | What fields must be sent to create a course on an LMS? | "I couldn't find sufficient information in the StackOne API specifications to answer this confidently." **Refused** but for the wrong reason. Retrieval found nothing rather than the system understanding the operation doesn't exist. | "I couldn't find sufficient information... The provided context does not include any operation related to creating a course on an LMS." **Informed refusal** that explicitly states the operation is missing from the indexed specs. |
| 5 | What is the response body when listing an employee? | "I couldn't find sufficient information in the StackOne API specifications regarding the explicit response body when listing an employee." **False refusal**. The answer is in the spec, but the chunker had split `EmployeesPaginated` across multiple character-boundary chunks so retrieval could not reconstruct it. | Returns the full `EmployeesPaginated` schema: `next_page` (deprecated), `next`, `data` (array of `Employee` with all 50+ fields enumerated), `raw` (RawResponse with method, url, body, response). Cited as `[hris: GET /unified/hris/employees]`. **Correct.** |

**Score: 0 to 1 of 5 became 5 of 5**, including the trap question.

## Aggregate eval results

Across the full 30-question gold set:

| Version | Hit@5 | MRR | Refusal Acc. | Keyword | Citation | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| Naive | 0.241 | 0.241 | 0.759 | 0.515 | 0.045 | 2305 | 3687 |
| Structural | 0.897 | 0.738 | 0.759 | 0.909 | 0.955 | 2527 | 5521 |

**Hit@5 went from 0.241 to 0.897**, a 3.7x improvement driven by structural chunking.

**Citation quality went from 0.045 to 0.955.** The naive system frequently returned malformed citations like `[spec_name: SchemaName]` (literal placeholder text); the structural system produces clean, verifiable citations.

**Refusal accuracy held steady at 0.759.** That means 23 of 30 refuse/proceed decisions were correct in both versions. Refusal accuracy is identical between naive and structural because the refusal layer is the same in both versions; only the chunker differs. The 5 official assignment questions all behave correctly. The 0.759 reflects borderline edge cases in the broader 30-question gold set, where trap questions sometimes have semantically related chunks (Course schemas, completion endpoints) that the threshold lets through. Calibrating thresholds against a larger labelled set is in the next steps.

**Latency** is dominated by the LLM generation call (around 1.5s out of 2.5s p50). Retrieval itself adds about 150ms.

## What I built

Four pieces, each fixing a specific failure mode from the baseline.

**1. Structural chunking with four chunk types.** Operations, schemas, auth, and overview chunks. Each one walks OpenAPI's natural structure rather than cutting at character boundaries. `$ref` resolver follows pointers up to depth 5 with cycle detection. Boilerplate error responses (4xx, 5xx) are stripped to reduce noise.

**2. Hybrid retrieval.** BM25 plus dense embeddings combined with Reciprocal Rank Fusion. Dense catches meaning, BM25 catches exact tokens like `x-account-id`. Fusion gives the best of both.

**3. Multi-signal refusal.** Replaces a single RRF threshold (which capped at 0.0164 for single-retriever matches and rejected legitimate retrievals) with three signals: both retrievers agreeing, dense-only with strong raw similarity, or absolute fused score floor.

**4. Grounded prompt with citations.** Strict rules to only use provided context, cite every claim inline as `[spec: METHOD /path]` or `[spec: SchemaName]`, and refuse with a fixed phrase when context is insufficient. A regex extractor filtered to known spec names prevents false positives like `array[Employee]` being parsed as a citation.

## How the eval set was built

`ai_exercise/eval/gold.jsonl` contains 30 hand-curated questions across five categories: 5 official (the assignment's exact sample questions, ground truth verified by reading the specs directly), 10 paraphrases, 5 cross-spec, 5 unanswerable (including the LMS create-course trap), and 5 edge cases.

Each entry stores the question text, an `answerable` flag, the expected chunk identifier, and a list of `expected_keywords`. The eval is run via a CLI (`make eval`) and prints the comparison table directly.

The metrics are deterministic IR metrics (Hit@k, MRR, refusal accuracy, keyword match, latency percentiles) rather than LLM-as-judge scores. This was a deliberate tradeoff for reproducibility, cost, and debuggability during iteration. For production a layered approach makes sense, with deterministic metrics for offline regression testing and sampled LLM-as-judge (RAGAS-style) for online quality monitoring. See "Suggested Improvements" for how I would add this.

## Limitations

- Operation chunks get bloated when `$ref` resolution inlines large enums (`LanguageEnum` has 400+ locale codes). This dilutes the embedding signal.
- Overview chunks can outrank operation chunks when queries mention category names like "Courses" or "Candidates", because BM25 matches on those tokens.
- Refusal thresholds (0.008 fused, 0.35 raw dense) were tuned on 30 questions. Borderline trap questions sometimes slip through.
- Cross-spec questions like "which APIs support pagination?" only retrieve from one or two specs.
- BM25 index is rebuilt in memory on every server restart.

## Suggested Improvements

What I would build next to take this from a working take-home to production. Five items, in priority order.

### 1. Fix operation chunk bloat

**Problem.** The `$ref` resolver inlines full schemas including huge enums, which drowns the operation's actual signal in noise.

**Fix.** Operation chunks reference schemas by name (e.g. "Response: `EmployeesPaginated`") instead of inlining them. Schema chunks remain the source of truth for fields. Each chunk does one job.

**Tradeoff.** Cleaner retrieval, no information loss, but the LLM may need both an operation chunk and its referenced schema chunk in context for some questions. Worth it.

### 2. Chunk-type preference in retrieval

**Problem.** Overview chunks containing category names like "Courses" beat the actual `lms_list_courses` operation chunk on BM25 scoring.

**Fix.** Score boost based on query intent. Words like "endpoint", "list", "create" prefer operation chunks. Words like "fields", "schema" prefer schema chunks. Words like "authenticate" prefer auth chunks.

**Tradeoff.** Cheap to implement (keyword heuristic plus a multiplier). Brittle for paraphrased queries; would replace with an intent classifier in production.

### 3. Layered evaluation: deterministic + sampled LLM-as-judge

**Problem.** My eval uses deterministic IR metrics (Hit@k, MRR, refusal accuracy) which are great for development but cannot detect answer-quality drift in production. A retrieval-correct answer can still be poorly phrased.

**Fix.** Two-tier eval system:
- **Offline (PR gate).** Keep the current IR metrics. Add RAGAS-style faithfulness and answer relevancy on the gold set, computed by `gpt-4o-as-judge`. Block PRs on regression.
- **Online (production sampling).** Sample 5-10% of `/chat` requests asynchronously through the same RAGAS judges. Track scores over time. Auto-route low-faithfulness runs (< 0.7) to a human review queue.

This combines reproducibility (deterministic IR for development) with quality monitoring (LLM-as-judge for production drift).

**Tradeoff.** Cost of judge calls (mitigated by sampling). Needs observability infra (Langfuse or LangSmith).

**References.** Es et al. (2023) RAGAS framework, arXiv:2309.15217.

### 4. Production deployment and observability

**Problem.** Production needs cloud deployment, monitoring, secrets management, rate limiting, and cost controls.

**Fix.**
- **Deploy** to Cloud Run or AWS Fargate. FastAPI is already containerized via the included Dockerfile.
- **Persist Chroma** to a managed vector DB (Pinecone, Weaviate, or Chroma Cloud) instead of local disk.
- **Persist BM25 index** to disk on shutdown so cold starts don't re-tokenize 800 chunks.
- **Secrets** in a vault (AWS Secrets Manager, GCP Secret Manager). Currently `OPENAI_API_KEY` is in `.env`.
- **Observability** via Langfuse: per-query traces, latency percentiles, token usage, retrieval-quality dashboards.
- **Rate limiting and auth** on the `/chat` endpoint. Currently anyone with the URL can call it.
- **Cost controls.** Per-tenant token budget, alerting when spend exceeds a daily threshold.

### 5. Latency and cost reductions

**Problem.** p50 latency is 2.5s, dominated by the LLM generation call (~1.5s). Cost is ~$0.005 per query at gpt-4o prices.

**Fix.**
- **Stream LLM tokens** to the client. First-token latency drops from 2.5s to ~500ms.
- **Route trivial queries** (auth, overview, simple schema lookups) to `gpt-4o-mini`. Cuts cost ~10x and latency ~2x for the easy cases.
- **Cache (query, top-k chunks, answer) tuples** with a short TTL. Repeat queries return in <100ms.
- **Spec refresh by content hash** so unchanged specs skip re-embedding on `/load`.