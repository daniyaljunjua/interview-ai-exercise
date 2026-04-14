"""Generate a response using an LLM."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

_SYSTEM_PROMPT = """\
You are a StackOne API assistant. Answer questions using ONLY the \
provided context from StackOne's OpenAPI specifications.

Rules:
1. If the context does not contain sufficient information to answer \
the question, say: "I couldn't find sufficient information in the \
StackOne API specifications to answer this confidently."
2. Every factual claim must cite its source inline as \
[spec_name: METHOD /path] for endpoints or [spec_name: SchemaName] \
for schemas.
3. Do not use outside knowledge. Do not guess. Do not extrapolate \
from read schemas to infer write schemas.
4. If the user asks about an operation that does not appear in the \
context, explicitly state that the operation was not found in the \
indexed specifications.
5. For authentication and global request conventions, cite as \
[spec_name: auth].
6. For API overview or category information, cite as \
[spec_name: overview]."""

# Matches valid citations: [spec_name: auth], [spec_name: METHOD /path],
# or [spec_name: SchemaName].
_SPEC_NAMES = "stackone|hris|ats|lms|iam|crm|marketing"
_CITATION_RE = re.compile(
    rf"\[({_SPEC_NAMES}):\s*([^\]]+)\]",
    re.IGNORECASE,
)


def create_prompt(query: str, context: list[str]) -> str:
    """Create a grounded prompt combining query and numbered context chunks."""
    numbered = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, chunk in enumerate(context)
    )
    return f"""{_SYSTEM_PROMPT}

CONTEXT:
{numbered}

QUESTION: {query}

ANSWER:"""


def extract_citations(answer: str) -> list[str]:
    """Extract inline citations from an LLM answer.

    Only returns citations matching [spec_name: content] where spec_name
    is one of the known StackOne spec names (stackone, hris, ats, lms,
    iam, crm, marketing).
    """
    seen: set[str] = set()
    citations: list[str] = []
    for spec, content in _CITATION_RE.findall(answer):
        citation = f"{spec}: {content.strip()}"
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations


def get_completion(client: OpenAI, prompt: str, model: str) -> str:
    """Get completion from OpenAI"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""
