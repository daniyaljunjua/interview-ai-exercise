"""Tests for ai_exercise/llm/completions.py."""

from ai_exercise.llm.completions import (
    create_prompt,
    extract_citations,
)

# --- create_prompt ---


def test_prompt_contains_system_instructions() -> None:
    prompt = create_prompt("How to auth?", ["chunk1"])
    assert "ONLY the provided context" in prompt
    assert "Do not use outside knowledge" in prompt


def test_prompt_contains_refusal_instruction() -> None:
    prompt = create_prompt("test", ["c"])
    assert (
        "I couldn't find sufficient information in the StackOne API "
        "specifications to answer this confidently" in prompt
    )


def test_prompt_contains_citation_instruction() -> None:
    prompt = create_prompt("test", ["c"])
    assert "[spec_name: METHOD /path]" in prompt
    assert "[spec_name: SchemaName]" in prompt


def test_prompt_contains_auth_citation_format() -> None:
    prompt = create_prompt("test", ["c"])
    assert "[spec_name: auth]" in prompt


def test_prompt_contains_overview_citation_format() -> None:
    prompt = create_prompt("test", ["c"])
    assert "[spec_name: overview]" in prompt


def test_prompt_numbers_chunks() -> None:
    prompt = create_prompt("q", ["first chunk", "second chunk"])
    assert "[1] first chunk" in prompt
    assert "[2] second chunk" in prompt


def test_prompt_contains_question() -> None:
    prompt = create_prompt("What is the base URL?", ["ctx"])
    assert "QUESTION: What is the base URL?" in prompt


def test_prompt_contains_operation_not_found_rule() -> None:
    prompt = create_prompt("test", ["c"])
    assert "operation was not found" in prompt


# --- extract_citations ---


def test_extract_operation_citation() -> None:
    answer = "Use GET /accounts [stackone: GET /accounts] to list them."
    assert extract_citations(answer) == ["stackone: GET /accounts"]


def test_extract_schema_citation() -> None:
    answer = "The field is in [hris: Employee] schema."
    assert extract_citations(answer) == ["hris: Employee"]


def test_extract_multiple_citations() -> None:
    answer = (
        "Auth uses basic [stackone: auth]. "
        "Employees via [hris: GET /unified/hris/employees]."
    )
    cites = extract_citations(answer)
    assert "stackone: auth" in cites
    assert "hris: GET /unified/hris/employees" in cites
    assert len(cites) == 2


def test_extract_deduplicates() -> None:
    answer = "[stackone: GET /accounts] and again [stackone: GET /accounts]."
    assert extract_citations(answer) == ["stackone: GET /accounts"]


def test_extract_skips_numeric_refs() -> None:
    answer = "As shown in [1] and [2], the auth uses basic."
    assert extract_citations(answer) == []


def test_extract_empty_answer() -> None:
    assert extract_citations("") == []


def test_extract_no_brackets() -> None:
    assert extract_citations("No citations here at all.") == []


def test_extract_ignores_type_brackets() -> None:
    """array[Employee] is a type annotation, not a citation."""
    assert extract_citations("data (array[Employee], REQUIRED)") == []


def test_extract_ignores_bare_schema_name() -> None:
    """[RawResponse] without a spec prefix is not a citation."""
    assert extract_citations("See [RawResponse] for details.") == []


def test_extract_ignores_bare_type() -> None:
    """[integer] is not a citation."""
    assert extract_citations("The field is [integer].") == []


def test_extract_valid_operation_citation() -> None:
    assert extract_citations("[hris: GET /accounts]") == [
        "hris: GET /accounts",
    ]


def test_extract_valid_auth_citation() -> None:
    assert extract_citations("[stackone: auth]") == [
        "stackone: auth",
    ]


def test_extract_valid_schema_citation() -> None:
    assert extract_citations("[hris: Employee]") == [
        "hris: Employee",
    ]
