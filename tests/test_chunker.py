"""Tests for the OpenAPI structural chunker."""

from __future__ import annotations

from typing import Any

import pytest

from ai_exercise.loading.openapi_chunker import (
    _is_boilerplate_schema,
    chunk_spec,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal OpenAPI spec structures
# ---------------------------------------------------------------------------

MINIMAL_SPEC: dict[str, Any] = {
    "openapi": "3.1.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "servers": [{"url": "https://api.example.com"}],
    "security": [{"basic": []}],
    "tags": [{"name": "Users"}, {"name": "Accounts"}],
    "paths": {
        "/users": {
            "get": {
                "operationId": "list_users",
                "tags": ["Users"],
                "summary": "List all users",
                "description": "Returns a paginated list of users.",
                "parameters": [
                    {
                        "name": "page",
                        "in": "query",
                        "schema": {
                            "type": "integer",
                            "default": 1,
                        },
                        "required": False,
                        "description": "Page number",
                    },
                ],
                "responses": {
                    "200": {
                        "description": "Success",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": (
                                        "#/components/schemas"
                                        "/UsersPaginated"
                                    ),
                                },
                            },
                        },
                    },
                    "400": {
                        "description": "Bad request",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": (
                                        "#/components/schemas"
                                        "/BadRequestResponse"
                                    ),
                                },
                            },
                        },
                    },
                },
            },
            "post": {
                "operationId": "create_user",
                "tags": ["Users"],
                "summary": "Create a user",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": (
                                    "#/components/schemas"
                                    "/UserCreate"
                                ),
                            },
                        },
                    },
                },
                "responses": {
                    "201": {
                        "description": "Created",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": (
                                        "#/components/schemas/User"
                                    ),
                                },
                            },
                        },
                    },
                },
            },
        },
    },
    "components": {
        "securitySchemes": {
            "basic": {
                "type": "http",
                "scheme": "basic",
            },
        },
        "schemas": {
            "User": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique ID",
                    },
                    "name": {
                        "type": "string",
                        "description": "User name",
                    },
                    "email": {
                        "type": ["string", "null"],
                        "description": "Email address",
                    },
                },
            },
            "UserCreate": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "User name",
                    },
                    "email": {
                        "type": "string",
                        "nullable": True,
                    },
                },
            },
            "UsersPaginated": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {
                            "$ref": "#/components/schemas/User",
                        },
                    },
                    "next": {"type": ["string", "null"]},
                },
            },
            "BadRequestResponse": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "status_code": {"type": "integer"},
                },
            },
        },
    },
}


def _spec_with_circular_ref() -> dict[str, Any]:
    """Spec where Employee.manager refs Employee."""
    return {
        "openapi": "3.1.0",
        "info": {"title": "Circular Test", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {},
        "components": {
            "schemas": {
                "Employee": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "manager": {
                            "$ref": (
                                "#/components/schemas/Employee"
                            ),
                        },
                    },
                },
            },
        },
    }


def _spec_with_deprecated_param() -> dict[str, Any]:
    """Spec with a deprecated parameter."""
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Deprecated Test",
            "version": "1.0.0",
        },
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/accounts": {
                "get": {
                    "operationId": "list_accounts",
                    "summary": "List accounts",
                    "parameters": [
                        {
                            "name": "provider",
                            "in": "query",
                            "schema": {"type": "string"},
                            "deprecated": True,
                            "description": "Use providers",
                        },
                        {
                            "name": "providers",
                            "in": "query",
                            "schema": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    ],
                    "responses": {
                        "200": {"description": "OK"},
                    },
                },
            },
        },
        "components": {"schemas": {}},
    }


def _spec_with_nullable_default() -> dict[str, Any]:
    """Spec with expires_in that has a default."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Session Test",
            "version": "1.0.0",
        },
        "servers": [{"url": "https://api.example.com"}],
        "paths": {},
        "components": {
            "schemas": {
                "ConnectSessionCreate": {
                    "type": "object",
                    "required": ["origin_owner_id"],
                    "description": (
                        "Request body for creating a session."
                    ),
                    "properties": {
                        "origin_owner_id": {
                            "type": "string",
                            "description": "The origin owner id",
                        },
                        "expires_in": {
                            "type": "number",
                            "default": 1800,
                            "nullable": True,
                            "description": (
                                "How long the session should be"
                                " valid for in seconds"
                            ),
                        },
                    },
                },
            },
        },
    }


def _get_chunks_by_type(
    chunks: list[Any], chunk_type: str,
) -> list[Any]:
    """Filter chunks by chunk_type metadata."""
    return [
        c for c in chunks
        if c.metadata and c.metadata["chunk_type"] == chunk_type
    ]


def _get_chunk_by_meta(
    chunks: list[Any], key: str, value: str,
) -> Any:
    """Find first chunk matching metadata key=value."""
    return next(
        c for c in chunks
        if c.metadata and c.metadata.get(key) == value
    )


# ---------------------------------------------------------------------------
# Tests: chunk counts and types
# ---------------------------------------------------------------------------


class TestChunkSpec:
    """Test the main chunk_spec entry point."""

    def test_produces_all_four_chunk_types(self) -> None:
        """All four chunk types should be present."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        types = {
            c.metadata["chunk_type"]
            for c in chunks
            if c.metadata
        }
        assert types == {"auth", "overview", "operation", "schema"}

    def test_operation_count(self) -> None:
        """Two ops: GET /users and POST /users."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        ops = _get_chunks_by_type(chunks, "operation")
        assert len(ops) == 2

    def test_schema_count_skips_boilerplate(self) -> None:
        """4 schemas, 1 boilerplate -> 3 schema chunks."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        schemas = _get_chunks_by_type(chunks, "schema")
        assert len(schemas) == 3
        names = {c.metadata["schema_name"] for c in schemas}
        assert "BadRequestResponse" not in names
        assert "User" in names
        assert "UsersPaginated" in names

    def test_one_auth_chunk_per_spec(self) -> None:
        """Exactly one auth chunk per spec."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        auth = _get_chunks_by_type(chunks, "auth")
        assert len(auth) == 1


# ---------------------------------------------------------------------------
# Tests: operation chunk content
# ---------------------------------------------------------------------------


class TestOperationChunks:
    """Test operation chunk formatting and metadata."""

    def test_operation_metadata(self) -> None:
        """Metadata has spec, type, path, method, tags."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        op = _get_chunk_by_meta(
            chunks, "operation_id", "list_users"
        )
        assert op.metadata["spec"] == "test"
        assert op.metadata["chunk_type"] == "operation"
        assert op.metadata["path"] == "/users"
        assert op.metadata["method"] == "GET"
        assert op.metadata["tags"] == "Users"

    def test_operation_text_has_key_info(self) -> None:
        """Text includes method, path, summary, params."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        op = _get_chunk_by_meta(
            chunks, "operation_id", "list_users"
        )
        text = op.page_content
        assert "GET /users" in text
        assert "List all users" in text
        assert "page" in text
        assert "default=1" in text

    def test_ref_resolution_in_response(self) -> None:
        """Response $ref resolves to show actual fields."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        op = _get_chunk_by_meta(
            chunks, "operation_id", "list_users"
        )
        text = op.page_content
        assert "UsersPaginated" in text
        assert "id" in text
        assert "name" in text

    def test_error_responses_stripped(self) -> None:
        """400 response should NOT appear in chunk text."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        op = _get_chunk_by_meta(
            chunks, "operation_id", "list_users"
        )
        text = op.page_content
        assert "400" not in text
        assert "BadRequestResponse" not in text

    def test_request_body_resolved(self) -> None:
        """POST shows UserCreate fields in request body."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        op = _get_chunk_by_meta(
            chunks, "operation_id", "create_user"
        )
        text = op.page_content
        assert "Request Body: UserCreate" in text
        assert "name" in text

    def test_security_shown(self) -> None:
        """Security section appears with auth type."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        op = _get_chunk_by_meta(
            chunks, "operation_id", "list_users"
        )
        assert "Security:" in op.page_content
        assert "Basic Auth" in op.page_content

    def test_deprecated_param_marked(self) -> None:
        """Deprecated params have DEPRECATED marker."""
        spec = _spec_with_deprecated_param()
        chunks = chunk_spec("test", spec)
        op = _get_chunks_by_type(chunks, "operation")[0]
        assert "DEPRECATED" in op.page_content


# ---------------------------------------------------------------------------
# Tests: schema chunks
# ---------------------------------------------------------------------------


class TestSchemaChunks:
    """Test schema chunk formatting."""

    def test_schema_metadata(self) -> None:
        """Metadata includes schema_name and required_fields."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        s = _get_chunk_by_meta(chunks, "schema_name", "User")
        assert s.metadata["spec"] == "test"
        assert s.metadata["chunk_type"] == "schema"
        assert s.metadata["required_fields"] == "id,name"

    def test_schema_text_has_properties(self) -> None:
        """Text lists properties with required markers."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        s = _get_chunk_by_meta(chunks, "schema_name", "User")
        text = s.page_content
        assert "Schema: User" in text
        assert "id" in text
        assert "REQUIRED" in text
        assert "email" in text

    def test_nullable_3_1_style(self) -> None:
        """3.1 type: ["string", "null"] shows nullable."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        s = _get_chunk_by_meta(chunks, "schema_name", "User")
        assert "nullable" in s.page_content

    def test_nullable_3_0_style(self) -> None:
        """3.0 nullable: true shows nullable."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        s = _get_chunk_by_meta(
            chunks, "schema_name", "UserCreate"
        )
        assert "nullable" in s.page_content

    def test_default_value_shown(self) -> None:
        """expires_in shows default=1800."""
        spec = _spec_with_nullable_default()
        chunks = chunk_spec("test", spec)
        s = _get_chunk_by_meta(
            chunks, "schema_name", "ConnectSessionCreate"
        )
        text = s.page_content
        assert "default=1800" in text
        assert "expires_in" in text
        assert "seconds" in text


# ---------------------------------------------------------------------------
# Tests: auth chunks
# ---------------------------------------------------------------------------


class TestAuthChunks:
    """Test auth chunk formatting."""

    def test_auth_chunk_text(self) -> None:
        """Auth chunk has title, URL, scheme details."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        auth = _get_chunks_by_type(chunks, "auth")[0]
        text = auth.page_content
        assert "Spec: test" in text
        assert "Title: Test API" in text
        assert "https://api.example.com" in text
        assert "Basic Authentication" in text
        assert "API key" in text
        assert "All endpoints" in text

    def test_auth_chunk_has_citation_marker(self) -> None:
        """Auth chunk contains a Citation line for consistent LLM citing."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        auth = _get_chunks_by_type(chunks, "auth")[0]
        assert "Citation: [test: auth]" in auth.page_content

    def test_auth_metadata(self) -> None:
        """Auth metadata is minimal: spec + chunk_type."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        auth = _get_chunks_by_type(chunks, "auth")[0]
        assert auth.metadata == {
            "spec": "test",
            "chunk_type": "auth",
        }

    def test_auth_chunk_no_categories(self) -> None:
        """Auth chunk must NOT contain Available API categories."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        auth = _get_chunks_by_type(chunks, "auth")[0]
        assert "Available API categories" not in auth.page_content


# ---------------------------------------------------------------------------
# Tests: overview chunks
# ---------------------------------------------------------------------------


class TestOverviewChunks:
    """Test overview chunk formatting."""

    def test_one_overview_chunk_per_spec(self) -> None:
        """Exactly one overview chunk per spec."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        overviews = _get_chunks_by_type(chunks, "overview")
        assert len(overviews) == 1

    def test_overview_metadata(self) -> None:
        """Overview metadata has spec and chunk_type."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        ov = _get_chunks_by_type(chunks, "overview")[0]
        assert ov.metadata == {
            "spec": "test",
            "chunk_type": "overview",
        }

    def test_overview_has_citation_marker(self) -> None:
        """Overview chunk has its own citation marker."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        ov = _get_chunks_by_type(chunks, "overview")[0]
        assert "Citation: [test: overview]" in ov.page_content

    def test_overview_has_categories(self) -> None:
        """Overview chunk lists API categories from tags."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        ov = _get_chunks_by_type(chunks, "overview")[0]
        text = ov.page_content
        assert "Available API categories" in text
        assert "Users" in text
        assert "Accounts" in text

    def test_overview_has_title_and_url(self) -> None:
        """Overview chunk contains title and base URL."""
        chunks = chunk_spec("test", MINIMAL_SPEC)
        ov = _get_chunks_by_type(chunks, "overview")[0]
        text = ov.page_content
        assert "Title: Test API" in text
        assert "https://api.example.com" in text


# ---------------------------------------------------------------------------
# Tests: boilerplate schema filtering
# ---------------------------------------------------------------------------


class TestBoilerplateFiltering:
    """Test that boilerplate schemas are skipped."""

    @pytest.mark.parametrize(
        "name",
        [
            "BadRequestResponse",
            "UnauthorizedResponse",
            "ForbiddenResponse",
            "NotFoundResponse",
            "InternalServerErrorResponse",
            "TooManyRequestsResponse",
            "NotImplementedResponse",
            "BadGatewayResponse",
        ],
    )
    def test_known_boilerplate_filtered(
        self, name: str,
    ) -> None:
        """Known error response schemas are filtered."""
        assert _is_boilerplate_schema(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "User",
            "Employee",
            "ConnectSessionCreate",
            "LinkedAccount",
            "Course",
        ],
    )
    def test_domain_schemas_not_filtered(
        self, name: str,
    ) -> None:
        """Domain schemas are not filtered."""
        assert _is_boilerplate_schema(name) is False


# ---------------------------------------------------------------------------
# Tests: circular $ref handling
# ---------------------------------------------------------------------------


class TestCircularRefs:
    """Test that circular references don't crash."""

    def test_circular_ref_does_not_crash(self) -> None:
        """Circular ref produces chunks without error."""
        spec = _spec_with_circular_ref()
        chunks = chunk_spec("test", spec)
        assert len(chunks) > 0

    def test_circular_ref_noted_in_text(self) -> None:
        """Circular ref is noted in schema text."""
        spec = _spec_with_circular_ref()
        chunks = chunk_spec("test", spec)
        s = _get_chunk_by_meta(
            chunks, "schema_name", "Employee"
        )
        text = s.page_content
        assert "manager" in text
        assert "circular" in text.lower()
