"""Structural OpenAPI chunker: operation, schema, and auth chunks.

Produces human-readable text chunks (not raw JSON) with metadata,
designed for embedding-based retrieval over StackOne API specs.
"""

from __future__ import annotations

from typing import Any

from ai_exercise.models import Document

# Schemas that are standard HTTP error shapes — don't create chunks for these.
BOILERPLATE_SCHEMA_PATTERNS = {
    "BadRequestResponse",
    "UnauthorizedResponse",
    "ForbiddenResponse",
    "NotFoundResponse",
    "RequestTimedOutResponse",
    "ConflictResponse",
    "PreconditionFailedResponse",
    "UnprocessableEntityResponse",
    "TooManyRequestsResponse",
    "InternalServerErrorResponse",
    "NotImplementedResponse",
    "BadGatewayResponse",
}

# Only include success responses in operation chunks.
SUCCESS_STATUS_CODES = {"200", "201", "202"}

MAX_REF_DEPTH = 5


# ---------------------------------------------------------------------------
# $ref resolution
# ---------------------------------------------------------------------------


def _schema_lookup(spec: dict[str, Any]) -> dict[str, Any]:
    """Build a name -> definition lookup from components.schemas."""
    schemas: dict[str, Any] = spec.get("components", {}).get(
        "schemas", {}
    )
    return schemas


def _resolve_ref(
    ref_or_schema: dict[str, Any],
    spec: dict[str, Any],
    depth: int = 0,
    visited: set[str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve a $ref pointer, returning (resolved_name, schema_dict).

    If no $ref, returns (None, original_schema).
    Handles circular refs via visited set and depth limit.
    """
    if visited is None:
        visited = set()

    ref = ref_or_schema.get("$ref")
    if not ref or depth > MAX_REF_DEPTH:
        return None, ref_or_schema

    # Only handle local refs: #/components/schemas/Foo
    if not ref.startswith("#/components/schemas/"):
        return None, ref_or_schema

    name = ref.split("/")[-1]
    if name in visited:
        return name, {"_circular": True}

    visited.add(name)
    schemas = _schema_lookup(spec)
    definition = schemas.get(name, {})

    # The resolved schema itself may contain a $ref (allOf wrapper, etc.)
    if "$ref" in definition:
        return _resolve_ref(definition, spec, depth + 1, visited)

    return name, definition


# ---------------------------------------------------------------------------
# Type formatting
# ---------------------------------------------------------------------------


def _is_nullable(schema: dict[str, Any]) -> bool:
    """Check nullable in both OpenAPI 3.0 and 3.1 style."""
    # 3.0 style
    if schema.get("nullable"):
        return True
    # 3.1 style: type can be a list like ["string", "null"]
    t = schema.get("type")
    return bool(isinstance(t, list) and "null" in t)


def _base_type(schema: dict[str, Any]) -> str:
    """Get the base type string, handling 3.1 list-style types."""
    t = schema.get("type")
    if isinstance(t, list):
        return next((x for x in t if x != "null"), "any")
    return t or "any"


def _format_type(
    schema: dict[str, Any],
    spec: dict[str, Any],
    depth: int = 0,
    visited: set[str] | None = None,
) -> str:
    """Format a schema's type as a human-readable string."""
    if visited is None:
        visited = set()

    # Handle $ref
    if "$ref" in schema:
        name, _ = _resolve_ref(schema, spec, depth, visited.copy())
        return name or "object"

    # Handle allOf (common pattern: allOf with one $ref + nullable)
    if "allOf" in schema:
        parts = []
        for sub in schema["allOf"]:
            parts.append(_format_type(sub, spec, depth, visited.copy()))
        return " & ".join(parts)

    # Handle oneOf / anyOf
    for key in ("oneOf", "anyOf"):
        if key in schema:
            non_null = [s for s in schema[key] if s.get("type") != "null"]
            if len(non_null) == 1:
                return _format_type(non_null[0], spec, depth, visited.copy())
            parts = [_format_type(s, spec, depth, visited.copy()) for s in non_null]
            return " | ".join(parts)

    base = _base_type(schema)

    if base == "array":
        items = schema.get("items", {})
        item_type = _format_type(items, spec, depth, visited.copy())
        return f"array[{item_type}]"

    if schema.get("enum"):
        vals = " | ".join(str(v) for v in schema["enum"])
        return f"enum: {vals}"

    return base


# ---------------------------------------------------------------------------
# Property formatting
# ---------------------------------------------------------------------------


def _format_property_line(
    name: str,
    prop: dict[str, Any],
    required: bool,
    spec: dict[str, Any],
    depth: int = 0,
    visited: set[str] | None = None,
) -> str:
    """Format a single property as a human-readable line."""
    if visited is None:
        visited = set()

    type_str = _format_type(prop, spec, depth, visited.copy())
    parts = [type_str]

    if required:
        parts.append("REQUIRED")
    else:
        parts.append("optional")

    if prop.get("default") is not None:
        parts.append(f"default={prop['default']}")

    if prop.get("deprecated"):
        parts.append("DEPRECATED")

    if _is_nullable(prop):
        parts.append("nullable")

    qualifier = ", ".join(parts)
    desc = prop.get("description", "")
    line = f"- {name} ({qualifier})"
    if desc:
        line += f": {desc}"
    return line


def _format_properties_block(
    schema: dict[str, Any],
    spec: dict[str, Any],
    depth: int = 0,
    visited: set[str] | None = None,
    indent: int = 2,
    max_fields: int = 40,
) -> list[str]:
    """Format all properties of a schema as indented lines.

    For nested objects/refs, recurse up to MAX_REF_DEPTH.
    """
    if visited is None:
        visited = set()

    lines: list[str] = []
    prefix = " " * indent

    # Handle allOf by merging
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for sub in schema["allOf"]:
            if "$ref" in sub:
                _, resolved = _resolve_ref(sub, spec, depth, visited.copy())
                merged.update(resolved.get("properties", {}))
            else:
                merged.update(sub.get("properties", {}))
        schema = {**schema, "properties": merged}

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    for count, (prop_name, prop_schema) in enumerate(properties.items()):
        if count >= max_fields:
            lines.append(
                f"{prefix}- ... ({len(properties) - count} more fields)"
            )
            break

        is_required = prop_name in required_fields
        line = _format_property_line(
            prop_name, prop_schema, is_required, spec, depth, visited.copy()
        )
        lines.append(f"{prefix}{line}")

        # For $ref properties, show nested fields if depth allows
        if depth < MAX_REF_DEPTH:
            ref_name, resolved = _resolve_ref(
                prop_schema, spec, depth + 1, visited.copy()
            )
            if ref_name and not resolved.get("_circular"):
                if resolved.get("properties"):
                    nested = _format_properties_block(
                        resolved,
                        spec,
                        depth + 1,
                        visited | {ref_name},
                        indent + 4,
                        max_fields=15,
                    )
                    if nested:
                        lines.append(f"{prefix}    Each {ref_name} has:")
                        lines.extend(nested)
            elif ref_name and resolved.get("_circular"):
                lines.append(f"{prefix}    (circular reference to {ref_name})")

            # Handle array items with $ref
            if prop_schema.get("type") == "array" or (
                isinstance(prop_schema.get("type"), list)
                and "array" in prop_schema["type"]
            ):
                items = prop_schema.get("items", {})
                item_name, item_resolved = _resolve_ref(
                    items, spec, depth + 1, visited.copy()
                )
                if (
                    item_name
                    and not item_resolved.get("_circular")
                    and item_resolved.get("properties")
                ):
                    nested = _format_properties_block(
                        item_resolved,
                        spec,
                        depth + 1,
                        visited | {item_name},
                        indent + 4,
                        max_fields=15,
                    )
                    if nested:
                        lines.append(
                            f"{prefix}    Each {item_name} has:"
                        )
                        lines.extend(nested)

    return lines


# ---------------------------------------------------------------------------
# Response formatting for operation chunks
# ---------------------------------------------------------------------------


def _format_response(
    responses: dict[str, Any],
    spec: dict[str, Any],
) -> list[str]:
    """Format success response schema as readable lines."""
    lines: list[str] = []

    for code in SUCCESS_STATUS_CODES:
        resp = responses.get(code)
        if not resp:
            continue

        content = resp.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})
        if not schema:
            lines.append(f"Response {code}: (no body)")
            continue

        # Resolve top-level $ref
        ref_name, resolved = _resolve_ref(schema, spec)
        if ref_name:
            lines.append(f"Response {code}: {ref_name}")
        else:
            lines.append(f"Response {code}:")

        if resolved.get("properties"):
            prop_lines = _format_properties_block(
                resolved, spec, depth=1, visited={ref_name} if ref_name else None
            )
            lines.extend(prop_lines)
        break  # Only show the first success response

    return lines


# ---------------------------------------------------------------------------
# Parameter formatting
# ---------------------------------------------------------------------------


def _format_parameters(
    parameters: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[str]:
    """Format operation parameters as readable lines."""
    lines: list[str] = []
    for param in parameters:
        # Resolve parameter $ref (some specs use $ref for shared params)
        if "$ref" in param:
            _, param = _resolve_ref(param, spec)

        name = param.get("name", "?")
        location = param.get("in", "?")
        schema = param.get("schema", {})
        type_str = _format_type(schema, spec)
        required = param.get("required", False)
        deprecated = param.get("deprecated", False)
        desc = param.get("description", "")

        parts = [location, type_str]
        parts.append("required" if required else "optional")

        if schema.get("default") is not None:
            parts.append(f"default={schema['default']}")

        if deprecated:
            parts.append("DEPRECATED")

        if _is_nullable(schema):
            parts.append("nullable")

        qualifier = ", ".join(parts)
        line = f"  - {name} ({qualifier})"
        if desc:
            line += f": {desc}"
        lines.append(line)

    return lines


# ---------------------------------------------------------------------------
# Chunk builders
# ---------------------------------------------------------------------------


def _build_operation_chunk(
    spec_name: str,
    path: str,
    method: str,
    operation: dict[str, Any],
    spec: dict[str, Any],
    path_params: list[dict[str, Any]],
) -> Document:
    """Build a single operation chunk."""
    op_id = operation.get("operationId", "")
    tags = ", ".join(operation.get("tags", []))
    summary = operation.get("summary", "")
    description = operation.get("description", "")

    parts = [
        f"Spec: {spec_name}",
        f"Operation: {method.upper()} {path}" + (f" ({op_id})" if op_id else ""),
    ]
    if tags:
        parts.append(f"Tags: {tags}")
    if summary:
        parts.append(f"Summary: {summary}")
    if description:
        parts.append(f"Description: {description}")

    # Merge path-level and operation-level parameters
    all_params = list(path_params) + operation.get("parameters", [])
    if all_params:
        parts.append("")
        parts.append("Parameters:")
        parts.extend(_format_parameters(all_params, spec))

    # Request body
    request_body = operation.get("requestBody", {})
    if request_body:
        rb_content = request_body.get("content", {})
        rb_json = rb_content.get("application/json", {})
        rb_schema = rb_json.get("schema", {})
        if rb_schema:
            ref_name, resolved = _resolve_ref(rb_schema, spec)
            parts.append("")
            if ref_name:
                parts.append(f"Request Body: {ref_name}")
            else:
                parts.append("Request Body:")
            if resolved.get("properties"):
                prop_lines = _format_properties_block(
                    resolved,
                    spec,
                    depth=1,
                    visited={ref_name} if ref_name else None,
                )
                parts.extend(prop_lines)

    # Success response only
    responses = operation.get("responses", {})
    if responses:
        resp_lines = _format_response(responses, spec)
        if resp_lines:
            parts.append("")
            parts.extend(resp_lines)

    # Security
    security = operation.get("security") or spec.get("security", [])
    if security:
        scheme_names = []
        for req in security:
            scheme_names.extend(req.keys())
        if scheme_names:
            parts.append("")
            sec_schemes = spec.get("components", {}).get("securitySchemes", {})
            sec_descs = []
            for sname in scheme_names:
                scheme_def = sec_schemes.get(sname, {})
                scheme_type = scheme_def.get("type", "")
                scheme_scheme = scheme_def.get("scheme", "")
                if scheme_type == "http" and scheme_scheme:
                    sec_descs.append(f"HTTP {scheme_scheme.title()} Auth")
                elif scheme_type == "apiKey":
                    sec_descs.append(
                        f"API Key ({scheme_def.get('in', '')}: "
                        f"{scheme_def.get('name', '')})"
                    )
                else:
                    sec_descs.append(sname)
            parts.append(f"Security: {', '.join(sec_descs)}")

    text = "\n".join(parts)

    metadata: dict[str, Any] = {
        "spec": spec_name,
        "chunk_type": "operation",
        "path": path,
        "method": method.upper(),
    }
    if op_id:
        metadata["operation_id"] = op_id
    if tags:
        metadata["tags"] = tags

    return Document(page_content=text, metadata=metadata)


def _is_boilerplate_schema(name: str) -> bool:
    """Check if a schema name matches known boilerplate error shapes."""
    if name in BOILERPLATE_SCHEMA_PATTERNS:
        return True
    # Catch variations like "ForbiddenResponse", "ConflictResponse", etc.
    lower = name.lower()
    return lower.endswith("response") and any(
        keyword in lower
        for keyword in [
            "badrequest",
            "unauthorized",
            "forbidden",
            "notfound",
            "timedout",
            "conflict",
            "precondition",
            "unprocessable",
            "toomany",
            "internalserver",
            "notimplemented",
            "badgateway",
        ]
    )


def _build_schema_chunk(
    spec_name: str,
    schema_name: str,
    schema: dict[str, Any],
    spec: dict[str, Any],
) -> Document | None:
    """Build a schema chunk. Returns None for boilerplate schemas."""
    if _is_boilerplate_schema(schema_name):
        return None

    description = schema.get("description", "")

    parts = [
        f"Spec: {spec_name}",
        f"Schema: {schema_name}",
    ]
    if description:
        parts.append(f"Description: {description}")

    required_fields = schema.get("required", [])
    if required_fields:
        parts.append(f"\nRequired fields: {', '.join(required_fields)}")

    parts.append("\nProperties:")
    prop_lines = _format_properties_block(schema, spec, depth=0)
    if prop_lines:
        parts.extend(prop_lines)
    else:
        parts.append("  (no properties)")

    # Handle enum-only schemas (no properties, just enum values)
    if not schema.get("properties") and schema.get("enum"):
        parts.pop()  # Remove "(no properties)"
        parts.pop()  # Remove "Properties:"
        vals = " | ".join(str(v) for v in schema["enum"])
        parts.append(f"\nValues: {vals}")

    text = "\n".join(parts)

    metadata: dict[str, Any] = {
        "spec": spec_name,
        "chunk_type": "schema",
        "schema_name": schema_name,
    }
    if required_fields:
        metadata["required_fields"] = ",".join(required_fields)

    return Document(page_content=text, metadata=metadata)


def _build_auth_chunk(
    spec_name: str,
    spec: dict[str, Any],
) -> Document:
    """Build an auth chunk capturing global auth and request conventions."""
    info = spec.get("info", {})
    title = info.get("title", spec_name)

    # Base URL from servers
    servers = spec.get("servers", [])
    base_url = servers[0]["url"] if servers else "N/A"

    parts = [
        f"Spec: {spec_name}",
        f"Title: {title}",
        f"Base URL: {base_url}",
        "Authentication and global request conventions",
    ]

    # Security schemes
    sec_schemes = spec.get("components", {}).get("securitySchemes", {})
    global_security = spec.get("security", [])

    for scheme_name, scheme_def in sec_schemes.items():
        parts.append(f"\nSecurity scheme: {scheme_name}")
        scheme_type = scheme_def.get("type", "")
        scheme_scheme = scheme_def.get("scheme", "")

        if scheme_type == "http" and scheme_scheme == "basic":
            parts.append("  Type: HTTP Basic Authentication")
            parts.append(
                "  How to use: Send your StackOne API key as the HTTP username. "
                "Leave the password empty."
            )
        elif scheme_type == "http":
            parts.append(f"  Type: HTTP {scheme_scheme.title()} Authentication")
        elif scheme_type == "apiKey":
            parts.append(
                f"  Type: API Key (in {scheme_def.get('in', '')}, "
                f"name: {scheme_def.get('name', '')})"
            )
        else:
            parts.append(f"  Type: {scheme_type}")

        if scheme_def.get("description"):
            parts.append(f"  Description: {scheme_def['description']}")

        # Check if globally applied
        for req in global_security:
            if scheme_name in req:
                parts.append(
                    "  Applied to: All endpoints"
                    " (global security requirement)"
                )
                break

    # Common headers (x-account-id, etc.) — extract from path parameters
    # that appear across many operations
    common_headers = _extract_common_headers(spec)
    if common_headers:
        parts.append("\nCommon headers for unified endpoints:")
        for header in common_headers:
            name = header.get("name", "")
            desc = header.get("description", "")
            required = header.get("required", False)
            req_str = "required for most unified endpoints" if required else "optional"
            line = f"  - {name} (header, string, {req_str})"
            if desc:
                line += f":\n    {desc}"
            parts.append(line)

    # List available tags/categories
    tags = spec.get("tags", [])
    if tags:
        tag_names = [t.get("name", "") for t in tags if t.get("name")]
        if tag_names:
            parts.append(f"\nAvailable API categories: {', '.join(tag_names)}")

    text = "\n".join(parts)

    return Document(
        page_content=text,
        metadata={"spec": spec_name, "chunk_type": "auth"},
    )


def _extract_common_headers(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Find header parameters that appear across many operations."""
    header_counts: dict[str, dict[str, Any]] = {}

    for _path, path_item in spec.get("paths", {}).items():
        for method in ("get", "post", "put", "patch", "delete"):
            operation = path_item.get(method)
            if not operation:
                continue
            all_params = path_item.get("parameters", []) + operation.get(
                "parameters", []
            )
            for param in all_params:
                if param.get("in") == "header":
                    name = param.get("name", "")
                    if name not in header_counts:
                        header_counts[name] = param
                        header_counts[name]["_count"] = 0
                    header_counts[name]["_count"] = (
                        header_counts[name].get("_count", 0) + 1
                    )

    # Return headers that appear in 3+ operations
    results = []
    for _name, param in header_counts.items():
        if param.get("_count", 0) >= 3:
            clean = {k: v for k, v in param.items() if k != "_count"}
            results.append(clean)
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def chunk_spec(spec_name: str, spec: dict[str, Any]) -> list[Document]:
    """Chunk an OpenAPI spec into operation, schema, and auth chunks.

    Args:
        spec_name: Name of the spec (e.g. "stackone", "hris").
        spec: Parsed OpenAPI spec dict.

    Returns:
        List of Document objects with human-readable text and metadata.
    """
    chunks: list[Document] = []

    # 1. Auth chunk (one per spec)
    chunks.append(_build_auth_chunk(spec_name, spec))

    # 2. Operation chunks (one per path+method)
    for path, path_item in spec.get("paths", {}).items():
        path_params = path_item.get("parameters", [])
        for method in ("get", "post", "put", "patch", "delete"):
            operation = path_item.get(method)
            if not operation:
                continue
            chunks.append(
                _build_operation_chunk(
                    spec_name, path, method, operation, spec, path_params
                )
            )

    # 3. Schema chunks (one per named schema, skip boilerplate)
    schemas = _schema_lookup(spec)
    for schema_name, schema_def in schemas.items():
        doc = _build_schema_chunk(spec_name, schema_name, schema_def, spec)
        if doc is not None:
            chunks.append(doc)

    return chunks
