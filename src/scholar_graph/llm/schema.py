"""Pydantic -> Anthropic structured-output JSON Schema.

The structured-outputs API accepts a strict subset of JSON Schema: every
object needs ``additionalProperties: false`` and a ``required`` entry for
every property, and numeric/string constraints are rejected. Pydantic emits
neither of those conventions and does emit those constraints, so this module
translates between the two.

Doing it here rather than hand-writing schemas per call site means the
pydantic model stays the single source of truth for a node's output shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# Constraints the structured-output compiler rejects. They are dropped from the
# wire schema and re-applied client-side by pydantic on parse, so validation is
# not lost — only the server-side enforcement of it.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "default",
    }
)

_SUPPORTED_FORMATS = frozenset(
    {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    }
)


def _normalise(node: Any) -> Any:
    if isinstance(node, list):
        return [_normalise(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_KEYWORDS:
            continue
        if key == "format" and value not in _SUPPORTED_FORMATS:
            continue
        out[key] = _normalise(value)

    if out.get("type") == "object" or "properties" in out:
        properties = out.get("properties", {})
        out["additionalProperties"] = False
        # Structured outputs require every declared property to be required.
        # Optionality is expressed as a nullable type instead.
        out["required"] = sorted(properties)
    return out


def to_output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a schema for ``model`` that the structured-output API accepts."""
    normalised: dict[str, Any] = _normalise(model.model_json_schema())
    return normalised
