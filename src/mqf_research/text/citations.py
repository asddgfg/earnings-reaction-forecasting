"""Validate the selective-news JSON contract and exact source citations.

Adapted from the frozen v3 schema and contextual-short-line citation guard.
Successful validation establishes structural and textual support, not whether
the financial interpretation or predicted pricing mechanism is correct.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import jsonschema


def normalize_space(text: str) -> str:
    """Collapse whitespace without changing the case or words in an excerpt."""
    return re.sub(r"\s+", " ", text).strip()


def _closed_object(properties: dict[str, Any]) -> dict[str, Any]:
    """Require every declared field and reject unexpected response fields."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def response_schema() -> dict[str, Any]:
    """Return a fresh copy of the production selective-news response schema.

    The response contains a variable-length ``signals`` list, including an
    explicitly justified empty list. It is not a fixed list of ten scores.
    """
    string = {"type": "string", "minLength": 1}
    evidence = _closed_object(
        {
            "source_id": string,
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "quote": string,
        }
    )
    fields = (
        "tag",
        "fact",
        "incremental_information",
        "pricing_mechanism",
        "business_horizon",
        "timing_reason",
        "target_window_limitation",
        "embedding_text",
    )
    properties = {name: string for name in fields}
    choices = {
        "novelty": ["changed", "repeated", "unknown"],
        "statement_status": ["realized", "management_forecast", "conditional", "mixed"],
        "business_effect": ["improving", "deteriorating", "mixed", "unclear"],
        "next_session_relevance": ["direct", "conditional", "uncertain"],
    }
    for name, values in choices.items():
        properties[name] = {"type": "string", "enum": values}
    properties["evidence"] = {"type": "array", "items": evidence, "minItems": 1}
    return _closed_object(
        {
            "event_id": string,
            "packet_id": string,
            "signals": {"type": "array", "items": _closed_object(properties)},
            "no_signal_reason": {"type": ["string", "null"]},
        }
    )


def citation_segments(
    source: Mapping[int, list[dict[str, Any]]], start_line: int, end_line: int
) -> list[str]:
    """Reassemble only visible, contiguous fragments in a citation range.

    Missing text must never become apparent evidence by concatenating the
    fragments on either side of a retrieval gap. A line break is joinable only
    after the entire previous line and before the start of the next line.
    """
    if (
        end_line < start_line
        or end_line - start_line > 500
        or not all(number in source for number in range(start_line, end_line + 1))
    ):
        raise ValueError("Citation references unavailable lines")
    parts: list[str] = []
    visible = ""
    previous: tuple[int, int, int] | None = None
    for number in range(start_line, end_line + 1):
        for row in sorted(source[number], key=lambda item: item.get("fragment_start", 0)):
            start = row.get("fragment_start", 0)
            end = start + len(row["text"])
            length = row.get("line_length", end)
            if start < 0 or end > length:
                raise ValueError("Invalid source fragment offsets")
            contiguous = (
                previous is None
                or (number == previous[0] and start == previous[1])
                or (number == previous[0] + 1 and start == 0 and previous[1] == previous[2])
            )
            if not contiguous:
                parts.append(visible)
                visible = ""
            elif previous is not None and number != previous[0]:
                visible += " "
            visible += row["text"]
            previous = (number, end, length)
    parts.append(visible)
    return parts


def _is_complete_line(source: Mapping[int, list[dict]], evidence: dict) -> bool:
    """Check whether a short quote covers one complete supplied source line."""
    if evidence["start_line"] != evidence["end_line"]:
        return False
    rows = sorted(source[evidence["start_line"]], key=lambda row: row.get("fragment_start", 0))
    cursor = 0
    text = ""
    lengths: set[int] = set()
    for row in rows:
        if row.get("fragment_start", 0) != cursor:
            return False
        cursor += len(row["text"])
        text += row["text"]
        lengths.add(row.get("line_length", cursor))
    return (
        len(lengths) == 1
        and cursor == next(iter(lengths))
        and normalize_space(text) == normalize_space(evidence["quote"])
    )


def validate_response(result: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    """Validate identities, schema and exact citations without changing inputs.

    Raises ``jsonschema.ValidationError`` for a malformed JSON response and
    ``ValueError`` for identity or evidence mismatches. Short numeric table
    cells require a complete source line plus textual context from that same
    source. Every signal must cite at least one current disclosure.
    """
    jsonschema.validate(result, response_schema())
    if any(result[field] != packet[field] for field in ("event_id", "packet_id")):
        raise ValueError("Event or packet identity mismatch")
    if not result["signals"] and not (result["no_signal_reason"] or "").strip():
        raise ValueError("An empty signal list requires a reason")
    allowed: dict[str, dict[int, list[dict]]] = {}
    for row in packet["current_lines"] + packet["history_lines"]:
        allowed.setdefault(row["source_id"], {}).setdefault(row["line"], []).append(row)
    current_ids = {row["source_id"] for row in packet["current_lines"]}
    for signal in result["signals"]:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,99}", signal["tag"]):
            raise ValueError("Tag must be a short snake_case name")
        if not any(item["source_id"] in current_ids for item in signal["evidence"]):
            raise ValueError("Signal lacks a current-source citation")
        short: list[dict] = []
        contextual_sources: set[str] = set()
        for evidence in signal["evidence"]:
            source = allowed.get(evidence["source_id"], {})
            segments = citation_segments(source, evidence["start_line"], evidence["end_line"])
            quote = normalize_space(evidence["quote"])
            if not quote or not any(quote in normalize_space(part) for part in segments):
                raise ValueError("Quote is not an exact contiguous excerpt")
            if len(quote) < 8:
                if not _is_complete_line(source, evidence):
                    raise ValueError("A short citation must match a complete supplied line")
                short.append(evidence)
            elif sum(character.isalpha() for character in quote) >= 3:
                contextual_sources.add(evidence["source_id"])
        if any(item["source_id"] not in contextual_sources for item in short):
            raise ValueError("Short citation lacks same-source supporting text")
    return result
