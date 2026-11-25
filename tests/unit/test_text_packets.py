"""Synthetic packet/citation tests: no production data, SDK, network or LLM."""

import copy
import hashlib
import json

import jsonschema
import pytest

from mqf_research.text.citations import response_schema, validate_response
from mqf_research.text.packets import (
    canonical_digest,
    chunk_rows,
    enrich_context,
    fragment_lines,
    serialize_packet,
    verify_cached_identity,
)


@pytest.fixture
def packet():
    """Construct an entirely synthetic event with current and historical text."""
    return {
        "event_id": "EXAMPLE:2024-01-02:2023-12-31",
        "packet_id": "EXAMPLE__0",
        "information_cutoff_utc": "2024-01-02T14:25:00+00:00",
        "current_lines": fragment_lines(
            [
                "Example is a company providing industrial services.",
                "Revenue guidance declined.",
                "5.0",
            ],
            "C_D001",
        ),
        "history_lines": fragment_lines(["Previous guidance was unchanged."], "H_D001"),
        "source_catalogue": {
            "C_D001": {"available_at_utc": "2024-01-02T12:00:00+00:00"},
            "H_D001": {"available_at_utc": "2023-10-02T12:00:00+00:00"},
        },
    }


def _result(packet):
    """Create a structurally valid response citing a supplied current line."""
    properties = response_schema()["properties"]["signals"]["items"]["properties"]
    signal = {
        field: "Evidence-backed description"
        for field, specification in properties.items()
        if specification.get("type") == "string"
    }
    signal.update(
        tag="reduced_guidance",
        novelty="changed",
        statement_status="management_forecast",
        business_effect="deteriorating",
        next_session_relevance="direct",
        evidence=[
            {
                "source_id": "C_D001",
                "start_line": 2,
                "end_line": 2,
                "quote": "Revenue guidance declined.",
            }
        ],
    )
    return {
        "event_id": packet["event_id"],
        "packet_id": packet["packet_id"],
        "signals": [signal],
        "no_signal_reason": None,
    }


def test_exact_quote_is_accepted_without_mutation(packet):
    """Successful validation preserves raw text, signal values and list order."""
    result = _result(packet)
    before = copy.deepcopy((packet, result))
    assert validate_response(result, packet) is result
    assert (packet, result) == before


@pytest.mark.parametrize("change", ["invented", "identity", "history_only", "missing_line"])
def test_unverifiable_signal_is_rejected(packet, change):
    """Reject the common fabrication, cross-event and unavailable-evidence cases."""
    result = _result(packet)
    evidence = result["signals"][0]["evidence"][0]
    if change == "invented":
        evidence["quote"] = "Revenue guidance increased."
    elif change == "identity":
        result["event_id"] = "ANOTHER_EVENT"
    elif change == "history_only":
        evidence.update(
            source_id="H_D001", start_line=1, end_line=1, quote="Previous guidance was unchanged."
        )
    else:
        evidence["end_line"] = 100
    with pytest.raises(ValueError):
        validate_response(result, packet)


def test_disconnected_fragments_cannot_fabricate_a_contiguous_quote(packet):
    """A retrieval gap must remain a gap even when its endpoints look plausible."""
    packet["current_lines"] = [
        {
            "source_id": "C_D001",
            "line": 2,
            "fragment_start": 0,
            "line_length": 100,
            "text": "Revenue ",
        },
        {
            "source_id": "C_D001",
            "line": 2,
            "fragment_start": 80,
            "line_length": 100,
            "text": "guidance declined.",
        },
    ]
    with pytest.raises(ValueError, match="contiguous"):
        validate_response(_result(packet), packet)


def test_short_numeric_quote_needs_same_source_context(packet):
    """A table number alone is insufficient; its own source must supply context."""
    result = _result(packet)
    numeric = {"source_id": "C_D001", "start_line": 3, "end_line": 3, "quote": "5.0"}
    result["signals"][0]["evidence"].append(numeric)
    validate_response(result, packet)
    result["signals"][0]["evidence"] = [
        numeric,
        {
            "source_id": "H_D001",
            "start_line": 1,
            "end_line": 1,
            "quote": "Previous guidance was unchanged.",
        },
    ]
    with pytest.raises(ValueError, match="same-source"):
        validate_response(result, packet)


def test_short_numeric_substring_is_not_a_complete_line(packet):
    """A number selected from a longer line cannot bypass the short-cell guard."""
    packet["current_lines"][2]["text"] = "15.0"
    packet["current_lines"][2]["line_length"] = 4
    result = _result(packet)
    result["signals"][0]["evidence"].append(
        {
            "source_id": "C_D001",
            "start_line": 3,
            "end_line": 3,
            "quote": "5.0",
        }
    )
    with pytest.raises(ValueError, match="complete supplied line"):
        validate_response(result, packet)


def test_empty_signal_list_needs_reason_and_extra_fields_are_rejected(packet):
    """Preserve the variable-length contract while making empty responses explicit."""
    result = _result(packet)
    result["signals"] = []
    with pytest.raises(ValueError, match="requires a reason"):
        validate_response(result, packet)
    result["no_signal_reason"] = "Only previously known financial totals were provided."
    validate_response(result, packet)
    result["predicted_return"] = 0.1
    with pytest.raises(jsonschema.ValidationError):
        validate_response(result, packet)


def test_fragmenting_and_chunking_preserve_full_unicode_source():
    """Long Unicode lines, empty lines and final fragments survive packetization."""
    lines = ["財報 guidance " * 3000, "", "Final meaningful line."]
    rows = fragment_lines(lines, "C_D001", max_chars=97)
    flattened = [row for chunk in chunk_rows(rows, max_chars=900) for row in chunk]
    assert flattened == rows
    for number, original in enumerate(lines, 1):
        assert "".join(row["text"] for row in flattened if row["line"] == number) == original


def test_transport_limit_counts_bytes_and_never_truncates():
    """UTF-8 payload size, rather than character count, determines the guard."""
    packet = {"text": "財報" * 20}
    serialized = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    byte_count = len(serialized.encode("utf-8"))
    assert serialize_packet(packet, byte_count) == serialized
    with pytest.raises(ValueError, match="byte guard"):
        serialize_packet(packet, byte_count - 1)
    assert packet["text"] == "財報" * 20


def test_context_uses_only_same_packet_and_returns_independent_copy(packet):
    """Enrichment adds an exact pointer without inventing attributes or sharing mutable rows."""
    before = copy.deepcopy(packet)
    enriched = enrich_context(packet)
    evidence = enriched["company_context"]["business_description_evidence"]
    assert evidence[0]["quote"] == packet["current_lines"][0]["text"]
    enriched["current_lines"][0]["text"] = "Modified caller copy"
    assert packet == before


@pytest.mark.parametrize("timestamp", ["2099-01-01T00:00:00+00:00", "2024-01-02T12:00:00"])
def test_context_rejects_future_or_ambiguous_availability(packet, timestamp):
    """Future evidence and timezone-free timestamps fail closed."""
    packet["source_catalogue"]["C_D001"]["available_at_utc"] = timestamp
    with pytest.raises(ValueError):
        enrich_context(packet)


def test_optional_context_never_displaces_original_full_source(packet):
    """When only the original packet fits, return it whole without context."""
    size = len(serialize_packet(packet).encode("utf-8"))
    assert enrich_context(packet, size) == packet
    assert "company_context" not in packet


def test_cache_identity_keeps_legacy_prompt_and_rejects_relabeling(packet):
    """An old response remains old even after the active prompt changes."""
    old = {"model": "legacy-model", "prompt_sha256": "old"}
    active = {"model": "new-model", "prompt_sha256": "new"}
    policy = {"legacy_runtimes": [old], "active_runtime": active}
    result = _result(packet)
    envelope = {
        "runtime": old,
        "packet_sha256": canonical_digest(packet),
        "request_sha256": canonical_digest({"packet": packet, "runtime": old}),
        "result": result,
    }
    before = copy.deepcopy(envelope)
    verify_cached_identity(envelope, packet, policy)
    assert envelope == before
    envelope["runtime"] = active
    with pytest.raises(ValueError, match="Cached input changed"):
        verify_cached_identity(envelope, packet, policy)


def test_hash_matches_historical_json_spacing():
    """Canonical hashes retain historical spacing rather than a new compact encoding."""
    value = {"b": 2, "a": "財報"}
    expected = hashlib.sha256('{"a": "財報", "b": 2}'.encode("utf-8")).hexdigest()
    assert canonical_digest(value) == expected
