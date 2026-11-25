"""Pure packet helpers preserving full text and historical cache identities."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime
from typing import Any

from .citations import validate_response

DEFAULT_MAX_INPUT_BYTES = 180_000
CONTEXT_VERSION = "same-packet-business-description-v1"
_DESCRIPTION = re.compile(
    r"\b(?:is (?:a|an|the) .{0,180}(?:company|provider|manufacturer|designer)"
    r"|provid(?:es?|ing) .{0,180}(?:services|benefits|solutions)"
    r"|(?:designs|design|manufacture|manufacturer|designer).{0,180}"
    r"(?:products|systems|apparel|footwear|solutions)"
    r"|industry leader in|dedicated to .{0,180}(?:services|systems|solutions))\b",
    re.IGNORECASE,
)
_ABOUT = re.compile(r"^about\s+[^:]{1,100}:?$", re.IGNORECASE)


def canonical_digest(value: Any) -> str:
    """Hash JSON using the historical cache serialization, including spacing.

    Compact separators would produce different hashes for identical Python
    values and invalidate preserved extraction envelopes, so they are not used.
    """
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serialize_packet(packet: dict[str, Any], max_bytes: int = DEFAULT_MAX_INPUT_BYTES) -> str:
    """Serialize all packet text or fail before a request; never truncate.

    The limit counts UTF-8 bytes, not Python characters. This function validates
    local serialization only and does not claim that an SDK delivered a prompt.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    text = json.dumps(packet, ensure_ascii=False, allow_nan=False)
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("Input exceeds transport byte guard; repartition in a new version")
    return text


def fragment_lines(lines: list[str], source_id: str, max_chars: int = 16_000) -> list[dict]:
    """Split long lines into lossless fragments retaining original line numbers."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    rows = []
    for number, text in enumerate(lines, 1):
        for start in range(0, max(1, len(text)), max_chars):
            rows.append(
                {
                    "source_id": source_id,
                    "line": number,
                    "fragment_start": start,
                    "line_length": len(text),
                    "text": text[start : start + max_chars],
                }
            )
    return rows


def chunk_rows(rows: list[dict], max_chars: int = 64_000) -> list[list[dict]]:
    """Group fragments stably without dropping or splitting them again.

    Historical chunking reserves 60 characters of metadata per fragment. A
    single oversized fragment remains intact in its own chunk; final serialized
    byte-limit validation must still run before any request.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: list[list[dict]] = []
    current: list[dict] = []
    size = 0
    for row in rows:
        added = len(row["text"]) + 60
        if current and size + added > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(row)
        size += added
    if current:
        chunks.append(current)
    return chunks


def _aware_timestamp(value: str) -> datetime:
    """Parse an explicit-offset timestamp; reject ambiguous local clock values."""
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.utcoffset() is None:
        raise ValueError("Source and cutoff timestamps must contain a timezone")
    return stamp


def enrich_context(packet: dict[str, Any], max_bytes: int = DEFAULT_MAX_INPUT_BYTES) -> dict:
    """Add at most two exact business-description pointers from the same packet.

    Existing text has priority over optional context. No sector, expectation,
    price, or remembered company attribute is inferred. The result is a deep
    copy, including when no room remains for the optional context structure.
    """
    if "company_context" in packet:
        raise ValueError("Unexpected pre-existing company_context")
    serialize_packet(packet, max_bytes)
    result = copy.deepcopy(packet)
    rows = result["current_lines"]
    cutoff = _aware_timestamp(result["information_cutoff_utc"])
    candidates: list[dict] = []
    used: set[tuple[str, int]] = set()
    for position, row in enumerate(rows):
        text = row["text"].strip()
        if (
            not text
            or row.get("fragment_start", 0) != 0
            or row.get("line_length", len(row["text"])) != len(row["text"])
        ):
            continue
        block = [row]
        if _ABOUT.match(text):
            for following in rows[position + 1 : position + 7]:
                if (
                    following["source_id"] != row["source_id"]
                    or following["line"] != block[-1]["line"] + 1
                    or following.get("fragment_start", 0) != 0
                    or following.get("line_length", len(following["text"]))
                    != len(following["text"])
                ):
                    break
                if sum(len(item["text"]) for item in block) + len(following["text"]) > 1_800:
                    break
                if re.match(
                    r"^(?:forward.looking|safe harbor|contacts?|financial|condensed|consolidated)",
                    following["text"].strip(),
                    re.IGNORECASE,
                ):
                    break
                block.append(following)
            if not _DESCRIPTION.search(" ".join(item["text"] for item in block)):
                continue
        elif not _DESCRIPTION.search(text) or len(text) > 1_800:
            continue
        keys = {(item["source_id"], item["line"]) for item in block}
        if keys & used:
            continue
        source_id = row["source_id"]
        available = _aware_timestamp(result["source_catalogue"][source_id]["available_at_utc"])
        if available > cutoff:
            raise ValueError("Company-context source is later than the packet cutoff")
        candidates.append(
            {
                "source_id": source_id,
                "start_line": block[0]["line"],
                "end_line": block[-1]["line"],
                "quote": " ".join(item["text"] for item in block),
            }
        )
        used |= keys
        if len(candidates) == 2:
            break
    result["company_context"] = {
        "business_description_evidence": candidates,
        "scope": "Exact excerpts already present in this packet; not new evidence or a new signal by themselves.",
        "unverified_attributes": [
            "market-assigned growth category",
            "investor expectations",
            "valuation multiple",
            "reaction sensitivity",
        ],
        "missing_context_rule": "Use only supplied evidence; do not fill gaps with remembered company knowledge.",
    }
    # Drop optional pointers before considering a shell-only fallback. Never
    # remove any current/history source lines to make the enriched packet fit.
    while True:
        try:
            serialize_packet(result, max_bytes)
            return result
        except ValueError:
            if candidates:
                candidates.pop()
            else:
                return copy.deepcopy(packet)


def verify_cached_identity(envelope: dict, raw_packet: dict, policy: dict) -> None:
    """Check the actual cached runtime and packet identity without relabeling.

    Legacy packets were extracted without context enrichment. Applying the new
    prompt label to an old response would invalidate its request digest even if
    the signal text happened to be unchanged. No result is rewritten here.
    """
    runtime = envelope["runtime"]
    if runtime in policy["legacy_runtimes"]:
        packet = raw_packet
    elif runtime == policy["active_runtime"]:
        packet = enrich_context(raw_packet)
    else:
        raise ValueError("Unapproved cached extraction runtime")
    if envelope["packet_sha256"] != canonical_digest(packet):
        raise ValueError("Cached input changed")
    request = {"packet": packet, "runtime": runtime}
    if envelope["request_sha256"] != canonical_digest(request):
        raise ValueError("Cached request identity changed")
    validate_response(envelope["result"], packet)
    reason = (envelope["result"].get("no_signal_reason") or "").lower()
    if "truncat" in reason or "omitted from the supplied" in reason:
        raise ValueError("Cached response reports incomplete input")
