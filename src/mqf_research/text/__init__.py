"""Read-only, maintained interfaces for the project's frozen text artifacts.

This package does not import a model SDK, download an encoder, call an LLM, or
change extraction policies. Historical extraction entry points remain frozen.
"""

from .citations import response_schema, validate_response
from .packets import enrich_context, serialize_packet, verify_cached_identity
from .release import load_repaired_inputs, validate_release

__all__ = [
    "enrich_context",
    "load_repaired_inputs",
    "response_schema",
    "serialize_packet",
    "validate_release",
    "validate_response",
    "verify_cached_identity",
]
