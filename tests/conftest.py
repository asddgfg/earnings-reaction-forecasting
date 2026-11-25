"""Shared safety fixtures: unit tests have no permission to use the network."""

import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any accidental real connection; API tests must use local fake clients."""

    def blocked(*args, **kwargs):
        """Explain why an outbound connection is not a valid unit-test operation."""
        raise RuntimeError("Network access is disabled in the maintained test suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def no_live_credentials(monkeypatch):
    """Remove common API credential variables from each test's temporary process state."""
    for name in (
        "OPENAI_API_KEY",
        "EODHD_API_KEY",
        "EODHD_API_TOKEN",
        "ALPHA_VANTAGE_API_KEY",
        "BENZINGA_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
