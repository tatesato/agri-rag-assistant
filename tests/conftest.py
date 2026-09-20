"""Prevent accidental external network calls during the test suite."""
import socket
import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail immediately if a test attempts to open a network connection."""
    def blocked(*args, **kwargs):
        raise AssertionError("Network access is forbidden in tests.")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
