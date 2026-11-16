"""Shared fixtures: one fixture world and one trained artefact bundle per test session."""

import socket
import threading
import time
from contextlib import contextmanager

import pytest
import uvicorn

from api.main import create_app
from exportspredictor import artefacts as store
from exportspredictor.fixture import build_world
from exportspredictor.pipeline import build_artefacts

FIXTURE_VERSION = "fixture-test"


@pytest.fixture(scope="session")
def fixture_world():
    return build_world()


@pytest.fixture(scope="session")
def fixture_artefacts(fixture_world):
    return build_artefacts(fixture_world, version=FIXTURE_VERSION)


@pytest.fixture(scope="session")
def fixture_artefacts_dir(fixture_artefacts, tmp_path_factory):
    """A folder holding one saved artefact version, as the API would find it."""
    root = tmp_path_factory.mktemp("artefacts")
    store.save(fixture_artefacts, root)
    return root


@pytest.fixture
def live_api():
    """Start the API on a free local port over an artefact root: `with live_api(root) as url:`."""

    @contextmanager
    def serve(root):
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(create_app(root), host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        while not server.started and thread.is_alive():
            time.sleep(0.05)
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=5)

    return serve


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
