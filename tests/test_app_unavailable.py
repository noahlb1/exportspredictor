"""When the API is down or not ready, every app page says so plainly instead of failing."""

import socket
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = [APP_DIR / "Home.py", *sorted((APP_DIR / "pages").glob("*.py"))]


def _unreachable_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


def _errors(page_path: Path) -> str:
    page = AppTest.from_file(str(page_path), default_timeout=30)
    page.run()
    assert not page.exception
    return " ".join(element.value for element in page.error)


@pytest.mark.parametrize("page_path", PAGES, ids=lambda path: path.stem)
def test_api_down_shows_how_to_start_it(monkeypatch, page_path):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", _unreachable_url())

    errors = _errors(page_path)

    assert "API is unavailable" in errors
    assert "podman compose up" in errors


@pytest.mark.parametrize("page_path", PAGES, ids=lambda path: path.stem)
def test_api_without_artefacts_says_they_are_not_loaded(monkeypatch, live_api, tmp_path, page_path):
    with live_api(tmp_path) as url:
        monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", url)
        errors = _errors(page_path)

    assert "artefacts are not loaded" in errors
