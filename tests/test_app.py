"""The Streamlit Product -> Markets page, run with AppTest against a live local API on fixture artefacts."""

import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from api.main import create_app

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGE = APP_DIR / "pages" / "1_Product_to_Markets.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def api_url(fixture_artefacts_dir):
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(fixture_artefacts_dir), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{url}/health").status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.05)
    yield url
    server.should_exit = True
    thread.join(timeout=5)


def _run_page(monkeypatch, api_url: str, hs6: str) -> AppTest:
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", api_url)
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.run()
    page.text_input(key="hs6").input(hs6).run()
    return page


def test_page_shows_markets_ranked_by_export_gap(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "220830")

    assert not page.exception
    table = page.dataframe[0].value
    assert list(table["Market"]) and set(table["Market"]) == {"Japan", "Canada", "India"}
    gaps = list(table["Export Gap (USD)"])
    assert gaps == sorted(gaps, reverse=True)


def test_page_shows_baseline_year_banner(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "220830")

    banner = " ".join(element.value for element in page.info)
    assert "2024" in banner and "provisional" in banner


def test_unknown_product_shows_a_message(monkeypatch, api_url):
    page = _run_page(monkeypatch, api_url, "999999")

    assert not page.exception
    assert any("999999" in element.value for element in page.warning)


def test_api_down_shows_a_clear_message(monkeypatch):
    monkeypatch.setenv("EXPORTSPREDICTOR_API_URL", f"http://127.0.0.1:{_free_port()}")
    page = AppTest.from_file(str(PAGE), default_timeout=30)
    page.run()

    assert not page.exception
    assert any("API" in element.value for element in page.error)
