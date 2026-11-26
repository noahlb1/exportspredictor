"""The app's only route to the model: HTTP calls to the API.

The API base URL comes from `EXPORTSPREDICTOR_API_URL` (default http://localhost:8000).
"""

import os

import httpx

DEFAULT_API_URL = "http://localhost:8000"
STARTING = "The API is running but the model is still loading."  # the API answers 503 until its artefacts load


class ApiUnavailable(RuntimeError):
    """The API could not be reached or is not ready."""


class NotFound(LookupError):
    """The API answered 404, e.g. an unknown Product."""


def api_url() -> str:
    return os.environ.get("EXPORTSPREDICTOR_API_URL", DEFAULT_API_URL).rstrip("/")


def get(path: str, **params) -> dict:
    """GET a JSON resource from the API, mapping failures to ApiUnavailable / NotFound."""
    try:
        response = httpx.get(f"{api_url()}{path}", params=params or None, timeout=30)
    except httpx.TransportError as error:
        raise ApiUnavailable(f"Cannot reach the API at {api_url()}.") from error
    if response.status_code == 404:
        raise NotFound(response.json().get("detail", "Not found"))
    if response.status_code == 503:
        raise ApiUnavailable(STARTING)
    response.raise_for_status()
    return response.json()


class InvalidRequest(ValueError):
    """The API answered 422: the request (e.g. a Scenario) is invalid."""


def post(path: str, body: dict) -> dict:
    """POST JSON to the API, mapping failures to ApiUnavailable / InvalidRequest."""
    try:
        response = httpx.post(f"{api_url()}{path}", json=body, timeout=60)
    except httpx.TransportError as error:
        raise ApiUnavailable(f"Cannot reach the API at {api_url()}.") from error
    if response.status_code == 422:
        detail = response.json().get("detail", "Invalid request")
        raise InvalidRequest(detail if isinstance(detail, str) else "; ".join(d.get("msg", "") for d in detail))
    if response.status_code == 503:
        raise ApiUnavailable(STARTING)
    response.raise_for_status()
    return response.json()
