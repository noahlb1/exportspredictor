"""The one-command run: compose.yaml wires the API and app together, and fetches the artefact.

podman is not run in the test suite, so these tests check the compose file as data, run the API
healthcheck command against a live local API and run the fetch command on a `file://` release.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import FIXTURE_VERSION

from exportspredictor import artefacts as store
from exportspredictor import release
from exportspredictor.paths import artefacts_dir

REPO = Path(__file__).resolve().parents[1]
COMPOSE = REPO / "compose.yaml"


@pytest.fixture(scope="module")
def services() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]


def test_compose_has_an_api_an_app_and_a_fetch_service(services):
    assert set(services) == {"api", "app", "fetch"}


def test_compose_up_alone_does_not_start_the_fetch(services):
    assert services["fetch"]["profiles"] == ["fetch"]
    assert "profiles" not in services["api"]
    assert "profiles" not in services["app"]


def test_fetch_writes_into_the_folder_the_api_serves(services):
    fetch, api = services["fetch"], services["api"]
    target = fetch["environment"]["EXPORTSPREDICTOR_ARTEFACTS_DIR"]
    assert target == api["environment"]["EXPORTSPREDICTOR_ARTEFACTS_DIR"]
    assert f"./artefacts:{target}" in fetch["volumes"]
    assert fetch["image"] == api["image"]


def test_a_fresh_clone_has_the_artefacts_folder_the_services_mount_and_tracks_nothing_in_it(services):
    """podman refuses to start a container whose bind-mount source is missing, so the folder must come
    with the clone; the artefact versions fetched into it must stay untracked."""
    sources = {volume.split(":", 1)[0] for service in services.values() for volume in service.get("volumes", [])}
    assert sources == {"./artefacts"}
    tracked = subprocess.run(
        ["git", "ls-files", "artefacts"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    assert tracked == ["artefacts/.gitkeep"]
    version = f"artefacts/{release.RELEASED_VERSION}/{store.METADATA_FILE}"
    assert subprocess.run(["git", "check-ignore", "-q", "--no-index", version], cwd=REPO, check=False).returncode == 0


def test_the_fetch_command_unpacks_a_release_into_the_artefacts_folder(services, fixture_artefacts_dir, tmp_path):
    """The fetch service's command, run here instead of in the container, with `--url` appended as
    `compose run --rm fetch --url URL` appends it."""
    command = services["fetch"]["entrypoint"]
    assert command[0] == "python"
    packed = release.pack(fixture_artefacts_dir, tmp_path / "release")
    root = tmp_path / "artefacts"
    environment = {**os.environ, "EXPORTSPREDICTOR_ARTEFACTS_DIR": str(root)}

    subprocess.run([sys.executable, *command[1:], "--url", packed.as_uri()], env=environment, timeout=120, check=True)

    assert (root / FIXTURE_VERSION / store.METADATA_FILE).exists()


def test_app_starts_only_once_the_api_is_healthy(services):
    assert services["app"]["depends_on"] == {"api": {"condition": "service_healthy"}}
    assert "healthcheck" in services["api"]


def test_app_reaches_the_api_by_its_service_name(services):
    assert services["app"]["environment"]["EXPORTSPREDICTOR_API_URL"] == "http://api:8000"


def test_artefact_folder_is_mounted_where_the_api_reads_it(services):
    api = services["api"]
    target = api["environment"]["EXPORTSPREDICTOR_ARTEFACTS_DIR"]
    assert f"./artefacts:{target}:ro" in api["volumes"]


def test_services_build_from_the_containerfile(services):
    for service in services.values():
        assert service["build"]["dockerfile"] == "Containerfile"
    assert (REPO / "Containerfile").exists()


def test_advisor_reaches_the_app_on_port_8501(services):
    assert "8501:8501" in services["app"]["ports"]


def _healthcheck(services, api_url: str) -> int:
    command = services["api"]["healthcheck"]["test"]
    assert command[:2] == ["CMD", "python"]
    script = [part.replace("http://localhost:8000", api_url) for part in command[2:]]
    return subprocess.run([sys.executable, *script], capture_output=True, timeout=30, check=False).returncode


def test_healthcheck_passes_when_artefacts_are_loaded(services, live_api, fixture_artefacts_dir):
    with live_api(fixture_artefacts_dir) as url:
        assert _healthcheck(services, url) == 0


def test_healthcheck_fails_while_no_artefacts_are_loaded(services, live_api, tmp_path):
    with live_api(tmp_path) as url:
        assert _healthcheck(services, url) != 0


def test_healthcheck_passes_once_the_real_v2_artefacts_are_loaded(services, live_api):
    """The mounted `./artefacts` folder holds the real v2 artefacts (skipped without them)."""
    root = artefacts_dir()
    try:
        bundle = store.load_latest(root)
    except (store.ArtefactsNotFound, store.UnreadableArtefacts) as error:
        pytest.skip(f"No real format-2 artefacts under {root}: {error}")
    if bundle.metadata.get("source", "fixture") == "fixture":
        pytest.skip("The artefacts are fixture artefacts")

    with live_api(root) as url:
        assert _healthcheck(services, url) == 0


def test_the_app_never_resolves_the_api_through_the_host_dns_search_domains(services):
    """While the API container is stopped, a bare `api` must not fall through to the host's search
    domains (where it can resolve to a public host and hang the page); it must fail fast."""
    assert services["app"]["dns_search"] == ["dns.podman"]
