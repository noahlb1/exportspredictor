"""Shared fixtures: one fixture world and one trained artefact bundle per test session."""

import pytest

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
