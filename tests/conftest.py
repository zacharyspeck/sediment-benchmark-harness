"""Shared test fixtures. Everything here is hermetic -- no corpus, no network."""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from graders.fabrication import FabricationScorer  # noqa: E402
from harness.cli import load_fabrication_config, load_grading_config  # noqa: E402
from harness.ledger import load_ledger  # noqa: E402
from harness.paths import permit_root  # noqa: E402
from harness.runner import load_items  # noqa: E402

MINI_LEDGER = os.path.join(REPO, "fixtures", "mini_ledger.json")
FIXTURE_ITEMS = os.path.join(REPO, "fixtures", "fixture_items.yaml")


@pytest.fixture(scope="session", autouse=True)
def _permit_tmp(tmp_path_factory):
    """Let hermetic tests write under pytest's tmp tree.

    Without this every Runner/write_report call in the suite would be refused
    by harness.paths. Registering the tmp base here keeps the guard's default
    narrow in shipped code -- nothing outside the test suite calls permit_root.
    The adversarial tests in tests/test_paths.py deliberately use paths outside
    this tree, so they are unaffected.
    """
    permit_root(str(tmp_path_factory.getbasetemp()))


@pytest.fixture(scope="session")
def repo_root():
    return REPO


@pytest.fixture(scope="session")
def ledger():
    return load_ledger(MINI_LEDGER, "normalized")


@pytest.fixture(scope="session")
def grading_config():
    return load_grading_config()


@pytest.fixture(scope="session")
def fabrication_config():
    return load_fabrication_config()


@pytest.fixture(scope="session")
def scorer(ledger, fabrication_config):
    return FabricationScorer(ledger, fabrication_config)


@pytest.fixture(scope="session")
def items():
    return load_items(FIXTURE_ITEMS)
