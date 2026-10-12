import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit external"
        " data sources (TFDS/OpenML).",
    )
    parser.addoption(
        "--freshness",
        action="store_true",
        default=False,
        help="Run freshness tests that regenerate LaTeX tables from"
        " the latest parquet results and diff against committed files.",
    )
    parser.addoption(
        "--reproducibility",
        action="store_true",
        default=False,
        help="Run reproducibility tests that train a real model twice on"
        " synthetic data and assert identical outputs.",
    )
    parser.addoption(
        "--provenance",
        action="store_true",
        default=False,
        help="Run the test that checks the git metadata embedded in the"
        " result parquets against the current code state.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require network access to TFDS/OpenML",
    )
    config.addinivalue_line(
        "markers",
        "freshness: marks tests that regenerate tables and compare against"
        " the committed copies in tex/tables/",
    )
    config.addinivalue_line(
        "markers",
        "reproducibility: marks end-to-end determinism tests (slower than"
        " unit tests; require --reproducibility to run)",
    )
    config.addinivalue_line(
        "markers",
        "provenance: marks tests that check the git metadata of the result"
        " parquets (require --provenance to run)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--integration"):
        skip_int = pytest.mark.skip(reason="pass --integration to run")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip_int)
    if not config.getoption("--freshness"):
        skip_fresh = pytest.mark.skip(reason="pass --freshness to run")
        for item in items:
            if item.get_closest_marker("freshness"):
                item.add_marker(skip_fresh)
    if not config.getoption("--reproducibility"):
        skip_repro = pytest.mark.skip(reason="pass --reproducibility to run")
        for item in items:
            if item.get_closest_marker("reproducibility"):
                item.add_marker(skip_repro)
    if not config.getoption("--provenance"):
        skip_prov = pytest.mark.skip(reason="pass --provenance to run")
        for item in items:
            if item.get_closest_marker("provenance"):
                item.add_marker(skip_prov)
