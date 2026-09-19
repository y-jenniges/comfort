"""
Pytest configuration .

Integration tests that require a real COMFORT SQLite database are marked with
@pytest.mark.integration. They are skipped unless the environment variable
COMFORT_DB_PATH is set.

Tests that require a real bathymetry NetCDF file (but no COMFORT database) are
marked with @pytest.mark.needs_bathymetry. They are skipped unless the
environment variable BATHYMETRY_PATH is set. The two markers are independent:
a test needing both would carry both markers.
"""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

_GATED_MARKERS = {
    "integration": ("COMFORT_DB_PATH", "COMFORT_DB_PATH not set (integration test)"),
    "needs_bathymetry": ("BATHYMETRY_PATH", "BATHYMETRY_PATH not set (needs_bathymetry test)"),
}


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with pytest.mark.integration/needs_bathymetry if their
    required environment variable is not set."""
    for item in items:
        for marker, (env_var, reason) in _GATED_MARKERS.items():
            if marker in item.keywords and not os.environ.get(env_var):
                item.add_marker(pytest.mark.skip(reason=reason))
