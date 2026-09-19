"""
Pytest configuration .

Tests that require a real bathymetry NetCDF file are marked with
@pytest.mark.needs_bathymetry. They are skipped unless the environment
variable BATHYMETRY_PATH is set.
"""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with pytest.mark.needs_bathymetry if BATHYMETRY_PATH is not set."""
    skip_needs_bathymetry = pytest.mark.skip(reason="BATHYMETRY_PATH not set (needs_bathymetry test)")
    for item in items:
        if "needs_bathymetry" in item.keywords:
            if not os.environ.get("BATHYMETRY_PATH"):
                item.add_marker(skip_needs_bathymetry)
