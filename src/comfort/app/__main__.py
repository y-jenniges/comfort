"""Launch the COMFORT Data Explorer app.

Usage:
    python -m comfort.app
"""
from __future__ import annotations
from .explorer import app

app.run(debug=True)
