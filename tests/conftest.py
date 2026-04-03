"""
conftest.py — shared pytest fixtures.

All tests run from the project root (srfm-lab/).
lib/ is added to sys.path here so tests can import without LEAN.
"""

import sys
import os

# Ensure lib/ is on path for all tests
lib_dir = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, os.path.abspath(lib_dir))
