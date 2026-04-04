"""
setup.py — Build the fast_indicators C extension.

Usage:
    python setup.py build_ext --inplace
    pip install -e .
"""

import os
import sys
from setuptools import setup, Extension, find_packages

# Compiler flags by platform
extra_compile_args = ["-O3", "-march=native", "-ffast-math", "-std=c99"]
extra_link_args    = ["-lm"]

if sys.platform == "win32":
    # MSVC flags
    extra_compile_args = ["/O2", "/fp:fast"]
    extra_link_args    = []
elif sys.platform == "darwin":
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args    += ["-lomp"]

src_dir = os.path.join(os.path.dirname(__file__), "src")

fast_indicators_ext = Extension(
    name    = "fast_indicators.fast_indicators",
    sources = [os.path.join(src_dir, "indicators_py.c")],
    include_dirs = [src_dir],
    extra_compile_args = extra_compile_args,
    extra_link_args    = extra_link_args,
    language = "c",
)

setup(
    name         = "fast_indicators",
    version      = "0.1.0",
    description  = "High-performance C indicators and BH physics for SRFM quant lab",
    author       = "SRFM Lab",
    packages     = find_packages(),
    ext_modules  = [fast_indicators_ext],
    python_requires = ">=3.9",
    install_requires = [
        "numpy>=1.21",
    ],
    extras_require = {
        "bench": ["pandas>=1.3", "ta-lib"],
        "dev"  : ["pytest", "pytest-benchmark"],
    },
    classifiers = [
        "Programming Language :: C",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
