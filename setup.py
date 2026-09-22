"""Build the TinyOpponent and tonkai_fast extensions in place.

    python setup.py build_ext --inplace

This is the path used on Windows; tiny_opponent/CMakeLists.txt does the same
job for anyone who prefers cmake.
"""

import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "tiny_opponent",
    ["tiny_opponent/engine.cpp", "tiny_opponent/bindings.cpp"],
    include_dirs=["tiny_opponent"],
    cxx_std=17,
)

# The V3 agent's chess rules, ported from Python (see tonkai_fast/fast.cpp).
fast = Pybind11Extension(
    "tonkai_fast",
    ["tonkai_fast/fast.cpp"],
    cxx_std=20,
    # Bit-identical to Python: no fused multiply-add, no reassociation.
    extra_compile_args=["/fp:strict"] if sys.platform == "win32" else ["-ffp-contract=off"],
)

setup(
    name="tiny_opponent",
    version="0.1.0",
    ext_modules=[ext, fast],
    cmdclass={"build_ext": build_ext},
)
