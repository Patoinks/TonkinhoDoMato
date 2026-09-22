"""Build the TinyOpponent extension in place.

    python setup.py build_ext --inplace

This is the path used on Windows; tiny_opponent/CMakeLists.txt does the same
job for anyone who prefers cmake.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext = Pybind11Extension(
    "tiny_opponent",
    ["tiny_opponent/engine.cpp", "tiny_opponent/bindings.cpp"],
    include_dirs=["tiny_opponent"],
    cxx_std=17,
)

setup(
    name="tiny_opponent",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
