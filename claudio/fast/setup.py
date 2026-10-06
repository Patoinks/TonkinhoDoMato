"""Claudio's own C++ module (TonkAI's committed rules + Claudio's root search):
    python setup.py build_ext --inplace
"""
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

fast = Pybind11Extension(
    "claudio_fast",
    ["claudio_fast.cpp"],
    cxx_std=20,
    extra_compile_args=["/fp:strict"] if sys.platform == "win32" else ["-ffp-contract=off"],
)
setup(name="claudio-fast", ext_modules=[fast], cmdclass={"build_ext": build_ext})
