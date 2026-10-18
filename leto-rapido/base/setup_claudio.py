"""Build Claudio's own copy of tonkai_fast (the committed source, frozen):
    python setup_claudio.py build_ext --inplace
"""
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

fast = Pybind11Extension(
    "tonkai_fast",
    ["tonkai_fast/fast.cpp"],
    cxx_std=20,
    extra_compile_args=["/fp:strict"] if sys.platform == "win32" else ["-ffp-contract=off"],
)
setup(name="claudio-base", ext_modules=[fast], cmdclass={"build_ext": build_ext})
