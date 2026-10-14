"""Claudio's own C++ module (TonkAI's committed rules + Claudio's root search):
    python setup.py build_ext --inplace
"""
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

compile_args = ["/fp:strict"] if sys.platform == "win32" else ["-ffp-contract=off"]
# Optional local build. Keep portable binaries as the default; strict floating
# point still prevents FMA from changing material scores or search ordering.
arch = os.environ.get("CLAUDIO_NATIVE_ARCH", "").upper()
if arch:
    if arch != "AVX2" or sys.platform != "win32":
        raise ValueError("CLAUDIO_NATIVE_ARCH supports only AVX2 with Windows/MSVC")
    compile_args.append("/arch:AVX2")

fast = Pybind11Extension(
    "claudio_fast",
    ["claudio_fast.cpp"],
    cxx_std=20,
    extra_compile_args=compile_args,
)
setup(name="claudio-fast", ext_modules=[fast], cmdclass={"build_ext": build_ext})
