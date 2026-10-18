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
    if not os.environ.get("CLAUDIO_NO_PEXT"):
        compile_args.append("/DCLAUDIO_PEXT")   # BMI2 PEXT slider tables (every AVX2 CPU here has BMI2)

link_args = []
# Profile-guided optimization (MSVC): "gen" builds an instrumented module, a few games train it,
# "use" rebuilds it from that profile. Same code and same /fp:strict arithmetic, faster layout.
pgo = os.environ.get("CLAUDIO_PGO", "")
if pgo:
    compile_args.append("/GL")
    link_args += ["/LTCG", "/GENPROFILE" if pgo == "gen" else "/USEPROFILE"]
if os.environ.get("CLAUDIO_PDB"):      # symbols for a native profiler; same optimized code
    compile_args.append("/Zi")
    link_args.append("/DEBUG")
fast = Pybind11Extension(
    "claudio_fast",
    ["claudio_fast.cpp"],
    cxx_std=20,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)
setup(name="claudio-fast", ext_modules=[fast], cmdclass={"build_ext": build_ext})
