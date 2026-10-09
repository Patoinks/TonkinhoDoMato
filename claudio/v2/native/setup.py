import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

flags = ['/fp:strict', '/arch:AVX2'] if sys.platform == 'win32' else ['-ffp-contract=off']
setup(name='claudio-fast-v2', ext_modules=[Pybind11Extension('claudio_fast_v2', ['extra.cpp'], cxx_std=20,
                                                         extra_compile_args=flags)], cmdclass={'build_ext': build_ext})
