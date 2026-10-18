@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set CLAUDIO_NATIVE_ARCH=AVX2
set CLAUDIO_PGO=gen
python setup.py build_ext --inplace --force
