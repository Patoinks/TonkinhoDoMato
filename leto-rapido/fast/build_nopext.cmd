@echo off
call "C:\Program Files\Microsoft Visual Studio8\Community\VC\Auxiliary\Buildcvars64.bat" >nul
set CLAUDIO_NATIVE_ARCH=AVX2
set CLAUDIO_NO_PEXT=1
python setup.py build_ext --inplace --force
