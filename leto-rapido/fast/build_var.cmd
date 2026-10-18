@echo off
rem build_var.cmd OUTNAME "DEFINES": an AVX2+PEXT build saved as OUTNAME.pyd (experiments)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set CLAUDIO_NATIVE_ARCH=AVX2
set CLAUDIO_DEFINES=%~2
python setup.py build_ext --force >nul 2>&1
copy /Y build\lib.win-amd64-cpython-313\claudio_fast.cp313-win_amd64.pyd %1.pyd >nul
