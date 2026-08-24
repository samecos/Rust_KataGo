@echo off
setlocal

rem Lizzie normally does not inherit the CUDA Toolkit DLL directory.
set "KATAGO_CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
set "PATH=%KATAGO_CUDA_BIN%;%PATH%"

"D:\code\Rust_KataGo\target\release\katago-rs.exe" %*
set "KATAGO_EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %KATAGO_EXIT_CODE%
