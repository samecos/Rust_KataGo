@echo off
setlocal

rem Rust_KataGo (WSL build) GTP wrapper for Lizzie / Sabaki / KaTrain.
rem Point the GUI engine command at this .cmd; it forwards to the WSL
rem release binary with the CUDA runtime environment set up.
rem
rem Engine args after "gtp" are built in here; extra GTP-level flags can
rem be appended by the GUI only if they come after the ones below.

wsl.exe -d Ubuntu-24.04 -- bash -c "export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:/usr/lib/wsl/lib; export KATAGO_CUTLASS_ROOT=/mnt/d/code/cutlass; exec /mnt/d/code/Rust_KataGo/target/cudarocmopt-wsl/release/katago-rs gtp --config /mnt/d/code/Rust_KataGo/configs/gtp_wsl.cfg --model /mnt/d/code/b11fix.onnx"

endlocal
