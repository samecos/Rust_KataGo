@echo off
setlocal

rem ============================================================
rem 搜索改进全开模式(2026-08-25 本轮新参数体验入口)
rem
rem 打开的内容:
rem   ponderingEnabled=true        对手回合后台思考(P0-2)
rem   analysisWideRootNoise=0.25   分析选点发散(P0-1,推荐 0.2-0.3)
rem   useEvalCache=true            跨手洞见保留(已在 gtp_cuda.cfg 默认)
rem   evalCacheMinVisits=100       同上
rem   puctVarExploration=0.5       PUCT-V 子级方差探索(P2-6,实验:实测偏集中,可删)
rem
rem 想关掉某项:从下面 override-config 里删对应键即可。
rem 注意:pondering 会让 GUI 频繁 play/undo 时持续占 GPU,
rem       纯 GTP 控制台/对局场景体验最佳;GUI 分析建议去掉 ponderingEnabled。
rem ============================================================

rem Lizzie/GUI normally does not inherit the CUDA Toolkit DLL directory.
set "KATAGO_CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
set "PATH=%KATAGO_CUDA_BIN%;%PATH%"

set "ENGINE=D:\code\Rust_KataGo\target\release\katago-rs.exe"
set "CFG=D:\code\Rust_KataGo\configs\gtp_cuda.cfg"
set "MODEL=D:/code/b11fix.onnx"
set "OVERRIDES=ponderingEnabled=true,analysisWideRootNoise=0.25,puctVarExploration=0.5,useEvalCache=true,evalCacheMinVisits=100"

echo [new-features] pondering=on wideRootNoise=0.25 evalCache=on puctVar=0.5 1>&2

"%ENGINE%" gtp --config "%CFG%" --model %MODEL% --override-config %OVERRIDES% %*
set "KATAGO_EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %KATAGO_EXIT_CODE%
