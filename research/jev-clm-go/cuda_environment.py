"""Process-local Windows cuDNN isolation shared by CUDA research training."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def isolate_windows_cudnn(torch_module: Any) -> None:
    """Prevent wheel cuDNN from loading optional toolkit plugins at other versions.

    Only this Python process's PATH changes. The bundled wheel library path is
    preserved, and removed directories are recorded for training manifests.
    """
    if os.name != "nt":
        return
    bundled = (Path(torch_module.__file__).parent / "lib").resolve()
    kept = []
    removed = [part for part in os.environ.get("RUST_KATAGO_SYSTEM_CUDNN_PATHS", "").split(";") if part]
    for part in os.environ.get("PATH", "").split(";"):
        candidate = Path(part)
        if part and candidate.resolve() != bundled and (
            (candidate / "cudnn64_9.dll").is_file() or
            (candidate / "cudnn_engines_tensor_ir64_9.dll").is_file()
        ):
            if part not in removed:
                removed.append(part)
        else:
            kept.append(part)
    os.environ["PATH"] = ";".join(kept)
    os.environ["RUST_KATAGO_SYSTEM_CUDNN_PATHS"] = ";".join(removed)
