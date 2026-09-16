#!/usr/bin/env python3
"""Prepare, or explicitly build, an isolated SM120 FA4 strict-softmax candidate.

The default is CPU-only preparation: no third-party imports, CUDA API calls, or
compiler processes. --build is a separate operation for a scheduled build window.
It uses fake tensor/stream descriptors and hides all CUDA devices. Neither mode
launches a kernel or grants numerical/production approval.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / "target/fork-parity-20260908/fa4-reference"
DEFAULT_OUTPUT = ROOT / "target/fork-parity-20260908/fa4-strict-reference"

# This is intentionally independent of the installed FlashAttention Softmax.
# The surrounding MMA/load/scheduler code is copied from the pinned reference.
STRICT_SOFTMAX = '''# Derived from FlashAttention Softmax; copyright (c) 2025, Tri Dao.
# See LICENSE.FlashAttention.CuTe in the candidate root.
# Experimental strict FP32 arithmetic. No model-level validation is implied.
import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase
import flash_attn.cute.utils as utils


@dataclass
class StrictSoftmax(ParamsBase):
    natural_scale: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(natural_scale, num_rows, arch=80, softmax_scale=None):
        if softmax_scale is not None:
            raise ValueError("StrictSoftmax supports score_mod=None only")
        return StrictSoftmax(
            natural_scale, num_rows,
            cute.make_rmem_tensor(num_rows, Float32),
            cute.make_rmem_tensor(num_rows, Float32), arch, None,
        )

    def reset(self):
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    @cute.jit
    def online_softmax(
        self, acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        # Keep persistent state handles outside the staged row loop, as the
        # pinned upstream Softmax does. Direct self.row_* writes caused the
        # DSL to rebind the dataclass fields to inner-loop SSA results, which
        # escaped the enclosing KV loop and were invalid in finalize().
        row_max = self.row_max
        row_sum = self.row_sum
        natural_scale = self.natural_scale
        arch = self.arch
        row_scale = cute.make_fragment_like(row_max, Float32)
        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            # Scale the FP32 scores before max/subtraction, as Rust does.
            # Do not use exp2, log2(e), or (unscaled_score - max) * scale.
            acc_S_mn[r, None].store(acc_S_mn[r, None].load() * natural_scale)
            scaled_scores = acc_S_mn[r, None].load()
            previous_max = row_max[r]
            current_max = utils.fmax_reduce(
                scaled_scores,
                init_val=previous_max if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )
            current_max = cute.arch.warp_reduction_max(current_max, threads_in_group=4)
            row_max[r] = current_max
            if cutlass.const_expr(check_inf):
                current_max = 0.0 if current_max == -Float32.inf else current_max
            probabilities = cute.math.exp(scaled_scores - current_max, fastmath=False)
            if cutlass.const_expr(is_first):
                row_scale[r] = 1.0
                current_sum = utils.fadd_reduce(probabilities, init_val=None, arch=arch)
            else:
                row_scale[r] = cute.math.exp(previous_max - current_max, fastmath=False)
                current_sum = utils.fadd_reduce(
                    probabilities, init_val=row_sum[r] * row_scale[r], arch=arch,
                )
            row_sum[r] = current_sum
            # Remain FP32 here. The existing caller converts P to half exactly
            # once before the FP16-input, FP32-accumulator PV MMA.
            acc_S_mn[r, None].store(probabilities)
        return row_scale

    @cute.jit
    def finalize(self) -> cute.Tensor:
        row_max = self.row_max
        row_sum = self.row_sum
        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_fragment_like(row_max, Float32)
        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            invalid = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            denominator = Float32(1.0) if invalid else row_sum[r]
            row_scale[r] = cute.math.div(Float32(1.0), denominator, fastmath=False)
        # mLSE is fixed to None. Do not compute or store an unused logarithm.
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{name}: expected exactly one pinned source fragment, found {text.count(old)}")
    return text.replace(old, new, 1)


def artifact_inventory(output: Path) -> list[dict]:
    extensions = {".ptx", ".cubin", ".h", ".o", ".so", ".fatbin", ".mlir", ".ll"}
    return [identity(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.suffix in extensions]


def prepare_sources(reference: Path, output: Path) -> tuple[dict, dict]:
    recorded = json.loads((reference / "fa4_b14_cc6434eb7cd02bdd.json").read_text(encoding="utf-8"))
    source_ids = recorded["python_environment"]["module_sources"]
    for filename, module in (("flash_fwd.py", "flash_attn.cute.flash_fwd"),
                             ("flash_fwd_sm120.py", "flash_attn.cute.flash_fwd_sm120"),
                             ("softmax.py", "flash_attn.cute.softmax"),
                             ("mask.py", "flash_attn.cute.mask")):
        actual = sha256(reference / "source" / filename)
        expected = source_ids[module]["sha256"]
        if actual != expected:
            raise ValueError(f"reference source hash mismatch: {filename}: {actual} != {expected}")

    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    forward = (reference / "source/flash_fwd.py").read_text(encoding="utf-8")
    forward = replace_once(
        forward, "from flash_attn.cute.softmax import Softmax, apply_score_mod_inner",
        "from strict_softmax import StrictSoftmax as Softmax\n"
        "from flash_attn.cute.softmax import apply_score_mod_inner", "Softmax import",
    )
    forward = replace_once(
        forward,
        "softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)",
        "# Strict candidate: retain the natural FP32 scale; score_mod is disabled.\n"
        "        if const_expr(mLSE is not None or self.score_mod is not None):\n"
        "            raise ValueError(\"Strict candidate requires mLSE=None and score_mod=None\")\n"
        "        softmax_scale_natural, softmax_scale = softmax_scale, None",
        "natural scale",
    ).replace("softmax_scale_log2", "softmax_scale_natural")
    forward = replace_once(forward, "# normalize acc_O by row_sum and calculate the lse",
                           "# normalize acc_O by row_sum; this candidate has no LSE calculation",
                           "normalization comment")
    sm120 = (reference / "source/flash_fwd_sm120.py").read_text(encoding="utf-8")
    sm120 = replace_once(sm120,
                        "from flash_attn.cute.flash_fwd import FlashAttentionForwardSm80",
                        "from strict_flash_fwd import FlashAttentionForwardSm80", "SM120 parent")
    (source / "strict_flash_fwd.py").write_text(forward, encoding="utf-8", newline="\n")
    (source / "strict_flash_fwd_sm120.py").write_text(sm120, encoding="utf-8", newline="\n")
    (source / "strict_softmax.py").write_text(STRICT_SOFTMAX, encoding="utf-8", newline="\n")
    # Keep the original source next to the modified modules for a direct diff.
    originals = source / "original"
    originals.mkdir(exist_ok=True)
    for filename in ("flash_fwd.py", "flash_fwd_sm120.py", "softmax.py", "mask.py", "build_aot.py"):
        (originals / filename).write_bytes((reference / "source" / filename).read_bytes())
    for path in reference.glob("LICENSE.*"):
        (output / path.name).write_bytes(path.read_bytes())

    for path in source.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    strict_tree = ast.parse(STRICT_SOFTMAX)
    forbidden = {"exp2", "rcp_approx", "log", "log2"}
    strict_math_calls = {"exp": 0, "div": 0}
    for node in ast.walk(strict_tree):
        if isinstance(node, ast.For) and any(
                isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
                and child.value.id == "self" for statement in node.body for child in ast.walk(statement)):
            raise ValueError("staged row loops must use local aliases, not self fields")
    for node in ast.walk(strict_tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
                raise ValueError(f"forbidden strict-softmax call: {node.func.attr}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in strict_math_calls:
                strict_math_calls[node.func.attr] += 1
                if not any(keyword.arg == "fastmath" and isinstance(keyword.value, ast.Constant)
                           and keyword.value.value is False for keyword in node.keywords):
                    raise ValueError(f"strict {node.func.attr} must explicitly pass fastmath=False")
            for keyword in node.keywords:
                if keyword.arg == "fastmath" and not (isinstance(keyword.value, ast.Constant)
                                                      and keyword.value.value is False):
                    raise ValueError("strict softmax has a non-false fastmath argument")
    if strict_math_calls != {"exp": 2, "div": 1}:
        raise ValueError(f"unexpected strict math call counts: {strict_math_calls}")
    if "compute_softmax_scale_log2(" in forward or "softmax_scale_log2" in forward:
        raise ValueError("unexpected change-of-base calculation")
    if "rP.store(acc_S.load().to(self.dtype))" not in forward:
        raise ValueError("the expected pre-PV half P boundary changed")
    return recorded, {"source_hashes": [identity(p) for p in sorted(source.rglob("*.py"))],
                      "source_ast_audit": "PASS", "numeric_verified": False}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def command_version(path: Path) -> dict:
    result = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"{path} --version failed: {result.stderr}")
    return {"path": str(path.resolve()), "output": (result.stdout + result.stderr).strip()}


def build_candidate(args, manifest: dict) -> None:
    # This path is deliberately unreachable in the default preparation mode.
    # No fallback to real tensors, torch.cuda, or a visible CUDA device is allowed.
    for key in ("CUTE_DSL_COMPILER_OPT", "CUTE_DSL_LIBS", "NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"):
        if os.environ.get(key, "").strip():
            raise RuntimeError(f"inherited {key} is not allowed for the strict reference export")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_HOME"] = str(args.cuda_home)
    os.environ["CUDA_PATH"] = str(args.cuda_home)
    os.environ["FLASH_ATTENTION_ARCH"] = "sm_120"
    os.environ["CUTE_DSL_ARCH"] = "sm_120"
    os.environ["CUTE_DSL_PTXAS_PATH"] = str(args.cuda_home / "bin/ptxas")
    os.environ["CUTE_DSL_KEEP_PTX"] = "1"
    os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
    # The first export failed in build_module's ir.Module.parse(str(module))
    # while preparing KEEP=ir's clean clone. KEEP=ir-debug saves the raw module
    # BEFORE that parse; keep both so the next failure has source-located IR.
    # This changes diagnostics only, not arithmetic or compiler math options.
    os.environ["CUTE_DSL_KEEP"] = "ir-debug,ir,ptx,cubin"
    os.environ["CUTE_DSL_LINEINFO"] = "1"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(args.output_dir)
    os.environ["CUTE_DSL_CACHE_DIR"] = str(args.output_dir / ".dsl-cache")
    os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = str(args.output_dir / ".aot-cache")
    manifest["build_environment"] = {key: os.environ[key] for key in (
        "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "FLASH_ATTENTION_ARCH", "CUTE_DSL_ARCH",
        "CUTE_DSL_PTXAS_PATH", "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED",
        "CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR", "CUTE_DSL_CACHE_DIR", "CUTE_DSL_DISABLE_FILE_CACHING",
        "CUTE_DSL_LINEINFO",
    )}
    expected = manifest["reference_toolchain"]["distributions"]
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"toolchain versions differ from the reference: {observed} != {expected}")
    manifest["build_toolchain"] = {
        "python": sys.executable, "version": sys.version, "distributions": observed,
        "nvcc": command_version(args.cuda_home / "bin/nvcc"),
        "ptxas": command_version(args.cuda_home / "bin/ptxas"),
    }
    manifest["ir_diagnostic_capture"] = {
        "raw_before_clone_parse": True,
        "source_line_info": True,
        "reason": "first attempt failed parsing the clean IR clone, before arithmetic lowering",
        "arithmetic_changed_since_first_attempt": False,
        "state_handle_revision": "local aliases avoid self dataclass rebinding to nested-loop SSA",
    }

    import cutlass
    import cutlass.cute as cute
    import cutlass._mlir_helpers.math as dsl_math
    import flash_attn.cute.mask as flash_mask
    import flash_attn.cute.softmax as flash_softmax
    from flash_attn.cute.utils import AuxData
    from cutlass.cute.export.c_header_generator import CuteCHeaderGenerator

    for module, key in ((flash_mask, "flash_attn.cute.mask"),
                        (flash_softmax, "flash_attn.cute.softmax")):
        if sha256(Path(module.__file__)) != manifest["reference_toolchain"]["module_sources"][key]["sha256"]:
            raise RuntimeError(f"installed dependency source differs from pinned reference: {key}")
    manifest["build_toolchain"]["math_source"] = identity(Path(dsl_math.__file__))
    sys.path.insert(0, str(args.output_dir / "source"))
    from strict_flash_fwd_sm120 import FlashAttentionForwardSm120

    # Record the actual transitive Python source used by this export, beyond
    # distribution versions (local package patches must remain observable).
    manifest["build_toolchain"]["module_sources"] = {
        name: identity(Path(module.__file__))
        for name, module in sorted(sys.modules.items())
        if name.startswith(("cutlass", "flash_attn.cute", "quack"))
        and getattr(module, "__file__", None)
        and Path(module.__file__).is_file()
        and Path(module.__file__).suffix == ".py"
    }
    # AuxData contains no runtime tensors, as in the original AOT generator.
    original_generate = CuteCHeaderGenerator._generate_arguments

    def generate_without_aux(self, symbol_prefix, args_spec, positional, keywords):
        rectified = args_spec.get_rectified_args(positional, keywords)

        class Spec:
            signature = args_spec.signature

            @staticmethod
            def get_rectified_args(_positional, _keywords):
                return [None if isinstance(value, AuxData) else value for value in rectified]

        return original_generate(self, symbol_prefix, Spec(), positional, keywords)

    CuteCHeaderGenerator._generate_arguments = generate_without_aux
    b, s, h, d = args.batch, 361, 12, 32
    kernel = FlashAttentionForwardSm120(
        cutlass.Float16, d, d, 1, is_causal=False, is_local=False, pack_gqa=False,
        tile_m=args.tile_m, tile_n=args.tile_n, num_stages=1, num_threads=128,
        Q_in_regs=False, score_mod=None, mask_mod=None, has_aux_tensors=False,
        qk_acc_dtype=cutlass.Float32, pv_acc_dtype=cutlass.Float32,
    )
    packed_stride = (s * 3 * h * d, 3 * h * d, d, 1)
    out_stride = (s * h * d, h * d, d, 1)
    q, k, v = [cute.runtime.make_fake_tensor(cutlass.Float16, (b, s, h, d), packed_stride,
                                            assumed_align=16) for _ in range(3)]
    out = cute.runtime.make_fake_tensor(cutlass.Float16, (b, s, h, d), out_stride, assumed_align=16)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    compiled = cute.compile(
        kernel, q, k, v, out, None, manifest["parameters"]["sample_natural_scale_f32"],
        None, None, None, None, None, None, None, None,
        None, AuxData(), None, None, stream,
    )
    compiled.export_to_c(str(args.output_dir), args.artifact_stem, args.artifact_stem)
    artifacts = artifact_inventory(args.output_dir)
    missing = {".h", ".o", ".ptx", ".cubin"} - {Path(item["path"]).suffix for item in artifacts}
    if missing:
        raise RuntimeError(f"export incomplete; missing artifact types: {sorted(missing)}")
    manifest.update(status="EXPORTED_UNVERIFIED", build_completed=True, artifacts=artifacts,
                    codegen_semantics_verified=False, numeric_verified=False, production_eligible=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true", help="Explicitly invoke the offline AOT compiler")
    mode.add_argument("--prepare-only", action="store_true", help="CPU source preparation only (default)")
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-13.3"))
    parser.add_argument("--batch", type=int, default=14)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=96)
    parser.add_argument("--artifact-stem", default="fa4_strict_b14_m128_n96_fp32")
    args = parser.parse_args()
    args.reference_dir = args.reference_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if (args.output_dir == args.reference_dir or args.reference_dir in args.output_dir.parents
            or args.output_dir in args.reference_dir.parents):
        parser.error("output must be separate from the immutable reference directory")
    if (args.batch, args.tile_m, args.tile_n) != (14, 128, 96):
        parser.error("this first prototype supports only B14/M128/N96; use a separate audited variant for other shapes")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.artifact_stem):
        parser.error("artifact-stem must be a C identifier")
    if args.output_dir.exists() and artifact_inventory(args.output_dir):
        parser.error("output already contains compiler artifacts; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        # Retain a failed/exported attempt before a later preparation overwrites
        # its status, exact toolchain hashes and traceback.
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") not in ("PREPARING", "PREPARED_NOT_BUILT"):
            history = args.output_dir / "attempt-history"
            history.mkdir(exist_ok=True)
            digest = sha256(manifest_path)
            archived = history / f"{previous.get('status', 'UNKNOWN')}-{digest[:16]}.json"
            if not archived.exists():
                archived.write_bytes(manifest_path.read_bytes())
    manifest = {"schema": 1, "status": "PREPARING", "build_completed": False,
                "numeric_verified": False, "production_eligible": False, "artifacts": [],
                "generator": identity(Path(__file__)), "started_utc": datetime.now(timezone.utc).isoformat(),
                "preparation_runtime": {"python": sys.executable, "version": sys.version},
                "parameters": {"batch": 14, "sequence": 361, "heads": 12, "head_dim": 32,
                               "tile_m": 128, "tile_n": 96, "stages": 1, "threads": 128,
                               "qk_accumulator": "fp32", "pv_accumulator": "fp32",
                               "input_output": "fp16", "p_boundary": "FP32 exp/sum, then half before PV",
                               "scale_abi": "natural FP32 scale; never multiply by log2(e)",
                               "sample_natural_scale_f32": f32(f32(1.0 / f32(math.sqrt(f32(math.sqrt(32.0))))) ** 2),
                               "qkv_shape_bshd": [14, 361, 12, 32], "qkv_stride_elements": [415872, 1152, 32, 1],
                               "output_stride_elements": [138624, 384, 32, 1], "mLSE": None},
                "abi_status": "must derive from this export; static fake descriptors may change the old ABI",
                "acceptance": "codegen semantic audit, attention reference, TF3/ONNX all-head gates and clean ABBA pending"}
    write_json(manifest_path, manifest)
    try:
        reference, prepared = prepare_sources(args.reference_dir, args.output_dir)
        manifest.update(prepared, reference_toolchain=reference["python_environment"],
                        reference_metadata=identity(args.reference_dir / "fa4_b14_cc6434eb7cd02bdd.json"),
                        status="PREPARED_NOT_BUILT")
        write_json(manifest_path, manifest)
        if args.build:
            manifest["status"] = "BUILDING"
            write_json(manifest_path, manifest)
            build_candidate(args, manifest)
        print(f"{manifest['status']}: {manifest_path}")
        return 0
    except Exception as error:
        manifest.update(status="BUILD_FAILED" if args.build else "PREPARATION_FAILED",
                        error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                        artifacts=artifact_inventory(args.output_dir))
        print(f"{manifest['status']}: {error}; see {manifest_path}", file=sys.stderr)
        return 1
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
