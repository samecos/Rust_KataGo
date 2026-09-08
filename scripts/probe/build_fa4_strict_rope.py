#!/usr/bin/env python3
"""Prepare an isolated opt-in strict FA4 + shared-memory RoPE candidate.

Default: standard-library-only CPU source preparation. --build explicitly invokes
the inherited offline AOT path and is reserved for a scheduled build window.
The frozen r2 generator and its artifacts are never edited. No kernel is launched.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import textwrap


BASE_PATH = Path(__file__).with_name("build_fa4_strict.py")
BASE_SHA256 = "444bf26328eb95492e8cd252d764fc6fa5492d9d9d444ea85ac1824c40bf80e7"
R2_FORWARD_SHA256 = "39c3ad181b16e20efb554605e3e3cc5c87b57c9e6aefa6c9393a0f7921bb3d78"
R2_SOFTMAX_SHA256 = "2189afa893faf57d6f5175de0ab19ea0cb024365f4ba71a1a8880fe5ea7ab087"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "target/fork-parity-20260908/fa4-strict-rope-smem"


ROPE_HELPER = '''    @cute.jit
    def rotate_qk_smem(
        self, tile: cute.Tensor, rope_cos: cute.Tensor, rope_sin: cute.Tensor,
        start: Int32, head: Int32, seqlen: Int32, tidx: Int32,
    ):
        # Fixed B14/S361/H12/D32, 128 threads, stage 1. Each thread owns an
        # aligned logical 8-half chunk; indexing respects the CuTe swizzle.
        # All threads enter and leave this method; its caller supplies barriers.
        assert cute.size(tile, mode=[1]) == 32
        for group in cutlass.range_constexpr(cute.size(tile, mode=[0]) // 32):
            vector_index = tidx + group * 128
            row = vector_index // 4
            pair_base = (vector_index % 4) * 4
            pos = start + row
            for pair in cutlass.range_constexpr(4):
                pair_index = pair_base + pair
                d0 = pair_index * 2
                if pos < seqlen:
                    x = tile[row, d0].to(Float32)
                    y = tile[row, d0 + 1].to(Float32)
                    co = rope_cos[pos, head, pair_index]
                    sn = rope_sin[pos, head, pair_index]
                    # Match the current Rust q64serial PTX contraction boundary:
                    # u = RN(RN(x*co) - RN(y*sn)); v = FMA_RN(y,co,RN(x*sn)).
                    # sub has no rounding kwarg in this pinned DSL; absent
                    # fastmath yields ordinary FP32 subtraction. Explicit RN
                    # mul intrinsics prevent contraction across the u boundary.
                    xc = cute.math.mul(x, co, fastmath=False, rounding=RoundingMode.NEAREST_EVEN)
                    ys = cute.math.mul(y, sn, fastmath=False, rounding=RoundingMode.NEAREST_EVEN)
                    xs = cute.math.mul(x, sn, fastmath=False, rounding=RoundingMode.NEAREST_EVEN)
                    u = cute.math.sub(xc, ys, fastmath=False)
                    v = cute.math.fma(y, co, xs, fastmath=False, rounding=RoundingMode.NEAREST_EVEN)
                    # One FP32->half RN boundary before QK MMA, like Rust.
                    tile[row, d0] = u.to(cutlass.Float16)
                    tile[row, d0 + 1] = v.to(cutlass.Float16)
                else:
                    # No out-of-range coefficient loads for final Q/K tiles.
                    tile[row, d0] = cutlass.Float16(0.0)
                    tile[row, d0 + 1] = cutlass.Float16(0.0)

'''

ROPE_STAGE = '''        # Independent opt-in RoPE: cp.async wait + CTA barrier above
        # makes raw Q/K available. Q is rotated once; each fresh K tile once.
        if const_expr(is_first_n_block):
            self.rotate_qk_smem(
                smem_copy_params.rope_sQ, smem_copy_params.rope_cos,
                smem_copy_params.rope_sin, m_block * self.tile_m,
                head_idx, seqlen.seqlen_q, smem_copy_params.rope_tidx,
            )
        self.rotate_qk_smem(
            smem_copy_params.rope_sK[None, None, smem_pipe_read],
            smem_copy_params.rope_cos, smem_copy_params.rope_sin,
            n_block * self.tile_n, head_idx, seqlen.seqlen_k,
            smem_copy_params.rope_tidx,
        )
        # All in-place half stores must be visible before any ldmatrix/QK MMA.
        cute.arch.barrier()
'''


def edit_method(text: str, class_name: str, method_name: str, transform) -> str:
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    lines = text.splitlines(keepends=True)
    segment = "".join(lines[method.lineno - 1:method.end_lineno])
    return "".join(lines[:method.lineno - 1]) + transform(segment) + "".join(lines[method.end_lineno:])


def main() -> int:
    observed = hashlib.sha256(BASE_PATH.read_bytes()).hexdigest()
    if observed != BASE_SHA256:
        raise RuntimeError(f"strict r2 generator changed: {observed} != {BASE_SHA256}")
    spec = importlib.util.spec_from_file_location("_fa4_strict_rope_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned strict generator")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)  # stdlib imports and function definitions only
    original_prepare = base.prepare_sources
    original_build_source = inspect.getsource(base.build_candidate)

    def prepare_rope(reference: Path, output: Path):
        reference_metadata, prepared = original_prepare(reference, output)
        source = output / "source"
        forward_path = source / "strict_flash_fwd.py"
        if base.sha256(forward_path) != R2_FORWARD_SHA256:
            raise RuntimeError("prepared forward differs from frozen strict r2")
        if base.sha256(source / "strict_softmax.py") != R2_SOFTMAX_SHA256:
            raise RuntimeError("prepared softmax differs from frozen strict r2")
        text = forward_path.read_text(encoding="utf-8")
        text = base.replace_once(text, "from cutlass import Float32, Int32, const_expr",
                                 "from cutlass import Float32, Int32, const_expr\n"
                                 "from cutlass._mlir_helpers.math import RoundingMode", "rounding import")
        text = base.replace_once(text, "    @cute.jit\n    def load_Q(",
                                 ROPE_HELPER + "    @cute.jit\n    def load_Q(", "RoPE helper")

        def host_patch(segment):
            segment = base.replace_once(segment, "        mO: cute.Tensor,\n        mLSE:",
                                        "        mO: cute.Tensor,\n        mRoPECos: cute.Tensor,\n"
                                        "        mRoPESin: cute.Tensor,\n        mLSE:", "host coefficient args")
            segment = base.replace_once(segment, "        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()",
                                        "        if const_expr(mRoPECos.element_type != Float32 or mRoPESin.element_type != Float32):\n"
                                        "            raise TypeError(\"RoPE coefficient caches must be FP32\")\n"
                                        "        if const_expr(self.Q_in_regs or self.num_stages != 1):\n"
                                        "            raise ValueError(\"RoPE variant requires Q_in_regs=False and one stage\")\n"
                                        "        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()", "scope guard")
            return base.replace_once(segment, "            mO,\n            mLSE,",
                                     "            mO,\n            mRoPECos,\n            mRoPESin,\n            mLSE,",
                                     "kernel coefficient forwarding")

        def kernel_patch(segment):
            segment = base.replace_once(segment, "        mO: cute.Tensor,\n        mLSE:",
                                        "        mO: cute.Tensor,\n        mRoPECos: cute.Tensor,\n"
                                        "        mRoPESin: cute.Tensor,\n        mLSE:", "device coefficient args")
            return base.replace_once(segment, "            tOsVt=tOsVt,\n",
                                     "            tOsVt=tOsVt,\n"
                                     "            rope_sQ=sQ, rope_sK=sK, rope_cos=mRoPECos,\n"
                                     "            rope_sin=mRoPESin, rope_tidx=tidx,\n", "RoPE stage handles")

        text = edit_method(text, "FlashAttentionForwardSm80", "__call__", host_patch)
        text = edit_method(text, "FlashAttentionForwardSm80", "kernel", kernel_patch)
        text = edit_method(text, "FlashAttentionForwardSm80", "compute_one_n_block",
                           lambda segment: base.replace_once(segment,
                               "        sync()\n\n        # need predicates for the first tile",
                               "        sync()\n\n" + ROPE_STAGE + "\n        # need predicates for the first tile",
                               "post-wait shared rotation"))
        forward_path.write_text(text, encoding="utf-8", newline="\n")

        # Mechanically adapt the pinned offline builder instead of duplicating
        # toolchain policy or modifying CuTe globals. The exact definition used
        # is saved beside the kernel sources and recorded in the manifest.
        builder_source = base.replace_once(original_build_source,
            "    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)",
            "    rope_cos = cute.runtime.make_fake_tensor(cutlass.Float32, (361, 12, 16), (192, 16, 1), assumed_align=16)\n"
            "    rope_sin = cute.runtime.make_fake_tensor(cutlass.Float32, (361, 12, 16), (192, 16, 1), assumed_align=16)\n"
            "    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)", "fake RoPE descriptors")
        builder_source = base.replace_once(builder_source,
            "        kernel, q, k, v, out, None, manifest[\"parameters\"][\"sample_natural_scale_f32\"],",
            "        kernel, q, k, v, out, rope_cos, rope_sin, None, manifest[\"parameters\"][\"sample_natural_scale_f32\"],",
            "compile coefficient arguments")
        builder_source = base.replace_once(builder_source,
            '        "arithmetic_changed_since_first_attempt": False,',
            '        "strict_softmax_arithmetic_unchanged": True,\n'
            '        "new_rope_stage_unverified": True,', "RoPE diagnostic scope")
        builder_path = source / "build_candidate_rope.py"
        builder_path.write_text(builder_source, encoding="utf-8", newline="\n")
        for path in source.glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Only defines a Python function; its third-party imports/compiler calls
        # remain unreachable unless base.main receives the explicit --build flag.
        namespace = dict(base.__dict__)
        exec(compile(builder_source, str(builder_path), "exec"), namespace)
        base.build_candidate = namespace["build_candidate"]

        helper_tree = ast.parse(textwrap.dedent(ROPE_HELPER))
        math_calls = []
        for node in ast.walk(helper_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("mul", "sub", "fma"):
                math_calls.append(node.func.attr)
                if not any(k.arg == "fastmath" and isinstance(k.value, ast.Constant) and k.value.value is False for k in node.keywords):
                    raise RuntimeError("RoPE arithmetic must explicitly disable fastmath")
        if sorted(math_calls) != ["fma", "mul", "mul", "mul", "sub"]:
            raise RuntimeError(f"unexpected RoPE arithmetic: {math_calls}")
        if base.sha256(source / "strict_softmax.py") != R2_SOFTMAX_SHA256:
            raise RuntimeError("strict softmax was modified")
        prepared.update(
            generator=base.identity(Path(__file__)), base_generator=base.identity(BASE_PATH),
            source_hashes=[base.identity(p) for p in sorted(source.rglob("*.py"))],
            source_ast_audit="PASS", strict_softmax_unchanged=True,
            rope_variant={
                "id": "strict-rope-smem-b14-m128-n96", "opt_in": "this separate generator and a future separate probe variant",
                "method": "wait+barrier; first Q once, current K once; barrier before ldmatrix",
                "coefficient_dtype": "FP32", "coefficient_shape": [361, 12, 16],
                "coefficient_stride_elements": [192, 16, 1], "batch_shared": True,
                "coefficient_source": "existing Rust rope_cos/rope_sin CUDA cache; no trig computation or half conversion",
                "rotation": "u=RN(RN(x*co)-RN(y*sn)); v=FMA_RN(y,co,RN(x*sn)); then half RN",
                "input": "unrotated packed QKV; do not launch standalone RoPE first",
                "tail": "pos>=seqlen skips coefficient loads and zeroes the local half pair",
                "shared_memory_bytes_expected": 20480, "extra_cta_barriers_per_k_tile": 1,
                "q_in_regs": False, "stages": 1,
                "source_reference": "attention_fa2_q64.cu:120-127,157-164; Rust q64serial PTX:102-108",
            },
            abi_status="PROPOSED_ONLY: new Q,K,V,O,cos,sin,natural_scale,optional scheduler; derive actual device ABI after export",
            abi_proposal={
                "status": "NOT_DERIVED_NOT_COMPILED",
                "expected_pointer_order": ["Q", "K", "V", "O", "rope_cos", "rope_sin"],
                "following_arguments": ["natural FP32 scale", "compiler-specific scheduler, if retained"],
                "cache_layout": "FP32 [361,12,16], strides [192,16,1], batch shared, 16-byte aligned",
                "packed_qkv_byte_offsets": [0, 768, 1536],
                "grid_expected": [3, 12, 14], "block_expected": [128, 1, 1],
                "dynamic_smem_expected": 20480,
                "required": "derive exported header/PTX/host launch, pin new cubin hash, compare rotated half bits; no existing probe modified",
            },
            acceptance="new RoPE PTX audit and half-bit comparison, attention reference, whole-model gates and clean ABBA pending",
        )
        return reference_metadata, prepared

    base.DEFAULT_OUTPUT = OUTPUT
    base.prepare_sources = prepare_rope
    if not any(arg == "--artifact-stem" or arg.startswith("--artifact-stem=") for arg in sys.argv[1:]):
        sys.argv.extend(["--artifact-stem", "fa4_strict_rope_smem_b14_m128_n96"])
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
