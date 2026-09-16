# Derived from FlashAttention Softmax; copyright (c) 2025, Tri Dao.
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
