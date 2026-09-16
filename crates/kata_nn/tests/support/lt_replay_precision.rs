//! Pure numerical-capability guard shared by the replay and its CPU tests.
//! CUDA v13.3 cublasLt.h:995-1009 defines these bit flags. NVIDIA documents
//! that numerical implementation flags can be combined with bitwise OR:
//! https://docs.nvidia.com/cuda/cublas/index.html#cublasltnumericalimplflags-t

pub const ACCUMULATOR_32F: u64 = 0x02u64 << 8;
pub const ACCUMULATOR_TYPE_MASK: u64 = 0xffu64 << 8;
pub const INPUT_16F: u64 = 0x01u64 << 16;
pub const INPUT_TYPE_MASK: u64 = 0xffu64 << 16;

/// Accept FP16 input capability even if other input capabilities are reported.
/// Retain the diagnostic's strict, sole-FP32 accumulator requirement. The
/// actual matrix/scale/compute types are checked separately on descriptors.
pub fn flags_support_fp16_fp32(flags: u64) -> bool {
    flags & INPUT_16F != 0 && flags & ACCUMULATOR_TYPE_MASK == ACCUMULATOR_32F
}

#[cfg(test)]
mod tests {
    use super::flags_support_fp16_fp32;

    #[test]
    fn fp16_and_bf16_input_capabilities_can_coexist() {
        assert!(flags_support_fp16_fp32(197122)); // HMMA | FP32 | FP16 | BF16
    }

    #[test]
    fn original_fp16_fp32_capability_is_supported() {
        assert!(flags_support_fp16_fp32(66050)); // HMMA | FP32 | FP16
    }

    #[test]
    fn bf16_only_input_is_rejected() {
        assert!(!flags_support_fp16_fp32(131584)); // FP32 | BF16
    }

    #[test]
    fn missing_fp32_accumulator_is_rejected() {
        assert!(!flags_support_fp16_fp32(65538)); // HMMA | FP16 input
        assert!(!flags_support_fp16_fp32(0));
    }

    #[test]
    fn fp16_accumulator_is_rejected() {
        assert!(!flags_support_fp16_fp32(65794)); // HMMA | FP16 accum/input
    }

    #[test]
    fn multiple_accumulator_types_are_rejected() {
        assert!(!flags_support_fp16_fp32(66306)); // HMMA | FP16+FP32 accum | FP16
        assert!(!flags_support_fp16_fp32(67074)); // HMMA | FP32+FP64 accum | FP16
    }
}
