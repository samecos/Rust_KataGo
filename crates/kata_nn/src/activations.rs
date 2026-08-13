//! Neural-network activation identifiers.
//!
//! Corresponds to `cpp/neuralnet/activations.h`.

/// Identity activation: `f(x) = x`.
pub const ACTIVATION_IDENTITY: i32 = 0;

/// Rectified linear unit: `f(x) = max(x, 0)`.
pub const ACTIVATION_RELU: i32 = 1;

/// Mish activation: `f(x) = x * tanh(softplus(x))`.
pub const ACTIVATION_MISH: i32 = 2;

/// SiLU / Swish activation: `f(x) = x / (1 + exp(-x))`.
pub const ACTIVATION_SILU: i32 = 3;

/// Scaled Mish activation: `f(x) = x * tanh(softplus(8x))`.
pub const ACTIVATION_MISH_SCALE8: i32 = 12;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_ids() {
        assert_eq!(ACTIVATION_IDENTITY, 0);
        assert_eq!(ACTIVATION_RELU, 1);
        assert_eq!(ACTIVATION_MISH, 2);
        assert_eq!(ACTIVATION_SILU, 3);
        assert_eq!(ACTIVATION_MISH_SCALE8, 12);
    }
}
