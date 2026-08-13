//! Neural-network inference backends.
//!
//! Each submodule implements the [`crate::backend::Backend`] trait for a
//! specific execution provider. Backends that require external native
//! libraries are gated behind Cargo features so that the crate builds in
//! environments without those libraries.

pub mod trt;

#[cfg(feature = "trt")]
pub mod trt_ffi;
