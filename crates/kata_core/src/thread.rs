//! Multithreading configuration and primitives.
//!
//! Corresponds to `cpp/core/multithread.h` and `cpp/core/multithread.cpp`.
//!
//! KataGo's C++ codebase uses a `MULTITHREADING` macro to optionally compile
//! out threading primitives. In Rust the standard library always provides
//! `std::thread`, `std::sync::Mutex`, etc., so multithreading is always
//! considered enabled.

pub mod counter;
pub mod parallel;
pub mod priority_mutex;
pub mod queue;
pub mod test;

/// Always `true` in the Rust port; threading primitives are always available.
pub const IS_MULTITHREADING_ENABLED: bool = true;
