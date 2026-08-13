//! KataGo distributed training client placeholder.
//!
//! Corresponds to `KataGo/cpp/distributed/client.h`, `client.cpp`, and
//! `clienttask.cpp`. The C++ implementation communicates with the KataGo
//! distributed training server over HTTP(S). This crate exposes the expected
//! public API as a skeleton, to be filled in once distributed training support
//! is needed.

pub mod client;
pub mod task;

pub use client::{ClientError, Connection, Url};
pub use task::{ModelInfo, RunParameters, Task};
