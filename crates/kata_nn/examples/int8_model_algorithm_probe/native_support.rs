//! Copied narrow helpers from the reviewed synthetic native probe r1.
//! Source SHA: 9ac370e2342083a28628d463a54f004463665b36c3aa82fc14529c17750d4d3e.
//! Changes limited to module imports and visibility; no original file changes.
use super::{Result, Source, hash, require};
use cudarc::driver::{CudaStream, sys as ds};
use kata_nn::backends::cuda;
use std::{ffi::c_void, fs, path::PathBuf, sync::Arc};
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut *mut c_void) -> i32;
    fn GetModuleFileNameW(module: *mut c_void, buffer: *mut u16, size: u32) -> u32;
}
fn module_source(name: &str, address: *const c_void) -> Result<Source> {
    let mut module = std::ptr::null_mut();
    // FROM_ADDRESS | UNCHANGED_REFCOUNT: the real DLL symbol, never the Rust wrapper.
    require(
        unsafe { GetModuleHandleExW(6, address.cast(), &mut module) } != 0,
        "GetModuleHandleExW failed",
    )?;
    let mut path = vec![0u16; 32768];
    let n = unsafe { GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) } as usize;
    require(
        n > 0 && n < path.len(),
        "GetModuleFileNameW failed/truncated",
    )?;
    let path = PathBuf::from(String::from_utf16(&path[..n]).map_err(|e| e.to_string())?);
    let raw = fs::read(&path).map_err(|e| e.to_string())?;
    Ok(Source {
        logical_name: name.into(),
        path,
        bytes: raw.len() as u64,
        sha256: hash(&raw),
    })
}
pub(super) fn loaded_libraries() -> Result<(Source, Source)> {
    unsafe {
        let lt = cudarc::cublaslt::sys::culib()
            .get::<unsafe extern "C" fn() -> usize>(b"cublasLtGetVersion\0")
            .map_err(|e| e.to_string())?;
        let blas = cudarc::cublas::sys::culib()
            .get::<unsafe extern "C" fn(
                cudarc::cublas::sys::cublasHandle_t,
                *mut i32,
            ) -> cudarc::cublas::sys::cublasStatus_t>(b"cublasGetVersion_v2\0")
            .map_err(|e| e.to_string())?;
        Ok((
            module_source("loaded/cublasLt", *lt as *const c_void)?,
            module_source("loaded/cublas", *blas as *const c_void)?,
        ))
    }
}
pub(super) struct ProbeGraph {
    graph: ds::CUgraph,
    exec: ds::CUgraphExec,
    stream: Arc<CudaStream>,
}
impl Drop for ProbeGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = self.stream.context().bind_to_thread();
            if !self.exec.is_null() {
                let _ = cudarc::driver::result::graph::exec_destroy(self.exec);
            }
            if !self.graph.is_null() {
                let _ = cudarc::driver::result::graph::destroy(self.graph);
            }
        }
    }
}
impl ProbeGraph {
    pub(super) fn launch(&self) -> Result<()> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        unsafe {
            cudarc::driver::result::graph::launch(self.exec, self.stream.cu_stream())
                .map_err(|e| e.to_string())
        }
    }
}
struct CaptureCleanup {
    stream: Arc<CudaStream>,
    active: bool,
}
impl Drop for CaptureCleanup {
    fn drop(&mut self) {
        if self.active {
            unsafe {
                if let Ok(g) = cudarc::driver::result::stream::end_capture(self.stream.cu_stream())
                {
                    if !g.is_null() {
                        let _ = cudarc::driver::result::graph::destroy(g);
                    }
                }
            }
        }
    }
}
pub(super) fn capture(
    stream: &Arc<CudaStream>,
    operation: impl FnOnce() -> Result<()>,
) -> Result<ProbeGraph> {
    stream
        .begin_capture(ds::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| e.to_string())?;
    let mut cleanup = CaptureCleanup {
        stream: stream.clone(),
        active: true,
    };
    let result = operation();
    // Raw end first: do not instantiate a failed capture, and do not leak
    // the graph when instantiate fails (cudarc 0.19.9 safe API can leak it).
    let ended = unsafe { cudarc::driver::result::stream::end_capture(stream.cu_stream()) };
    cleanup.active = false;
    let raw = ended.map_err(|e| format!("CAPTURE_CLEANUP_FAILED: {e}"))?;
    let mut graph = ProbeGraph {
        graph: raw,
        exec: std::ptr::null_mut(),
        stream: stream.clone(),
    };
    require(
        stream.capture_status().map_err(|e| e.to_string())?
            == ds::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
        "CAPTURE_CLEANUP_FAILED: still capturing",
    )?;
    result?;
    require(!raw.is_null(), "empty captured graph")?;
    graph.exec = unsafe {
        cudarc::driver::result::graph::instantiate(raw, cuda::graph_instantiate_flags())
            .map_err(|e| e.to_string())?
    };
    Ok(graph)
}
