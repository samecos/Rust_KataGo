//! Native CUDA FP32 execution of the research compact/dense StudentNet.
//!
//! Invocations exclusively borrow a stream, cuBLAS handle, and FP32 workspace
//! from a bounded idle pool. Resources are returned only after synchronization
//! and finite-output validation; concurrent calls never share a workspace.

use crate::student_model::StudentModel;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig, sys as blas_sys};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    LaunchConfig, PushKernelArg, sys::CUdevice_attribute,
};
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

include!(concat!(env!("OUT_DIR"), "/cuda_kernels.rs"));

// BEGIN CPU_ONLY_EXECUTION_REUSE
mod execution_reuse {
    use std::sync::Mutex;
    use std::thread::ThreadId;

    pub(super) const AREA: usize = 361;
    pub(super) const SPATIAL: usize = 22;
    pub(super) const GLOBAL: usize = 19;
    pub(super) const MAX_IDLE_EXECUTIONS: usize = 8;

    pub(super) fn element_count(factors: &[usize]) -> Result<usize, String> {
        let value = factors
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .ok_or("student CUDA tensor size overflow")?;
        if value > i32::MAX as usize {
            return Err(format!(
                "student CUDA tensor has {value} elements, exceeds i32 indexing"
            ));
        }
        Ok(value)
    }

    /// Sizes of the actual batch prefix, independent of retained capacity.
    #[derive(Debug, PartialEq, Eq)]
    pub(super) struct WorkspaceLayout {
        pub capacity: usize,
        pub input: usize,
        pub global: usize,
        pub trunk: usize,
        pub columns: usize,
        pub projected_global: usize,
        pub combined: usize,
        pub board: usize,
        pub pass: usize,
        pub policy: usize,
        pub features: usize,
        pub value: usize,
        pub score: usize,
        pub ownership: usize,
    }
    impl WorkspaceLayout {
        pub fn new(batch: usize, width: usize) -> Result<Self, String> {
            if batch == 0 || !matches!(width, 48 | 64) {
                return Err(
                    "student CUDA workspace requires positive batch and width 48 or 64".into(),
                );
            }
            Ok(Self {
                capacity: batch,
                input: element_count(&[batch, SPATIAL, AREA])?,
                global: element_count(&[batch, GLOBAL])?,
                trunk: element_count(&[batch, width, AREA])?,
                columns: element_count(&[batch, width, 9, AREA])?,
                projected_global: element_count(&[batch, width])?,
                combined: element_count(&[batch, width + GLOBAL])?,
                board: element_count(&[batch, AREA])?,
                pass: batch,
                policy: element_count(&[batch, AREA + 1])?,
                features: element_count(&[batch, 64])?,
                value: element_count(&[batch, 3])?,
                score: batch,
                ownership: element_count(&[batch, AREA])?,
            })
        }
    }

    /// Geometric growth stops at the signed-index limit of the largest tensor.
    pub(super) fn growth_capacity(
        current: usize,
        requested: usize,
        width: usize,
    ) -> Result<usize, String> {
        WorkspaceLayout::new(requested, width)?;
        let maximum = i32::MAX as usize / element_count(&[width, 9, AREA])?;
        if current > maximum {
            return Err("student CUDA retained workspace exceeds its index limit".into());
        }
        let mut capacity = current.max(1);
        while capacity < requested {
            capacity = capacity.saturating_mul(2).min(maximum);
        }
        Ok(capacity)
    }

    /// The lock covers only movement of idle resources, never CUDA execution.
    /// Removed resources belong exclusively to the caller until returned.
    pub(super) struct IdleExecutionPool<T> {
        idle: Mutex<Vec<(ThreadId, T)>>,
    }
    impl<T> IdleExecutionPool<T> {
        pub fn new() -> Self {
            Self {
                idle: Mutex::new(Vec::new()),
            }
        }
        pub fn take(&self, owner: ThreadId) -> Result<Option<T>, String> {
            let mut idle = self
                .idle
                .lock()
                .map_err(|_| "student CUDA idle pool lock poisoned")?;
            let index = idle
                .iter()
                .position(|(previous, _)| *previous == owner)
                .or_else(|| idle.len().checked_sub(1));
            Ok(index.map(|index| idle.swap_remove(index).1))
        }
        pub fn put(&self, owner: ThreadId, resource: T) -> Result<bool, String> {
            let mut idle = self
                .idle
                .lock()
                .map_err(|_| "student CUDA idle pool lock poisoned")?;
            if idle.len() < MAX_IDLE_EXECUTIONS {
                idle.push((owner, resource));
                return Ok(true);
            }
            // CUDA resource destruction may wait; keep it outside the pool lock.
            drop(idle);
            drop(resource);
            Ok(false)
        }
        pub fn len(&self) -> Result<usize, String> {
            Ok(self
                .idle
                .lock()
                .map_err(|_| "student CUDA idle pool lock poisoned")?
                .len())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::sync::{
            Arc, Barrier,
            atomic::{AtomicUsize, Ordering},
        };

        #[test]
        fn retained_capacity_survives_large_small_large_batches() {
            let mut capacity = 0;
            for batch in [32, 1, 8, 32, 1, 8, 32] {
                capacity = growth_capacity(capacity, batch, 64).unwrap();
                assert_eq!(capacity, 32);
                let active = WorkspaceLayout::new(batch, 64).unwrap();
                assert_eq!(active.input, batch * 22 * 361);
                assert_eq!(active.columns, batch * 64 * 9 * 361);
                assert_eq!(active.policy, batch * 362);
                assert_eq!(active.value, batch * 3);
                assert_eq!(active.score, batch);
                assert_eq!(active.ownership, batch * 361);
            }
        }

        #[test]
        fn geometric_growth_stops_at_signed_index_limit() {
            for width in [48, 64] {
                let maximum = i32::MAX as usize / (width * 9 * AREA);
                assert_eq!(growth_capacity(0, maximum, width).unwrap(), maximum);
                assert!(growth_capacity(0, maximum + 1, width).is_err());
                assert!(growth_capacity(maximum + 1, 1, width).is_err());
            }
            assert_eq!(growth_capacity(2, 3, 48).unwrap(), 4);
            assert!(growth_capacity(0, 0, 64).is_err());
            assert!(growth_capacity(0, 1, usize::MAX).is_err());
            assert!(element_count(&[usize::MAX, 2]).is_err());
        }

        #[test]
        fn every_active_tensor_fits_its_retained_prefix() {
            for width in [48, 64] {
                let retained = WorkspaceLayout::new(32, width).unwrap();
                for batch in [1, 3, 8, 31, 32] {
                    let active = WorkspaceLayout::new(batch, width).unwrap();
                    for (active, retained) in [
                        (active.input, retained.input),
                        (active.global, retained.global),
                        (active.trunk, retained.trunk),
                        (active.columns, retained.columns),
                        (active.projected_global, retained.projected_global),
                        (active.combined, retained.combined),
                        (active.board, retained.board),
                        (active.pass, retained.pass),
                        (active.policy, retained.policy),
                        (active.features, retained.features),
                        (active.value, retained.value),
                        (active.score, retained.score),
                        (active.ownership, retained.ownership),
                    ] {
                        assert!(active <= retained);
                    }
                    assert!(element_count(&[batch, SPATIAL, 9, AREA]).unwrap() <= active.columns);
                }
            }
        }

        #[test]
        fn prefers_original_thread_and_allows_idle_migration() {
            let pool = IdleExecutionPool::new();
            let own = std::thread::current().id();
            let other = std::thread::spawn(|| std::thread::current().id())
                .join()
                .unwrap();
            pool.put(own, 10).unwrap();
            pool.put(other, 20).unwrap();
            assert_eq!(pool.take(own).unwrap(), Some(10));
            assert_eq!(pool.take(own).unwrap(), Some(20));
            assert_eq!(pool.take(other).unwrap(), None);
        }

        #[test]
        fn idle_limit_releases_excess_outside_the_lock() {
            struct CheckDrop {
                pool: Arc<IdleExecutionPool<CheckDrop>>,
                dropped: Arc<AtomicUsize>,
            }
            impl Drop for CheckDrop {
                fn drop(&mut self) {
                    assert!(self.pool.idle.try_lock().is_ok());
                    self.dropped.fetch_add(1, Ordering::SeqCst);
                }
            }
            let pool = Arc::new(IdleExecutionPool::new());
            let dropped = Arc::new(AtomicUsize::new(0));
            let owner = std::thread::current().id();
            for index in 0..MAX_IDLE_EXECUTIONS + 3 {
                let retained = pool
                    .put(
                        owner,
                        CheckDrop {
                            pool: pool.clone(),
                            dropped: dropped.clone(),
                        },
                    )
                    .unwrap();
                assert_eq!(retained, index < MAX_IDLE_EXECUTIONS);
            }
            assert_eq!(pool.len().unwrap(), MAX_IDLE_EXECUTIONS);
            assert_eq!(dropped.load(Ordering::SeqCst), 3);
            // Drain the retained objects to avoid the deliberate Arc test cycle.
            while let Some(resource) = pool.take(owner).unwrap() {
                drop(resource);
            }
            assert_eq!(dropped.load(Ordering::SeqCst), MAX_IDLE_EXECUTIONS + 3);
        }

        #[test]
        fn concurrent_checkouts_are_exclusive_and_not_serialized() {
            let pool = Arc::new(IdleExecutionPool::new());
            let owner = std::thread::current().id();
            pool.put(owner, 11).unwrap();
            pool.put(owner, 22).unwrap();
            let barrier = Arc::new(Barrier::new(3));
            std::thread::scope(|scope| {
                let mut workers = Vec::new();
                for _ in 0..2 {
                    let pool = pool.clone();
                    let barrier = barrier.clone();
                    workers.push(scope.spawn(move || {
                        let id = std::thread::current().id();
                        let resource = pool.take(id).unwrap().unwrap();
                        barrier.wait();
                        barrier.wait();
                        pool.put(id, resource).unwrap();
                        resource
                    }));
                }
                barrier.wait();
                assert_eq!(pool.len().unwrap(), 0);
                barrier.wait();
                assert_ne!(
                    workers.remove(0).join().unwrap(),
                    workers.remove(0).join().unwrap()
                );
            });
            assert_eq!(pool.len().unwrap(), 2);
        }

        #[test]
        fn discarded_checkout_is_not_available_for_reuse() {
            let pool = IdleExecutionPool::new();
            let owner = std::thread::current().id();
            pool.put(owner, String::from("failed execution")).unwrap();
            let discarded = pool.take(owner).unwrap().unwrap();
            drop(discarded);
            assert!(pool.take(owner).unwrap().is_none());
        }

        #[test]
        fn poisoned_pool_reports_an_error() {
            let pool = Arc::new(IdleExecutionPool::<usize>::new());
            let worker_pool = pool.clone();
            assert!(
                std::thread::spawn(move || {
                    let _held = worker_pool.idle.lock().unwrap();
                    panic!("intentional test poison");
                })
                .join()
                .is_err()
            );
            let owner = std::thread::current().id();
            assert!(pool.take(owner).is_err());
            assert!(pool.put(owner, 1).is_err());
            assert!(pool.len().is_err());
        }
    }
}
// END CPU_ONLY_EXECUTION_REUSE
use execution_reuse::{
    AREA, GLOBAL, IdleExecutionPool, MAX_IDLE_EXECUTIONS, SPATIAL, WorkspaceLayout, element_count,
    growth_capacity,
};

/// Raw research outputs. Engine perspective/softmax/score-unit conversion is
/// performed by the evaluator adapter, outside the CUDA network itself.
#[derive(Debug)]
pub struct StudentOutputs {
    pub policy_logits: Vec<f32>,
    pub value_logits: Vec<f32>,
    pub score: Vec<f32>,
    pub ownership_logits: Vec<f32>,
}

pub struct StudentCudaModel {
    context: Arc<CudaContext>,
    // Retain the module for the complete lifetime of its kernel functions.
    _module: Arc<CudaModule>,
    functions: BTreeMap<&'static str, CudaFunction>,
    weights: BTreeMap<String, CudaSlice<f32>>,
    width: usize,
    blocks: usize,
    target_id: String,
    executions: IdleExecutionPool<StudentExecution>,
    execution_creations: AtomicUsize,
    execution_reuses: AtomicUsize,
    workspace_growths: AtomicUsize,
    discarded_executions: AtomicUsize,
    retired_executions: AtomicUsize,
}

struct StudentExecution {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    workspace: Option<StudentWorkspace>,
}

struct StudentWorkspace {
    capacity: usize,
    input: CudaSlice<f32>,
    global: CudaSlice<f32>,
    x: CudaSlice<f32>,
    intermediate: CudaSlice<f32>,
    residual: CudaSlice<f32>,
    columns: CudaSlice<f32>,
    projected_global: CudaSlice<f32>,
    combined: CudaSlice<f32>,
    board: CudaSlice<f32>,
    pass: CudaSlice<f32>,
    policy: CudaSlice<f32>,
    features: CudaSlice<f32>,
    value: CudaSlice<f32>,
    score: CudaSlice<f32>,
    ownership: CudaSlice<f32>,
}

impl StudentWorkspace {
    fn new(stream: &Arc<CudaStream>, capacity: usize, width: usize) -> Result<Self, String> {
        let layout = WorkspaceLayout::new(capacity, width)?;
        let allocate = |count| {
            // Every actual-batch prefix is overwritten before it is read:
            // im2col/linear/pool/pack kernels write their complete count;
            // convolution SGEMM uses beta=0. Capacity tails are never read.
            unsafe { stream.alloc(count) }
                .map_err(|e| format!("student CUDA allocate {count} FP32 elements: {e}"))
        };
        Ok(Self {
            capacity,
            input: allocate(layout.input)?,
            global: allocate(layout.global)?,
            x: allocate(layout.trunk)?,
            intermediate: allocate(layout.trunk)?,
            residual: allocate(layout.trunk)?,
            columns: allocate(layout.columns)?,
            projected_global: allocate(layout.projected_global)?,
            combined: allocate(layout.combined)?,
            board: allocate(layout.board)?,
            pass: allocate(layout.pass)?,
            policy: allocate(layout.policy)?,
            features: allocate(layout.features)?,
            value: allocate(layout.value)?,
            score: allocate(layout.score)?,
            ownership: allocate(layout.ownership)?,
        })
    }
}

impl StudentExecution {
    fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        let stream = context
            .new_stream()
            .map_err(|e| format!("student CUDA inference stream: {e}"))?;
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("student CUDA cuBLAS handle: {e}"))?;
        let status = unsafe {
            blas_sys::cublasSetMathMode(
                *blas.handle(),
                blas_sys::cublasMath_t::CUBLAS_PEDANTIC_MATH,
            )
        };
        if status != blas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!("student CUDA FP32 cuBLAS math mode: {status:?}"));
        }
        Ok(Self {
            stream,
            blas,
            workspace: None,
        })
    }

    fn ensure_capacity(&mut self, batch: usize, width: usize) -> Result<bool, String> {
        let current = self
            .workspace
            .as_ref()
            .map_or(0, |workspace| workspace.capacity);
        if current >= batch {
            return Ok(false);
        }
        let capacity = growth_capacity(current, batch, width)?;
        // Publish only a complete allocation; errors are drained by apply.
        self.workspace = Some(StudentWorkspace::new(&self.stream, capacity, width)?);
        Ok(true)
    }
}

fn expected_tensors(width: usize, blocks: usize) -> BTreeMap<String, Vec<usize>> {
    let mut expected = BTreeMap::new();
    let mut pair = |name: &str, shape: Vec<usize>, outputs: usize| {
        expected.insert(format!("{name}.weight"), shape);
        expected.insert(format!("{name}.bias"), vec![outputs]);
    };
    pair("stem", vec![width, SPATIAL, 3, 3], width);
    pair("global_to_stem", vec![width, GLOBAL], width);
    for block in 0..blocks {
        for layer in ["conv1", "conv2"] {
            pair(
                &format!("blocks.{block}.{layer}"),
                vec![width, width, 3, 3],
                width,
            );
        }
    }
    pair("policy_board", vec![1, width, 1, 1], 1);
    pair("policy_pass", vec![1, width + GLOBAL], 1);
    pair("value_features", vec![64, width + GLOBAL], 64);
    pair("value", vec![3, 64], 3);
    pair("score_head", vec![1, 64], 1);
    pair("ownership", vec![1, width, 1, 1], 1);
    expected
}

fn config(count: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(count as u32)
}

fn finite(values: &[f32], name: &str) -> Result<(), String> {
    if let Some(index) = values.iter().position(|v| !v.is_finite()) {
        return Err(format!(
            "student CUDA {name} contains nonfinite value at {index}"
        ));
    }
    Ok(())
}

impl StudentCudaModel {
    pub fn load(model: &StudentModel, device_id: usize) -> Result<Self, String> {
        let width = model.manifest.width;
        let blocks = model.manifest.blocks;
        let valid_architecture = match model.manifest.variant.as_str() {
            "compact" => width == 48 && blocks == 4,
            "dense" => width == 64 && blocks == 6,
            _ => false,
        };
        if !valid_architecture {
            return Err("student CUDA requires compact (48/4) or dense (64/6) architecture".into());
        }
        if model.manifest.format_version != 1
            || model.manifest.input_version != 7
            || model.manifest.score_scale != 20.0
        {
            return Err("student CUDA unsupported format, input or score contract".into());
        }
        let expected = expected_tensors(width, blocks);
        if expected.len() != model.tensors.len() {
            return Err("student CUDA tensor inventory differs from architecture".into());
        }
        for (name, shape) in &expected {
            let tensor = model
                .tensors
                .get(name)
                .ok_or_else(|| format!("student CUDA missing tensor {name}"))?;
            if &tensor.shape != shape || tensor.values.len() != element_count(shape)? {
                return Err(format!(
                    "student CUDA invalid shape for {name}: {:?}, expected {shape:?}",
                    tensor.shape
                ));
            }
            finite(&tensor.values, name)?;
        }

        let context = CudaContext::new(device_id)
            .map_err(|e| format!("student CUDA device {device_id} initialization: {e}"))?;
        let major = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| format!("student CUDA device capability: {e}"))?;
        let minor = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| format!("student CUDA device capability: {e}"))?;
        let capability = (major, minor);
        let selected = CUDA_TARGETS
            .iter()
            .filter_map(|(id, cap, kernels)| {
                let (a, b) = cap.split_once('.')?;
                let target_cap = (a.parse::<i32>().ok()?, b.parse::<i32>().ok()?);
                let ptx = kernels.iter().find(|(name, _)| *name == "student")?.1;
                (target_cap <= capability).then_some((target_cap, *id, ptx))
            })
            .max_by_key(|(cap, _, _)| *cap)
            .ok_or_else(|| {
                format!("student CUDA has no compiled PTX for device capability {major}.{minor}")
            })?;
        let ptx_text = std::str::from_utf8(selected.2)
            .map_err(|e| format!("student CUDA PTX encoding: {e}"))?;
        let module = context
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_text.to_owned()))
            .map_err(|e| format!("student CUDA load {} PTX: {e}", selected.1))?;
        let mut functions = BTreeMap::new();
        for name in [
            "student_im2col",
            "student_conv_bias",
            "student_global_relu",
            "student_residual_relu",
            "student_pool_concat",
            "student_linear",
            "student_pack_policy",
        ] {
            functions.insert(
                name,
                module
                    .load_function(name)
                    .map_err(|e| format!("student CUDA load kernel {name}: {e}"))?,
            );
        }
        let stream = context
            .new_stream()
            .map_err(|e| format!("student CUDA weight stream: {e}"))?;
        let mut weights = BTreeMap::new();
        for (name, tensor) in &model.tensors {
            weights.insert(
                name.clone(),
                stream
                    .clone_htod(&tensor.values)
                    .map_err(|e| format!("student CUDA upload tensor {name}: {e}"))?,
            );
        }
        stream
            .synchronize()
            .map_err(|e| format!("student CUDA weight upload completion: {e}"))?;
        Ok(Self {
            context,
            _module: module,
            functions,
            weights,
            width,
            blocks,
            target_id: selected.1.to_owned(),
            executions: IdleExecutionPool::new(),
            execution_creations: AtomicUsize::new(0),
            execution_reuses: AtomicUsize::new(0),
            workspace_growths: AtomicUsize::new(0),
            discarded_executions: AtomicUsize::new(0),
            retired_executions: AtomicUsize::new(0),
        })
    }

    /// Actual device/library identity used to construct the loaded execution
    /// profile. Query failures are reported instead of inventing identity.
    pub fn execution_facts(&self) -> Result<serde_json::Value, String> {
        let major = self
            .context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| format!("student CUDA query capability: {e}"))?;
        let minor = self
            .context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|e| format!("student CUDA query capability: {e}"))?;
        let uuid = self
            .context
            .uuid()
            .map_err(|e| format!("student CUDA query UUID: {e}"))?;
        let uuid = hex::encode(uuid.bytes.map(|b| b as u8));
        let stream = self
            .context
            .new_stream()
            .map_err(|e| format!("student CUDA identity stream: {e}"))?;
        let blas = CudaBlas::new(stream)
            .map_err(|e| format!("student CUDA identity cuBLAS handle: {e}"))?;
        let mut cublas_version = 0;
        let status = unsafe { blas_sys::cublasGetVersion_v2(*blas.handle(), &mut cublas_version) };
        if status != blas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(format!("student CUDA query cuBLAS version: {status:?}"));
        }
        let mut driver_version = 0;
        let status = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) };
        if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("student CUDA query driver version: {status:?}"));
        }
        Ok(serde_json::json!({
            "device_ordinal": self.context.ordinal(), "device_uuid": uuid,
            "gpu_name": self.context.name().map_err(|e| format!("student CUDA query GPU name: {e}"))?,
            "compute_capability": format!("{major}.{minor}"), "driver_version": driver_version,
            "cublas_version": cublas_version, "math_mode": "pedantic", "precision": "fp32",
            "ptx_target": self.target_id,
        }))
    }

    /// CPU-side resource-lifetime diagnostics, separate from model math and
    /// loaded device identity. Counters may advance during concurrent calls.
    pub fn execution_reuse_stats(&self) -> Result<serde_json::Value, String> {
        Ok(serde_json::json!({
            "idle_limit": MAX_IDLE_EXECUTIONS,
            "idle_executions": self.executions.len()?,
            "execution_creations": self.execution_creations.load(Ordering::Relaxed),
            "execution_reuses": self.execution_reuses.load(Ordering::Relaxed),
            "workspace_growths": self.workspace_growths.load(Ordering::Relaxed),
            "discarded_executions": self.discarded_executions.load(Ordering::Relaxed),
            "retired_executions": self.retired_executions.load(Ordering::Relaxed),
        }))
    }

    /// Input is NCHW [batch,22,19,19] plus [batch,19] global features.
    /// This method returns only after device work and host copies complete.
    pub fn apply(
        &self,
        spatial: &[f32],
        global: &[f32],
        batch: usize,
    ) -> Result<StudentOutputs, String> {
        if batch == 0 || batch > i32::MAX as usize {
            return Err("student CUDA batch must be positive and fit i32".into());
        }
        if spatial.len() != element_count(&[batch, SPATIAL, AREA])?
            || global.len() != element_count(&[batch, GLOBAL])?
        {
            return Err("student CUDA requires spatial [B,22,19,19] and global [B,19]".into());
        }
        let layout = WorkspaceLayout::new(batch, self.width)?;
        finite(spatial, "spatial input")?;
        finite(global, "global input")?;
        // cuBLAS resources may migrate between threads only while idle. Bind
        // the owning context before touching any reused handle or allocation.
        self.context
            .bind_to_thread()
            .map_err(|e| format!("student CUDA bind inference context: {e}"))?;
        let owner = std::thread::current().id();
        let mut execution = match self.executions.take(owner)? {
            Some(execution) => {
                self.execution_reuses.fetch_add(1, Ordering::Relaxed);
                execution
            }
            None => {
                let execution = StudentExecution::new(&self.context)?;
                self.execution_creations.fetch_add(1, Ordering::Relaxed);
                execution
            }
        };
        // Keep every host destination alive through unconditional completion,
        // including a failure after one or more asynchronous output copies.
        let mut outputs = StudentOutputs {
            policy_logits: vec![0.0; layout.policy],
            value_logits: vec![0.0; layout.value],
            score: vec![0.0; layout.score],
            ownership_logits: vec![0.0; layout.ownership],
        };
        let result = (|| {
            if execution.ensure_capacity(batch, self.width)? {
                self.workspace_growths.fetch_add(1, Ordering::Relaxed);
            }
            self.apply_inner(
                &execution.stream,
                &execution.blas,
                execution
                    .workspace
                    .as_mut()
                    .expect("successful ensure_capacity"),
                spatial,
                global,
                batch,
                &mut outputs,
            )
        })();
        let completion = execution
            .stream
            .synchronize()
            .map_err(|e| format!("student CUDA inference completion: {e}"));
        let status = match (result, completion) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(error), Err(completion)) => Err(format!("{error}; {completion}")),
        };
        let status = status.and_then(|()| {
            finite(&outputs.policy_logits, "policy output")?;
            finite(&outputs.value_logits, "value output")?;
            finite(&outputs.score, "score output")?;
            finite(&outputs.ownership_logits, "ownership output")
        });
        if let Err(error) = status {
            self.discarded_executions.fetch_add(1, Ordering::Relaxed);
            // Execution remains checked out and is destroyed after draining;
            // neither partial work nor nonfinite outputs can enter the pool.
            return Err(error);
        }
        if !self.executions.put(owner, execution)? {
            self.retired_executions.fetch_add(1, Ordering::Relaxed);
        }
        Ok(outputs)
    }

    fn function(&self, name: &'static str) -> &CudaFunction {
        // The complete function inventory is validated during load.
        &self.functions[name]
    }

    fn weight(&self, name: &str) -> &CudaSlice<f32> {
        // The exact tensor inventory is validated during load.
        &self.weights[name]
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        stream: &Arc<CudaStream>,
        blas: &CudaBlas,
        input: &CudaView<'_, f32>,
        columns: &mut CudaViewMut<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        name: &str,
        input_channels: usize,
        output_channels: usize,
        batch: usize,
        kernel: usize,
        relu: bool,
    ) -> Result<(), String> {
        let k = input_channels * kernel * kernel;
        if kernel == 3 {
            let count = element_count(&[batch, k, AREA])?;
            unsafe {
                stream
                    .launch_builder(self.function("student_im2col"))
                    .arg(input)
                    .arg(&mut *columns)
                    .arg(&(input_channels as i32))
                    .arg(&(batch as i32))
                    .launch(config(count))
            }
            .map_err(|e| format!("student CUDA {name} im2col: {e}"))?;
        }
        // Row-major [K,361] columns and [out,K] weights are interpreted as
        // column-major [361,K] and [K,out], producing NCHW [out,361].
        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: blas_sys::cublasOperation_t::CUBLAS_OP_N,
                transb: blas_sys::cublasOperation_t::CUBLAS_OP_N,
                m: AREA as i32,
                n: output_channels as i32,
                k: k as i32,
                alpha: 1.0,
                lda: AREA as i32,
                ldb: k as i32,
                beta: 0.0,
                ldc: AREA as i32,
            },
            batch_size: batch as i32,
            stride_a: (k * AREA) as i64,
            stride_b: 0,
            stride_c: (output_channels * AREA) as i64,
        };
        let weight = self.weight(&format!("{name}.weight"));
        let gemm_input = if kernel == 3 {
            columns.as_view()
        } else {
            input.slice(..)
        };
        unsafe { blas.gemm_strided_batched(cfg, &gemm_input, weight, output) }
            .map_err(|e| format!("student CUDA {name} FP32 convolution: {e}"))?;
        let count = element_count(&[batch, output_channels, AREA])?;
        unsafe {
            stream
                .launch_builder(self.function("student_conv_bias"))
                .arg(output)
                .arg(self.weight(&format!("{name}.bias")))
                .arg(&(output_channels as i32))
                .arg(&(count as i32))
                .arg(&(i32::from(relu)))
                .launch(config(count))
        }
        .map_err(|e| format!("student CUDA {name} convolution bias: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        name: &str,
        input_channels: usize,
        output_channels: usize,
        batch: usize,
        relu: bool,
    ) -> Result<(), String> {
        let count = element_count(&[batch, output_channels])?;
        unsafe {
            stream
                .launch_builder(self.function("student_linear"))
                .arg(input)
                .arg(self.weight(&format!("{name}.weight")))
                .arg(self.weight(&format!("{name}.bias")))
                .arg(output)
                .arg(&(input_channels as i32))
                .arg(&(output_channels as i32))
                .arg(&(batch as i32))
                .arg(&(i32::from(relu)))
                .launch(config(count))
        }
        .map_err(|e| format!("student CUDA {name} linear: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_inner(
        &self,
        stream: &Arc<CudaStream>,
        blas: &CudaBlas,
        workspace: &mut StudentWorkspace,
        spatial: &[f32],
        global: &[f32],
        batch: usize,
        outputs: &mut StudentOutputs,
    ) -> Result<(), String> {
        let layout = WorkspaceLayout::new(batch, self.width)?;
        let mut input = workspace.input.slice_mut(..layout.input);
        let mut global_input = workspace.global.slice_mut(..layout.global);
        let mut x = workspace.x.slice_mut(..layout.trunk);
        let mut intermediate = workspace.intermediate.slice_mut(..layout.trunk);
        let mut residual = workspace.residual.slice_mut(..layout.trunk);
        let mut columns = workspace.columns.slice_mut(..layout.columns);
        let mut projected_global = workspace
            .projected_global
            .slice_mut(..layout.projected_global);
        let mut combined = workspace.combined.slice_mut(..layout.combined);
        let mut board = workspace.board.slice_mut(..layout.board);
        let mut pass = workspace.pass.slice_mut(..layout.pass);
        let mut policy = workspace.policy.slice_mut(..layout.policy);
        let mut features = workspace.features.slice_mut(..layout.features);
        let mut value = workspace.value.slice_mut(..layout.value);
        let mut score = workspace.score.slice_mut(..layout.score);
        let mut ownership = workspace.ownership.slice_mut(..layout.ownership);
        stream
            .memcpy_htod(spatial, &mut input)
            .map_err(|e| format!("student CUDA input upload: {e}"))?;
        stream
            .memcpy_htod(global, &mut global_input)
            .map_err(|e| format!("student CUDA global upload: {e}"))?;
        let trunk_count = layout.trunk;
        self.conv(
            stream,
            blas,
            &input.as_view(),
            &mut columns,
            &mut x,
            "stem",
            SPATIAL,
            self.width,
            batch,
            3,
            false,
        )?;
        self.linear(
            stream,
            &global_input.as_view(),
            &mut projected_global,
            "global_to_stem",
            GLOBAL,
            self.width,
            batch,
            false,
        )?;
        unsafe {
            stream
                .launch_builder(self.function("student_global_relu"))
                .arg(&mut x)
                .arg(&projected_global.as_view())
                .arg(&(self.width as i32))
                .arg(&(trunk_count as i32))
                .launch(config(trunk_count))
        }
        .map_err(|e| format!("student CUDA stem global broadcast: {e}"))?;
        for block in 0..self.blocks {
            self.conv(
                stream,
                blas,
                &x.as_view(),
                &mut columns,
                &mut intermediate,
                &format!("blocks.{block}.conv1"),
                self.width,
                self.width,
                batch,
                3,
                true,
            )?;
            self.conv(
                stream,
                blas,
                &intermediate.as_view(),
                &mut columns,
                &mut residual,
                &format!("blocks.{block}.conv2"),
                self.width,
                self.width,
                batch,
                3,
                false,
            )?;
            unsafe {
                stream
                    .launch_builder(self.function("student_residual_relu"))
                    .arg(&mut residual)
                    .arg(&x.as_view())
                    .arg(&(trunk_count as i32))
                    .launch(config(trunk_count))
            }
            .map_err(|e| format!("student CUDA residual block {block}: {e}"))?;
            std::mem::swap(&mut x, &mut residual);
        }
        let combined_count = layout.combined;
        unsafe {
            stream
                .launch_builder(self.function("student_pool_concat"))
                .arg(&x.as_view())
                .arg(&global_input.as_view())
                .arg(&mut combined)
                .arg(&(self.width as i32))
                .arg(&(batch as i32))
                .launch(config(combined_count))
        }
        .map_err(|e| format!("student CUDA pool/concat: {e}"))?;
        self.conv(
            stream,
            blas,
            &x.as_view(),
            &mut columns,
            &mut board,
            "policy_board",
            self.width,
            1,
            batch,
            1,
            false,
        )?;
        self.linear(
            stream,
            &combined.as_view(),
            &mut pass,
            "policy_pass",
            self.width + GLOBAL,
            1,
            batch,
            false,
        )?;
        unsafe {
            stream
                .launch_builder(self.function("student_pack_policy"))
                .arg(&board.as_view())
                .arg(&pass.as_view())
                .arg(&mut policy)
                .arg(&(batch as i32))
                .launch(config(batch * 362))
        }
        .map_err(|e| format!("student CUDA policy packing: {e}"))?;
        self.linear(
            stream,
            &combined.as_view(),
            &mut features,
            "value_features",
            self.width + GLOBAL,
            64,
            batch,
            true,
        )?;
        self.linear(
            stream,
            &features.as_view(),
            &mut value,
            "value",
            64,
            3,
            batch,
            false,
        )?;
        self.linear(
            stream,
            &features.as_view(),
            &mut score,
            "score_head",
            64,
            1,
            batch,
            false,
        )?;
        self.conv(
            stream,
            blas,
            &x.as_view(),
            &mut columns,
            &mut ownership,
            "ownership",
            self.width,
            1,
            batch,
            1,
            false,
        )?;
        let copy = |tensor: &CudaView<'_, f32>, destination: &mut Vec<f32>, name: &str| {
            stream
                .memcpy_dtoh(tensor, destination)
                .map_err(|e| format!("student CUDA copy {name}: {e}"))
        };
        copy(&policy.as_view(), &mut outputs.policy_logits, "policy")?;
        copy(&value.as_view(), &mut outputs.value_logits, "value")?;
        copy(&score.as_view(), &mut outputs.score, "score")?;
        copy(
            &ownership.as_view(),
            &mut outputs.ownership_logits,
            "ownership",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_sizes_reject_integer_overflow() {
        assert_eq!(element_count(&[2, 22, 361]).unwrap(), 15_884);
        assert!(element_count(&[usize::MAX, 2]).is_err());
        assert!(element_count(&[i32::MAX as usize, 2]).is_err());
    }

    #[test]
    fn fixed_architectures_have_full_tensor_inventories() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StudentCudaModel>();
        let compact = expected_tensors(48, 4);
        assert_eq!(compact.len(), 32);
        assert_eq!(compact["blocks.3.conv2.weight"], [48, 48, 3, 3]);
        let dense = expected_tensors(64, 6);
        assert_eq!(dense.len(), 40);
        assert_eq!(dense["value_features.weight"], [64, 83]);
    }

    #[test]
    #[ignore = "requires a CUDA GPU and STUDENT_CUDA_TEST_MODEL pointing to an exported fixture"]
    fn gpu_shape_finite_batch_and_concurrent_call_contract() {
        let path =
            std::env::var("STUDENT_CUDA_TEST_MODEL").expect("STUDENT_CUDA_TEST_MODEL required");
        let parsed = StudentModel::parse(&std::fs::read(path).unwrap()).unwrap();
        let model = StudentCudaModel::load(&parsed, 0).unwrap();
        let spatial: Vec<f32> = (0..SPATIAL * AREA)
            .map(|i| (i % 17) as f32 / 17.0)
            .collect();
        let global: Vec<f32> = (0..GLOBAL).map(|i| (i as f32 - 9.0) / 9.0).collect();
        assert!(model.apply(&spatial, &global, 0).is_err());
        assert!(
            model
                .apply(&spatial[..spatial.len() - 1], &global, 1)
                .is_err()
        );
        assert!(
            model
                .apply(&spatial, &global[..global.len() - 1], 1)
                .is_err()
        );
        let mut nonfinite = global.clone();
        nonfinite[2] = f32::INFINITY;
        assert!(model.apply(&spatial, &nonfinite, 1).is_err());
        let mut nonfinite = spatial.clone();
        nonfinite[13] = f32::NAN;
        assert!(model.apply(&nonfinite, &global, 1).is_err());
        let baseline = model.apply(&spatial, &global, 1).unwrap();
        let compare = |expected: &StudentOutputs, actual: &StudentOutputs, copies: usize| {
            for (reference, output) in [
                (&expected.policy_logits, &actual.policy_logits),
                (&expected.value_logits, &actual.value_logits),
                (&expected.score, &actual.score),
                (&expected.ownership_logits, &actual.ownership_logits),
            ] {
                assert_eq!(output.len(), reference.len() * copies);
                for chunk in output.chunks_exact(reference.len()) {
                    for (a, b) in reference.iter().zip(chunk) {
                        assert!(
                            (a - b).abs() <= 2e-5,
                            "batch/concurrent difference: {a} vs {b}"
                        );
                    }
                }
            }
        };
        std::thread::scope(|scope| {
            let first = scope.spawn(|| model.apply(&spatial, &global, 1).unwrap());
            let second = scope.spawn(|| model.apply(&spatial, &global, 1).unwrap());
            compare(&baseline, &first.join().unwrap(), 1);
            compare(&baseline, &second.join().unwrap(), 1);
        });
        let batched = model
            .apply(&spatial.repeat(3), &global.repeat(3), 3)
            .unwrap();
        compare(&baseline, &batched, 3);
    }
}
