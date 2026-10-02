//! Native adapter for int8_algorithm_plan. Compiled as a child of int8.rs so
//! descriptor ownership remains with the existing LtPlan / workspace.
use super::{check, CudaRuntime, CudaStream, Int8Workspace, LtPlan};
use crate::backends::int8_algorithm_plan as plan;
use cudarc::cublaslt::sys;
use cudarc::driver::sys as driver_sys;
use std::sync::Arc;

fn read<T: Copy + Default>(name: &str, get: impl FnOnce(*mut std::ffi::c_void, usize, *mut usize) -> sys::cublasStatus_t) -> Result<T, String> {
    let mut value = T::default();
    let mut written = 0usize;
    check(get(&mut value as *mut T as *mut _, std::mem::size_of::<T>(), &mut written), name)?;
    if written != std::mem::size_of::<T>() {
        return Err(format!("INT8 {name}: native ABI returned {written} bytes, expected {}", std::mem::size_of::<T>()));
    }
    Ok(value)
}

fn desc<T: Copy + Default>(p: &LtPlan, attr: sys::cublasLtMatmulDescAttributes_t) -> Result<T, String> {
    read("read matmul descriptor", |ptr, len, written| unsafe {
        sys::cublasLtMatmulDescGetAttribute(p.desc, attr, ptr, len, written)
    })
}
fn layout_value<T: Copy + Default>(layout: sys::cublasLtMatrixLayout_t, attr: sys::cublasLtMatrixLayoutAttribute_t) -> Result<T, String> {
    read("read matrix layout", |ptr, len, written| unsafe {
        sys::cublasLtMatrixLayoutGetAttribute(layout, attr, ptr, len, written)
    })
}
fn config<T: Copy + Default>(algo: &sys::cublasLtMatmulAlgo_t, attr: sys::cublasLtMatmulAlgoConfigAttributes_t) -> Result<T, String> {
    read("read algorithm configuration", |ptr, len, written| unsafe {
        sys::cublasLtMatmulAlgoConfigGetAttribute(algo, attr, ptr, len, written)
    })
}
fn optional_u16(algo: &sys::cublasLtMatmulAlgo_t, attr: sys::cublasLtMatmulAlgoConfigAttributes_t) -> Result<plan::Queried<u16>, String> {
    let mut value = 0u16;
    let mut written = 0usize;
    let status = unsafe { sys::cublasLtMatmulAlgoConfigGetAttribute(algo, attr,
        &mut value as *mut _ as *mut _, std::mem::size_of::<u16>(), &mut written) };
    if status == sys::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED { return Ok(plan::Queried::NotSupported); }
    check(status, "read optional u16 algorithm configuration")?;
    if written != std::mem::size_of::<u16>() { return Err("INT8 optional configuration native ABI mismatch".into()); }
    Ok(plan::Queried::Value(value))
}
fn cap<T: Copy + Default>(algo: &sys::cublasLtMatmulAlgo_t, attr: sys::cublasLtMatmulAlgoCapAttributes_t) -> Result<T, String> {
    read("read algorithm capability", |ptr, len, written| unsafe {
        sys::cublasLtMatmulAlgoCapGetAttribute(algo, attr, ptr, len, written)
    })
}

fn read_layout(layout: sys::cublasLtMatrixLayout_t) -> Result<plan::Layout, String> {
    use sys::cublasLtMatrixLayoutAttribute_t as A;
    Ok(plan::Layout {
        data_type: layout_value::<u32>(layout, A::CUBLASLT_MATRIX_LAYOUT_TYPE)?,
        order: layout_value::<i32>(layout, A::CUBLASLT_MATRIX_LAYOUT_ORDER)?,
        rows: layout_value::<u64>(layout, A::CUBLASLT_MATRIX_LAYOUT_ROWS)?,
        columns: layout_value::<u64>(layout, A::CUBLASLT_MATRIX_LAYOUT_COLS)?,
        leading_dimension: layout_value::<i64>(layout, A::CUBLASLT_MATRIX_LAYOUT_LD)?,
        batch_count: layout_value::<i32>(layout, A::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT)?,
        batch_stride: layout_value::<i64>(layout, A::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)?,
        plane_offset: layout_value::<i64>(layout, A::CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET)?,
        batch_mode: layout_value::<u32>(layout, A::CUBLASLT_MATRIX_LAYOUT_BATCH_MODE)?,
    })
}

impl LtPlan {
    fn read_descriptors(&self) -> Result<plan::Descriptors, String> {
        use sys::cublasLtMatmulDescAttributes_t as A;
        // Pointer attributes are inspected but never serialized. They must all
        // be NULL in this plain integer GEMM contract; no bias/scale side input.
        for attr in [A::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            A::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
            A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            A::CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, A::CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
            A::CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, A::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
            A::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER] {
            if desc::<usize>(self, attr)? != 0 { return Err("INT8 persistence rejects non-null auxiliary pointer".into()); }
        }
        Ok(plan::Descriptors {
            compute_type: desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_COMPUTE_TYPE)?,
            scale_type: desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_SCALE_TYPE)?,
            pointer_mode: desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_POINTER_MODE)?,
            transpose_a: desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_TRANSA)?,
            transpose_b: desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_TRANSB)?,
            transpose_c: desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_TRANSC)?,
            fill_mode: desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_FILL_MODE)?,
            epilogue: desc::<u32>(self, A::CUBLASLT_MATMUL_DESC_EPILOGUE)?,
            sm_count_target: desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET)?,
            fast_accum: desc::<i8>(self, A::CUBLASLT_MATMUL_DESC_FAST_ACCUM)?,
            scale_modes: [desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE)?,
                desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE)?,
                desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_C_SCALE_MODE)?,
                desc::<i32>(self, A::CUBLASLT_MATMUL_DESC_D_SCALE_MODE)?],
            auxiliary_pointers_null: true, alpha: 1, beta: 0, c_d_alias: true,
            layouts: [read_layout(self.a)?, read_layout(self.b)?, read_layout(self.c)?, read_layout(self.c)?],
        })
    }

    /// Called after the existing final algorithm choice, without requesting a
    /// second heuristic or modifying any algorithm attribute.
    pub(super) fn snapshot_algorithm(&self, rt: &CudaRuntime, key: plan::ProblemKey) -> Result<plan::Implementation, String> {
        use sys::cublasLtMatmulAlgoConfigAttributes_t as C;
        use sys::cublasLtMatmulAlgoCapAttributes_t as A;
        let mut checked: sys::cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
        check(unsafe { sys::cublasLtMatmulAlgoCheck(rt.cublaslt_handle().ok_or("INT8 requires cuBLASLt")?,
            self.desc, self.a, self.b, self.c, self.c, &self.algo, &mut checked) }, "AlgoCheck persisted algorithm")?;
        check(checked.state, "AlgoCheck result state")?;
        let native_bytes: Vec<u8> = self.algo.data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let implementation = plan::Implementation {
            descriptors: self.read_descriptors()?, opaque_algorithm_hex: hex::encode(native_bytes),
            attributes: plan::AlgorithmAttributes {
                algorithm_id: config::<i32>(&self.algo, C::CUBLASLT_ALGO_CONFIG_ID)?,
                tile_id: config::<u32>(&self.algo, C::CUBLASLT_ALGO_CONFIG_TILE_ID)?,
                stages_id: config::<u32>(&self.algo, C::CUBLASLT_ALGO_CONFIG_STAGES_ID)?,
                split_k: config::<i32>(&self.algo, C::CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
                reduction_scheme: config::<u32>(&self.algo, C::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?,
                cta_swizzling: config::<u32>(&self.algo, C::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)?,
                custom_option: config::<u32>(&self.algo, C::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?,
                inner_shape_id: optional_u16(&self.algo, C::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)?,
                cluster_shape_id: optional_u16(&self.algo, C::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)?,
                numerical_implementation_flags: cap::<u64>(&self.algo, A::CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS)?,
            },
            minimum_alignment_bytes: [cap::<u32>(&self.algo, A::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES)?,
                cap::<u32>(&self.algo, A::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES)?,
                cap::<u32>(&self.algo, A::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES)?,
                cap::<u32>(&self.algo, A::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES)?],
            required_workspace_bytes: checked.workspaceSize as u64,
            provided_workspace_bytes: rt.cublaslt_workspace_len() as u64,
            workspace_alignment_bytes: 256, workspace_ownership: "runtime-per-stream-exclusive".into(),
        };
        implementation.validate(key)?;
        Ok(implementation)
    }

    /// Rebuild descriptors, install exact native bytes, then re-read and compare
    /// everything. This path never calls AlgoGetHeuristic, AlgoInit or tuning.
    pub(super) fn restore_algorithm(rt: &CudaRuntime, key: plan::ProblemKey, expected: &plan::Implementation) -> Result<Self, String> {
        expected.validate(key)?;
        let (m, n, k) = key.cache_key();
        let mut p = Self::descriptors(m, n, k)?;
        let bytes = hex::decode(&expected.opaque_algorithm_hex).map_err(|e| e.to_string())?;
        for (word, chunk) in p.algo.data.iter_mut().zip(bytes.chunks_exact(8)) {
            *word = u64::from_ne_bytes(chunk.try_into().map_err(|_| "INT8 opaque word length")?);
        }
        let actual = p.snapshot_algorithm(rt, key)?;
        if &actual != expected { return Err("INT8 restored algorithm/descriptors/capabilities differ; no heuristic fallback".into()); }
        p.persisted_algorithm = Some(actual);
        Ok(p)
    }
}

fn no_capture(stream: &Arc<CudaStream>) -> Result<(), String> {
    if crate::backends::cuda_exec::capturing() || stream.capture_status().map_err(|e| e.to_string())?
        != driver_sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        return Err("INT8 algorithm plan installation/export is forbidden during capture".into());
    }
    Ok(())
}

fn validate_runtime(rt: &CudaRuntime, stream: &Arc<CudaStream>, binding: &plan::Binding) -> Result<(), String> {
    binding.validate()?;
    if !Arc::ptr_eq(stream.context(), &rt.device) { return Err("INT8 plan stream belongs to another context".into()); }
    rt.device.bind_to_thread().map_err(|e| e.to_string())?;
    let (major, minor) = rt.device.compute_capability().map_err(|e| e.to_string())?;
    let sm_count = rt.device.attribute(driver_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(|e| e.to_string())?;
    let mut driver = 0;
    let status = unsafe { driver_sys::cuDriverGetVersion(&mut driver) };
    if status != driver_sys::cudaError_enum::CUDA_SUCCESS { return Err(format!("INT8 plan driver version: {status:?}")); }
    let uuid = rt.device.uuid().map_err(|e| e.to_string())?;
    let uuid_hex = hex::encode(uuid.bytes.map(|v| v as u8));
    let gpu_uuid = format!("GPU-{}-{}-{}-{}-{}", &uuid_hex[..8], &uuid_hex[8..12], &uuid_hex[12..16], &uuid_hex[16..20], &uuid_hex[20..]);
    let device_sha256 = super::algorithm_device_fingerprint_sha256(rt)?;
    let host_abi = format!("{}-{}-{}", std::env::consts::ARCH, std::env::consts::OS,
        if cfg!(target_env = "msvc") { "msvc" } else if cfg!(target_env = "gnu") { "gnu" } else { "other" });
    if binding.cublaslt_version != unsafe { sys::cublasLtGetVersion() } as u64
        || binding.compute_capability != [major as u32, minor as u32]
        || binding.sm_count != sm_count as u32 || binding.driver_api_version != driver
        || binding.gpu_uuid != gpu_uuid || binding.device_fingerprint_sha256 != device_sha256
        || binding.pointer_width_bits != usize::BITS || binding.little_endian != cfg!(target_endian = "little")
        || binding.host_abi != host_abi
        || binding.algo_size_bytes as usize != std::mem::size_of::<sys::cublasLtMatmulAlgo_t>()
        || binding.algo_alignment_bytes as usize != std::mem::align_of::<sys::cublasLtMatmulAlgo_t>() {
        return Err("INT8 plan actual device/driver/LT-version/native-ABI mismatch".into());
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    if crate::tactic_plan::sha256_file(&exe)? != binding.executable_sha256 {
        return Err("INT8 plan actual executable mismatch".into());
    }
    Ok(())
}

impl Int8Workspace {
    pub(super) fn validate_algorithm_owner(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
        plan::validate_owner_identity(&self.algorithm_owner_stream, stream,
            self.algorithm_owner_stream.context(), stream.context(),
            [self.quantized.stream(), self.scales.stream(), self.dots.stream()],
            [self.quantized.context(), self.scales.context(), self.dots.context()])
    }

    pub(super) fn validate_algorithm_buffer_context<T>(&self, buffer: &cudarc::driver::CudaSlice<T>) -> Result<(), String> {
        if self.algorithm_session.is_some() {
            plan::require_same_context(self.algorithm_owner_stream.context(), buffer.context())?;
        }
        Ok(())
    }

    /// Covers input/shape validation, all pre/post kernels and integer GEMMs.
    /// No operation may continue after an installed session has been poisoned.
    pub(super) fn algorithm_operation<T>(&mut self, rt: &CudaRuntime, stream: &Arc<CudaStream>,
        operation: impl FnOnce(&mut Self) -> Result<T, String>) -> Result<T, String> {
        let result = (|| {
            if let Some(session) = &self.algorithm_session {
                session.ensure_healthy()?;
                self.validate_algorithm_owner(stream)?;
                plan::require_same_context(self.algorithm_owner_stream.context(), &rt.device)?;
            }
            operation(self)
        })();
        self.complete_algorithm_operation(result)
    }

    /// The model forward boundary calls both entry and completion, including
    /// failures in FP16/attention/head code outside public INT8 operations.
    pub(crate) fn algorithm_forward_entry(&mut self, rt: &CudaRuntime, stream: &Arc<CudaStream>) -> Result<(), String> {
        self.algorithm_operation(rt, stream, |_| Ok(()))
    }

    pub(crate) fn complete_algorithm_operation<T>(&mut self, result: Result<T, String>) -> Result<T, String> {
        if let Some(session) = &mut self.algorithm_session { session.complete_operation(result) } else { result }
    }

    fn can_install_algorithm_plan(&self, rt: &CudaRuntime, stream: &Arc<CudaStream>, binding: &plan::Binding) -> Result<(), String> {
        self.validate_algorithm_owner(stream)?;
        no_capture(&self.algorithm_owner_stream)?;
        if !self.plans.is_empty() || self.algorithm_session.is_some() {
            return Err("INT8 plan requires a fresh workspace with no previous cache/session".into());
        }
        if self.kernels.gemm_tune { return Err("INT8 persistence v1 does not authorize GEMM tuning".into()); }
        validate_runtime(rt, stream, binding)
    }

    /// Install before the first inference or warmup on this workspace. Inventory
    /// is caller-owned graph lowering output, not inferred from visited shapes.
    pub fn begin_algorithm_recording(&mut self, rt: &CudaRuntime, stream: &Arc<CudaStream>,
        binding: plan::Binding, inventory: Vec<plan::ProblemKey>) -> Result<(), String> {
        self.can_install_algorithm_plan(rt, stream, &binding)?;
        self.algorithm_session = Some(plan::Session::record(binding, inventory)?);
        Ok(())
    }

    /// Strict restore on a new workspace. Missing entries and runtime mismatch
    /// fail closed. No existing cache or old model identity is inherited.
    pub fn install_algorithm_plan(&mut self, rt: &CudaRuntime, stream: &Arc<CudaStream>,
        bytes: &[u8], expected_sha256: &str, expected_binding: &plan::Binding,
        expected_inventory: &[plan::ProblemKey]) -> Result<(), String> {
        self.can_install_algorithm_plan(rt, stream, expected_binding)?;
        let artifact = plan::Plan::parse_exact(bytes, expected_sha256, expected_binding, expected_inventory)?;
        self.algorithm_session = Some(plan::Session::restore(artifact)?);
        Ok(())
    }

    /// Synchronize this owning stream once, then require exact inventory
    /// coverage and return canonical bytes. This is a diagnostic/setup API;
    /// it is not legal inside a performance window or graph capture.
    /// A successful export proves this INT8 cache's launch coverage only.
    pub fn export_algorithm_plan(&mut self, stream: &Arc<CudaStream>) -> Result<Vec<u8>, String> {
        let result = (|| {
            self.validate_algorithm_owner(stream)?;
            no_capture(&self.algorithm_owner_stream)?;
            let session = self.algorithm_session.as_ref().ok_or("INT8 algorithm persistence was not installed")?;
            session.ensure_healthy()?;
            // Never synchronize a caller-provided stream after comparing only
            // its raw handle. This is the actual retained allocation owner.
            self.algorithm_owner_stream.synchronize().map_err(|e| e.to_string())?;
            session.export()?.canonical_bytes()
        })();
        self.complete_algorithm_operation(result)
    }
}
