//! Test-only, coordinator-owned cache policy lifetime. No backend/math changes.
//! Both workspace allocations must outlive finish (and this guard's Drop).
use super::Lane;
use cudarc::driver::{CudaStream, DevicePtr, sys};
use kata_nn::backends::cuda::CudaRuntime;
use serde_json::{Value, json};
use std::sync::Arc;

const WINDOW_BYTES: usize = 14 * 361 * 384 * 4;
const TOTAL_BYTES: usize = 2 * WINDOW_BYTES;
const LIMIT: sys::CUlimit = sys::CUlimit::CU_LIMIT_PERSISTING_L2_CACHE_SIZE;
const ATTR: sys::CUstreamAttrID =
    sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW;

fn checked(code: sys::CUresult, what: &str) -> Result<(), String> {
    if code == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {code:?}"))
    }
}

fn normal() -> sys::CUaccessPolicyWindow {
    sys::CUaccessPolicyWindow {
        base_ptr: std::ptr::null_mut(),
        num_bytes: 0,
        hitRatio: 0.0,
        hitProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_NORMAL,
        missProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_NORMAL,
    }
}

fn get(stream: &CudaStream) -> Result<sys::CUaccessPolicyWindow, String> {
    let mut value: sys::CUstreamAttrValue = unsafe { std::mem::zeroed() };
    checked(
        unsafe { sys::cuStreamGetAttribute(stream.cu_stream(), ATTR, &mut value) },
        "get access-policy window",
    )?;
    Ok(unsafe { value.accessPolicyWindow })
}

fn set(stream: &CudaStream, window: sys::CUaccessPolicyWindow) -> Result<(), String> {
    let mut value: sys::CUstreamAttrValue = unsafe { std::mem::zeroed() };
    value.accessPolicyWindow = window;
    checked(
        unsafe { sys::cuStreamSetAttribute(stream.cu_stream(), ATTR, &value) },
        "set access-policy window",
    )
}

fn same(a: &sys::CUaccessPolicyWindow, b: &sys::CUaccessPolicyWindow) -> bool {
    a.base_ptr == b.base_ptr
        && a.num_bytes == b.num_bytes
        && a.hitRatio.to_bits() == b.hitRatio.to_bits()
        && a.hitProp == b.hitProp
        && a.missProp == b.missProp
}

fn describe(lane: usize, stream: &CudaStream, value: &sys::CUaccessPolicyWindow) -> Value {
    json!({"lane":lane,"stream_id":stream.cu_stream() as usize,
        "base_ptr":value.base_ptr as usize,"num_bytes":value.num_bytes,
        "hit_ratio":value.hitRatio,"hit_ratio_bits":value.hitRatio.to_bits(),
        "hit_prop":format!("{:?}",value.hitProp),
        "miss_prop":format!("{:?}",value.missProp)})
}

pub(super) struct CacheScope<'a> {
    rt: &'a CudaRuntime,
    streams: Vec<Arc<CudaStream>>,
    old_windows: Vec<sys::CUaccessPolicyWindow>,
    old_limit: usize,
    finished: bool,
    meta: Value,
}

impl<'a> CacheScope<'a> {
    pub(super) fn begin(
        rt: &'a CudaRuntime,
        lanes: &[Lane],
        enabled: bool,
    ) -> Result<Self, String> {
        if lanes.len() != 2 {
            return Err("L2 diagnostic requires exactly two lanes".into());
        }
        rt.device
            .bind_to_thread()
            .map_err(|e| format!("bind coordinator: {e}"))?;
        rt.device
            .synchronize()
            .map_err(|e| format!("pre-policy sync: {e}"))?;
        let mut pointers = Vec::new();
        let mut streams = Vec::new();
        for lane in lanes {
            if lane.workspace.batch() != 14 || lane.workspace.act384.len() * 4 != WINDOW_BYTES {
                return Err("L2 diagnostic requires exact B14 act384 FP32 allocation".into());
            }
            let (pointer, guard) = lane.workspace.act384.device_ptr(&lane.stream);
            if pointer == 0
                || pointers.contains(&pointer)
                || streams
                    .iter()
                    .any(|s: &Arc<CudaStream>| s.cu_stream() == lane.stream.cu_stream())
            {
                return Err(
                    "L2 lanes require distinct nonnull act384 allocations and streams".into(),
                );
            }
            pointers.push(pointer);
            streams.push(lane.stream.clone());
            drop(guard);
        }
        // DevicePtr guards may record synchronization events when dropped.
        rt.device
            .synchronize()
            .map_err(|e| format!("post-pointer-guard sync: {e}"))?;
        let old_limit = rt
            .device
            .get_limit(LIMIT)
            .map_err(|e| format!("save L2 limit: {e}"))?;
        let old_windows = streams
            .iter()
            .map(|s| get(s))
            .collect::<Result<Vec<_>, _>>()?;
        let mut guard = Self {
            rt,
            streams,
            old_windows,
            old_limit,
            finished: false,
            meta: json!({"requested":enabled,"physical_batch":14,"lanes":2,
                "per_lane_bytes":WINDOW_BYTES,"total_window_bytes":TOTAL_BYTES,
                "original_limit_bytes":old_limit,"cleanup":{"status":"NOT_FINISHED"}}),
        };
        // Construct the cleanup guard before the first mutation. Failed setup
        // synchronizes and restores through Drop, and cannot produce PASS.
        let setup = guard.configure(&pointers, enabled);
        if let Err(error) = setup {
            let cleanup = guard.finish();
            return Err(format!("cache policy setup: {error}; cleanup={cleanup:?}"));
        }
        Ok(guard)
    }

    fn configure(&mut self, pointers: &[u64], enabled: bool) -> Result<(), String> {
        let max_persisting = self
            .rt
            .device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE)
            .map_err(|e| format!("query maximum set-aside: {e}"))?
            as usize;
        let max_window = self
            .rt
            .device
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE)
            .map_err(|e| format!("query maximum window: {e}"))? as usize;
        if max_window < WINDOW_BYTES || max_persisting == 0 {
            return Err("device cannot support this registered L2 diagnostic".into());
        }
        // Both arms receive the same reset before warmup. This resets persisting
        // status, not a claim to flush all L2 data or create a cold-cache test.
        for stream in &self.streams {
            set(stream, normal())?;
        }
        checked(
            unsafe { sys::cuCtxResetPersistingL2Cache() },
            "pre-arm persisting reset",
        )?;
        self.rt
            .device
            .synchronize()
            .map_err(|e| format!("post-reset sync: {e}"))?;
        let request = TOTAL_BYTES.min(max_persisting);
        if enabled {
            // Exactly one context-global set, sized for BOTH active windows.
            self.rt
                .device
                .set_limit(LIMIT, request)
                .map_err(|e| format!("set sum L2 limit: {e}"))?;
        }
        let actual = self
            .rt
            .device
            .get_limit(LIMIT)
            .map_err(|e| format!("read actual L2 limit: {e}"))?;
        if enabled && actual == 0 {
            return Err("candidate effective L2 limit is zero".into());
        }
        let ratio = if enabled {
            (actual as f64 / TOTAL_BYTES as f64).min(1.0) as f32
        } else {
            0.0
        };
        let mut effective = Vec::new();
        for (i, stream) in self.streams.iter().enumerate() {
            let desired = if enabled {
                sys::CUaccessPolicyWindow {
                    base_ptr: pointers[i] as *mut std::ffi::c_void,
                    num_bytes: WINDOW_BYTES,
                    hitRatio: ratio,
                    hitProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_PERSISTING,
                    missProp: sys::CUaccessProperty::CU_ACCESS_PROPERTY_STREAMING,
                }
            } else {
                normal()
            };
            set(stream, desired)?;
            let readback = get(stream)?;
            if !same(&desired, &readback) {
                return Err(format!("lane {i} policy readback mismatch"));
            }
            effective.push(describe(i, stream, &readback));
        }
        self.meta["original_windows"] = json!(
            self.streams
                .iter()
                .zip(&self.old_windows)
                .enumerate()
                .map(|(i, (s, w))| describe(i, s, w))
                .collect::<Vec<_>>()
        );
        self.meta["max_persisting_bytes"] = json!(max_persisting);
        self.meta["max_access_window_bytes"] = json!(max_window);
        self.meta["requested_limit_bytes"] = json!(if enabled { request } else { self.old_limit });
        self.meta["context_set_limit_calls_during_setup"] = json!(usize::from(enabled));
        self.meta["effective_limit_bytes"] = json!(actual);
        self.meta["derived_hit_ratio"] = json!(ratio);
        self.meta["act384_base_ptrs"] = json!(pointers);
        self.meta["windows"] = json!(effective);
        self.meta["pre_arm_reset_and_sync"] = json!(true);
        self.meta["policy_scope"] = json!(
            "act384 only on two fixed B14 streams; cache hint, no mathematical change; both arms reset before warmup"
        );
        Ok(())
    }

    pub(super) fn metadata(&self) -> Value {
        self.meta.clone()
    }

    pub(super) fn finish(&mut self) -> Result<Value, String> {
        if self.finished {
            return Ok(self.meta.clone());
        }
        self.rt
            .device
            .bind_to_thread()
            .map_err(|e| format!("bind cleanup context: {e}"))?;
        // Always attempt to join GPU completion on BOTH streams. Never clear or
        // restore policy while either stream might still be using its buffers.
        let mut errors = Vec::new();
        for (i, stream) in self.streams.iter().enumerate() {
            if let Err(e) = stream.synchronize() {
                errors.push(format!("lane {i} cleanup sync: {e}"));
            }
        }
        if let Err(e) = self.rt.device.synchronize() {
            errors.push(format!("context cleanup sync: {e}"));
        }
        if !errors.is_empty() {
            self.meta["cleanup"] = json!({"status":"FAILED_SYNC_POLICY_NOT_RESET","errors":errors});
            return Err(errors.join("; "));
        }
        let mut cleared = Vec::new();
        for (i, stream) in self.streams.iter().enumerate() {
            if let Err(e) = set(stream, normal()) {
                errors.push(e);
            }
            match get(stream) {
                Ok(w) => {
                    if !same(&w, &normal()) {
                        errors.push(format!("clear readback lane {i}"));
                    }
                    cleared.push(describe(i, stream, &w));
                }
                Err(e) => errors.push(e),
            }
        }
        let reset = checked(
            unsafe { sys::cuCtxResetPersistingL2Cache() },
            "cleanup persisting reset",
        );
        if let Err(e) = &reset {
            errors.push(e.clone());
        }
        if let Err(e) = self.rt.device.set_limit(LIMIT, self.old_limit) {
            errors.push(format!("restore old limit: {e}"));
        }
        let restored_limit = self
            .rt
            .device
            .get_limit(LIMIT)
            .map_err(|e| format!("read restored limit: {e}"));
        match &restored_limit {
            Ok(value) if *value == self.old_limit => {}
            Ok(value) => errors.push(format!("restored limit {value} != {}", self.old_limit)),
            Err(e) => errors.push(e.clone()),
        }
        let mut restored = Vec::new();
        for (i, (stream, old)) in self.streams.iter().zip(&self.old_windows).enumerate() {
            if let Err(e) = set(stream, *old) {
                errors.push(e);
            }
            match get(stream) {
                Ok(w) => {
                    if !same(old, &w) {
                        errors.push(format!("restore readback lane {i}"));
                    }
                    restored.push(describe(i, stream, &w));
                }
                Err(e) => errors.push(e),
            }
        }
        self.meta["cleanup"] = json!({"status":if errors.is_empty(){"PASS"}else{"FAILED"},
            "streams_synchronized":2,"context_synchronized":true,"cleared_windows":cleared,
            "reset":reset.is_ok(),"restored_limit_bytes":restored_limit.ok(),
            "restored_windows":restored,"errors":errors});
        if errors.is_empty() {
            self.finished = true;
            Ok(self.meta.clone())
        } else {
            Err(errors.join("; "))
        }
    }
}

impl Drop for CacheScope<'_> {
    fn drop(&mut self) {
        if !self.finished {
            if let Err(error) = self.finish() {
                // This fallback cannot turn a failed run into PASS. Owners keep
                // their workspaces alive until this attempt has returned.
                eprintln!("L2_DIAGNOSTIC_CLEANUP_FAILED {error}");
            }
        }
    }
}
