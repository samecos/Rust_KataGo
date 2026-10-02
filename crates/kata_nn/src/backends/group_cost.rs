//! Explicit FP16/INT8 whole-FFN direct diagnostics, never Graph performance.
//! No environment switch enables this sink. Existing apply() passes None.
//! Only an explicit recorder enables timing and route collection.

use super::cuda::{self, CudaRuntime};
use crate::quantization_plan::ResolvedRecipe;
use cudarc::driver::{result, sys, CudaEvent, CudaStream};
use serde::Serialize;
use std::{cell::RefCell, collections::BTreeMap, marker::PhantomData, rc::Rc, sync::Arc};

// Assigned only when an explicit diagnostic is prepared. Ordinary workspaces
// keep None: no atomic increment, allocation, event, or string construction.
pub(crate) fn next_slot_id() -> Result<u64, String> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_update(std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| "group-cost workspace identity exhausted".to_owned())
}

pub(crate) fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(message.to_owned()) }
}
fn sha(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase { Warmup, Measure }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) enum Role { Rms = 0, Dual = 1, Swiglu = 2, Down = 3, Gate = 4 }
impl Role { fn bit(self) -> u8 { 1 << self as u8 } }

/// Actual host dispatch observation. This is NOT a subsegment timing or an
/// exported LT algorithm identity. Fused roles share the one group duration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LaunchObservation {
    roles: u8,
    kind: &'static str,
    layout: &'static str,
    /// GEMM/quantization: [N, logical K, padded K, source stride, destination stride].
    /// SwiGLU: [H, 2H, padded H, source stride, destination stride].
    /// RMS/Gate: [channels; 5]. These are activation layouts, not LT descriptor
    /// rows/columns; physical activation rows are stored on ForwardSample.
    dimensions: [usize; 5],
}
const MAX_LAUNCH_OBSERVATIONS: usize = 16;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Route {
    launches: [Option<LaunchObservation>; MAX_LAUNCH_OBSERVATIONS],
    count: usize,
    overflow: bool,
}
impl Default for Route {
    fn default() -> Self { Self { launches: [None; MAX_LAUNCH_OBSERVATIONS], count: 0, overflow: false } }
}
impl Route {
    fn covers(&self) -> u8 {
        self.launches.iter().flatten().fold(0, |bits, item| bits | item.roles)
    }
    fn validate(&self, gate: bool) -> Result<(), String> {
        let expected = Role::Rms.bit() | Role::Dual.bit() | Role::Swiglu.bit() | Role::Down.bit()
            | if gate { Role::Gate.bit() } else { 0 };
        require(!self.overflow && self.count > 0 && self.covers() == expected,
                "group route is missing, overfull or crosses the declared boundary")
    }
}

struct Trace {
    active: Option<Route>,
    role: Role,
    setup_observed: bool,
}
thread_local! {
    // Constant, host-only storage. Disabled calls do not allocate or format.
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}
pub(crate) fn role(role: Role) {
    TRACE.with(|cell| { if let Some(trace) = cell.borrow_mut().as_mut() { trace.role = role; } });
}
pub(crate) fn enabled() -> bool {
    TRACE.with(|cell| cell.borrow().is_some())
}
pub(crate) fn setup_observed() {
    TRACE.with(|cell| { if let Some(trace) = cell.borrow_mut().as_mut() { trace.setup_observed = true; } });
}
pub(crate) fn launch(kind: &'static str, layout: &'static str, dimensions: [usize; 5], also: Option<Role>) {
    TRACE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if let Some(trace) = borrow.as_mut() {
            if let Some(route) = trace.active.as_mut() {
                if route.count == MAX_LAUNCH_OBSERVATIONS { route.overflow = true; return; }
                route.launches[route.count] = Some(LaunchObservation {
                    roles: trace.role.bit() | also.map_or(0, Role::bit), kind, layout, dimensions,
                });
                route.count += 1;
            }
        }
    });
}
fn start_trace() -> Result<(), String> {
    TRACE.with(|cell| {
        let mut trace = cell.borrow_mut();
        let trace = trace.as_mut().ok_or("group trace scope absent")?;
        require(trace.active.is_none(), "nested group route")?;
        trace.active = Some(Route::default());
        trace.role = Role::Rms;
        Ok(())
    })
}
fn end_trace() -> Result<Route, String> {
    TRACE.with(|cell| cell.borrow_mut().as_mut().ok_or("group trace scope absent")?
        .active.take().ok_or_else(|| "group route was not started".to_owned()))
}
pub(crate) struct TraceScope { _thread_local: PhantomData<Rc<()>> }
impl TraceScope {
    pub(crate) fn enter() -> Result<Self, String> {
        TRACE.with(|cell| {
            let mut trace = cell.borrow_mut();
            require(trace.is_none(), "nested group-cost diagnostic")?;
            *trace = Some(Trace { active: None, role: Role::Rms, setup_observed: false });
            Ok(Self { _thread_local: PhantomData })
        })
    }
    fn had_setup(&self) -> Result<bool, String> {
        TRACE.with(|cell| {
            let trace = cell.borrow();
            let trace = trace.as_ref().ok_or("group trace scope absent")?;
            require(trace.active.is_none(), "group remained active at forward completion")?;
            Ok(trace.setup_observed)
        })
    }
}
impl Drop for TraceScope { fn drop(&mut self) { TRACE.with(|cell| *cell.borrow_mut() = None); } }

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProjectionLayout {
    pub(crate) precision: &'static str,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) kp: usize,
    /// Standalone projection interface. A fused path need not materialize this
    /// half tensor; Route records the actual packed intermediate strides.
    pub(crate) input_stride: usize,
    pub(crate) output_stride: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GroupDescriptor {
    pub(crate) id: String,
    pub(crate) rms_layer: usize,
    pub(crate) ffn_layer: usize,
    /// Exclusive layer boundary; includes an adjacent mid-channel GateSiLU.
    pub(crate) end_layer: usize,
    pub(crate) mid: usize,
    pub(crate) hidden: usize,
    pub(crate) dual: ProjectionLayout,
    pub(crate) down: ProjectionLayout,
}
impl GroupDescriptor {
    fn gate(&self) -> bool { self.end_layer == self.ffn_layer + 2 }
}
pub(crate) fn validate_groups(groups: &[GroupDescriptor]) -> Result<(), String> {
    require(!groups.is_empty(), "no FFN groups")?;
    let mut previous_end = 0;
    let mut ids = std::collections::BTreeSet::new();
    for group in groups {
        require(!group.id.is_empty() && ids.insert(&group.id), "empty/duplicate FFN group ID")?;
        require(group.rms_layer >= previous_end && group.rms_layer.checked_add(1) == Some(group.ffn_layer)
            && (group.ffn_layer.checked_add(1) == Some(group.end_layer)
                || group.ffn_layer.checked_add(2) == Some(group.end_layer)), "invalid or overlapping group boundaries")?;
        require(group.mid > 0 && group.hidden > 0 && group.hidden.checked_mul(2) == Some(group.dual.n)
            && group.dual.k == group.mid && group.down.k == group.hidden && group.down.n == group.mid,
            "group projection dimensions differ from complete FFN")?;
        for projection in [&group.dual, &group.down] {
            require(matches!(projection.precision, "fp16" | "int8") && projection.kp >= projection.k
                && projection.input_stride >= projection.k && projection.output_stride >= projection.n,
                "unsupported precision or invalid group stride/padding")?;
        }
        previous_end = group.end_layer;
    }
    Ok(())
}

#[derive(Default)]
struct Boundaries { next_group: usize, active: Option<usize>, next_layer: usize, entered: bool }
impl Boundaries {
    fn enter(&mut self, groups: &[GroupDescriptor], li: usize, pending_splitk: bool) -> Result<Option<usize>, String> {
        require(!self.entered && li == self.next_layer, "duplicate/noncontiguous layer entry")?;
        self.entered = true;
        if self.active.is_none() && self.next_group < groups.len() {
            let group = &groups[self.next_group];
            require(li <= group.rms_layer, "missed group entry")?;
            if li == group.rms_layer {
                require(!pending_splitk, "FFN input depends on cross-boundary pending splitK")?;
                self.active = Some(self.next_group);
                return Ok(self.active);
            }
        }
        Ok(None)
    }
    fn leave(&mut self, groups: &[GroupDescriptor], li: usize, next: usize) -> Result<Option<usize>, String> {
        require(self.entered && li == self.next_layer && next > li, "invalid consumed layer range")?;
        self.entered = false;
        self.next_layer = next;
        if let Some(index) = self.active {
            require(next <= groups[index].end_layer, "fusion consumed past group end")?;
            if next == groups[index].end_layer {
                self.active = None;
                self.next_group += 1;
                return Ok(Some(index));
            }
        } else if self.next_group < groups.len() {
            require(next <= groups[self.next_group].rms_layer, "fusion skipped an unobserved group start")?;
        }
        Ok(None)
    }
    fn complete(&self, groups: &[GroupDescriptor]) -> Result<(), String> {
        require(!self.entered && self.active.is_none() && self.next_group == groups.len(), "incomplete FFN group coverage")
    }
}

fn tactics() -> Result<BTreeMap<String, Option<String>>, String> {
    let mut result = BTreeMap::new();
    for key in crate::tactic_plan::ALLOWED_TACTIC_KEYS.iter().copied()
        .chain(["KATAGO_CUDA_INT8_GEMM_TUNE", "KATAGO_CUDA_INT8_RMS_FUSION"]) {
        let value = match crate::tactic_plan::tactic_var(key) {
            Ok(value) => Some(value), Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err("non-Unicode diagnostic tactic".to_owned()),
        };
        result.insert(key.to_owned(), value);
    }
    Ok(result)
}
#[derive(Clone, Debug, Serialize)]
pub struct Binding {
    schema: &'static str,
    version: u32,
    mode: &'static str,
    model_sha256: String,
    graph_sha256: String,
    recipe_sha256: String,
    executable_sha256: String,
    backend_build: crate::tactic_plan::BackendBuildFingerprint,
    gpu_name: String,
    gpu_uuid_hex: String,
    compute_capability: String,
    sm_count: u32,
    driver_version: i32,
    tactics: BTreeMap<String, Option<String>>,
}
impl Binding {
    pub(crate) fn capture(rt: &CudaRuntime, recipe: &ResolvedRecipe) -> Result<Self, String> {
        let device = cuda::device_fingerprint(&rt.device)?;
        let mut driver_version = 0;
        let status = unsafe { sys::cuDriverGetVersion(&mut driver_version) };
        require(status == sys::cudaError_enum::CUDA_SUCCESS, "query diagnostic driver version failed")?;
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let uuid = rt.device.uuid().map_err(|e| e.to_string())?;
        let uuid_bytes: Vec<u8> = uuid.bytes.iter().map(|byte| *byte as u8).collect();
        Ok(Self { schema: "rustgo-group-cost", version: 1,
            mode: "DIRECT_DIAGNOSTIC", model_sha256: recipe.model_sha256.clone(),
            graph_sha256: recipe.graph_sha256.clone(), recipe_sha256: recipe.recipe_sha256.clone(),
            executable_sha256: crate::tactic_plan::sha256_file(&executable)?,
            backend_build: cuda::backend_build_fingerprint(), gpu_name: device.gpu_name,
            gpu_uuid_hex: hex::encode(uuid_bytes),
            compute_capability: device.compute_capability, sm_count: device.sm_count,
            driver_version, tactics: tactics()? })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GroupSample {
    group: GroupDescriptor,
    route: Route,
    elapsed_ms: f32,
    /// Preserve the exact f32 returned by CUDA; no .3 text rounding.
    elapsed_ms_bits: u32,
}
#[derive(Clone, Debug, Serialize)]
pub struct ForwardSample {
    binding: Binding,
    physical_batch: usize,
    activation_rows: usize,
    /// Process-local unique workspace ID, assigned only by explicit preparation.
    workspace_slot: u64,
    sequence: u64,
    /// Caller-supplied provenance label, not a GPU tensor/hash attestation.
    input_label_sha256: String,
    phase: Phase,
    setup_observed: bool,
    groups: Vec<GroupSample>,
    scope: &'static str,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum State { Ready, Recording, Undrained, Poisoned }
struct Cycle { state: State, phase: Phase }
impl Default for Cycle { fn default() -> Self { Self { state: State::Ready, phase: Phase::Warmup } } }
impl Cycle {
    fn can_start(&self, phase: Phase, warmed: bool) -> Result<(), String> {
        require(self.state == State::Ready, "group-cost forward is active, undrained or poisoned")?;
        require(phase == Phase::Warmup || warmed, "measure requires a drained setup-free warmup")
    }
    fn start(&mut self, phase: Phase, warmed: bool) -> Result<(), String> {
        self.can_start(phase, warmed)?;
        self.phase = phase; self.state = State::Recording;
        Ok(())
    }
    fn finish(&mut self, setup: bool, same_route: bool) -> Result<(), String> {
        require(self.state == State::Recording, "group forward is not recording")?;
        if self.phase == Phase::Measure {
            require(!setup, "setup/cache creation during measure; no steady sample emitted")?;
            require(same_route, "measured route differs from observed warmup")?;
        }
        self.state = State::Undrained;
        Ok(())
    }
    fn can_drain(&self) -> Result<(), String> {
        require(self.state == State::Undrained, "no completed group-cost sample to drain")
    }
    fn drain(&mut self) -> Result<(), String> { self.can_drain()?; self.state = State::Ready; Ok(()) }
    fn poison(&mut self) { self.state = State::Poisoned; }
}
fn check_owner(actual_stream: usize, owner_stream: usize, same_context: bool,
               actual_batch: usize, owner_batch: usize, actual_slot: Option<u64>,
               owner_slot: u64) -> Result<(), String> {
    require(actual_stream == owner_stream && same_context, "group-cost recorder belongs to another stream/context")?;
    require(actual_batch == owner_batch, "group-cost physical batch differs from prepared batch")?;
    require(owner_slot != 0 && actual_slot == Some(owner_slot), "group-cost recorder belongs to another workspace slot")
}

/// Per physical batch / stream session. Events are never shared between slots.
/// The only public constructor is CudaModel::prepare_group_cost, which checks
/// graph/recipe against the actually uploaded model and builds the inventory.
pub struct GroupCostRecorder {
    binding: Binding,
    owner: Arc<CudaStream>,
    batch: usize,
    rows: usize,
    workspace_slot: u64,
    groups: Vec<GroupDescriptor>,
    events: Vec<(CudaEvent, CudaEvent)>,
    terminal: CudaEvent,
    routes: Vec<Route>,
    elapsed: Vec<f32>,
    warmed_routes: Option<Vec<Route>>,
    boundaries: Boundaries,
    cycle: Cycle,
    setup: bool,
    sequence: u64,
    input_label: String,
}
impl GroupCostRecorder {
    pub(crate) fn new(binding: Binding, owner: &Arc<CudaStream>, batch: usize, rows: usize,
                      workspace_slot: u64, groups: Vec<GroupDescriptor>) -> Result<Self, String> {
        validate_groups(&groups)?;
        require(workspace_slot != 0, "group-cost workspace slot is unassigned")?;
        require(batch > 0 && rows == batch.checked_mul(361).ok_or("group row overflow")?, "invalid physical batch/rows")?;
        require(!super::cuda_exec::capturing() && owner.capture_status().map_err(|e| e.to_string())?
            == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE, "group cost cannot prepare in capture")?;
        require(std::env::var_os("KATAGO_CUDA_PROFILE").is_none()
            && std::env::var_os("KATAGO_CUDA_DEBUG_LAYER").is_none(), "group cost rejects PROFILE/DEBUG interference")?;
        let timing = Some(sys::CUevent_flags::CU_EVENT_DEFAULT);
        let mut events = Vec::with_capacity(groups.len());
        for _ in &groups {
            events.push((owner.context().new_event(timing).map_err(|e| e.to_string())?,
                         owner.context().new_event(timing).map_err(|e| e.to_string())?));
        }
        let terminal = owner.context().new_event(timing).map_err(|e| e.to_string())?;
        let count = groups.len();
        Ok(Self { binding, owner: owner.clone(), batch, rows, workspace_slot, groups, events, terminal,
            routes: vec![Route::default(); count], elapsed: vec![0.0; count], warmed_routes: None,
            boundaries: Boundaries::default(), cycle: Cycle::default(),
            setup: false, sequence: 0, input_label: String::new() })
    }
    pub(crate) fn begin(&mut self, stream: &Arc<CudaStream>, batch: usize, workspace_slot: Option<u64>, recipe: Option<&str>,
                        phase: Phase, input_label: &str) -> Result<TraceScope, String> {
        self.cycle.can_start(phase, self.warmed_routes.is_some())?;
        check_owner(stream.cu_stream() as usize, self.owner.cu_stream() as usize,
            Arc::ptr_eq(stream.context(), self.owner.context()), batch, self.batch, workspace_slot, self.workspace_slot)?;
        require(recipe == Some(self.binding.recipe_sha256.as_str()), "group-cost model recipe mismatch")?;
        require(!super::cuda_exec::capturing() && stream.capture_status().map_err(|e| e.to_string())?
            == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE, "DIRECT_DIAGNOSTIC rejects capture")?;
        require(std::env::var_os("KATAGO_CUDA_PROFILE").is_none()
            && std::env::var_os("KATAGO_CUDA_DEBUG_LAYER").is_none(), "group cost rejects PROFILE/DEBUG interference")?;
        require(tactics()? == self.binding.tactics, "diagnostic tactics changed after preparation")?;
        require(sha(input_label), "input label must be a SHA256 provenance label")?;
        let scope = TraceScope::enter()?;
        self.sequence = self.sequence.checked_add(1).ok_or("group sample sequence exhausted")?;
        self.input_label.clear(); self.input_label.push_str(input_label);
        self.routes.fill(Route::default()); self.elapsed.fill(0.0);
        self.boundaries = Boundaries::default(); self.setup = false;
        self.cycle.start(phase, self.warmed_routes.is_some())?;
        Ok(scope)
    }
    pub(crate) fn before_layer(&mut self, li: usize, pending_splitk: bool) -> Result<(), String> {
        if let Some(index) = self.boundaries.enter(&self.groups, li, pending_splitk)? {
            start_trace()?;
            self.events[index].0.record(&self.owner).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub(crate) fn after_layer(&mut self, li: usize, next: usize) -> Result<(), String> {
        if let Some(index) = self.boundaries.leave(&self.groups, li, next)? {
            self.events[index].1.record(&self.owner).map_err(|e| e.to_string())?;
            let route = end_trace()?;
            route.validate(self.groups[index].gate())?;
            self.routes[index] = route;
        }
        Ok(())
    }
    pub(crate) fn finish(&mut self, scope: &TraceScope) -> Result<(), String> {
        self.boundaries.complete(&self.groups)?;
        self.setup = scope.had_setup()?;
        self.terminal.record(&self.owner).map_err(|e| e.to_string())?;
        // Exactly one explicit diagnostic synchronization after the forward.
        self.terminal.synchronize().map_err(|e| e.to_string())?;
        for (index, (start, end)) in self.events.iter().enumerate() {
            // SAFETY: every event was recorded once on this owned stream, all
            // pairs precede the terminal event, and terminal is synchronized.
            // Events/context remain owned here. CudaEvent::elapsed_ms would
            // synchronize twice per pair, so deliberately do NOT call it.
            let ms = unsafe { result::event::elapsed(start.cu_event(), end.cu_event()) }
                .map_err(|e| e.to_string())?;
            require(ms.is_finite() && ms >= 0.0, "invalid CUDA group elapsed time")?;
            self.elapsed[index] = ms;
        }
        self.cycle.finish(self.setup, self.warmed_routes.as_ref() == Some(&self.routes))?;
        Ok(())
    }
    pub(crate) fn poison(&mut self) { self.cycle.poison(); }
    /// True only after a successful setup-free warmup has been drained and no
    /// forward/sample is outstanding. This is not a throughput/Graph claim.
    pub fn ready_to_measure(&self) -> bool {
        self.cycle.state == State::Ready && self.warmed_routes.is_some()
    }
    /// Drain only after apply_with_group_cost succeeds. Copy/serialization is
    /// intentionally outside all event intervals and outside the forward.
    pub fn drain(&mut self) -> Result<ForwardSample, String> {
        self.cycle.can_drain()?;
        let groups = self.groups.iter().cloned().enumerate().map(|(index, group)| GroupSample {
            group, route: self.routes[index], elapsed_ms: self.elapsed[index],
            elapsed_ms_bits: self.elapsed[index].to_bits(),
        }).collect();
        let sample = ForwardSample { binding: self.binding.clone(), physical_batch: self.batch,
            activation_rows: self.rows, workspace_slot: self.workspace_slot, sequence: self.sequence, input_label_sha256: self.input_label.clone(),
            phase: self.cycle.phase, setup_observed: self.setup, groups,
            scope: "whole RMS+FFN+adjacent Gate; fused work included, no subsegment timings; DIRECT_DIAGNOSTIC only" };
        if self.cycle.phase == Phase::Warmup {
            self.warmed_routes = if self.setup { None } else { Some(self.routes.clone()) };
        }
        self.cycle.drain()?;
        Ok(sample)
    }
}

#[cfg(test)]
#[path = "group_cost_tests.rs"]
mod tests;
