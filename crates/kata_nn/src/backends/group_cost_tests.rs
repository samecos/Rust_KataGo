//! CPU tests for complete boundaries, route coverage and recorder lifecycle.
//! No CUDA context, stream, event, model file or GPU is constructed here.
use super::*;

#[test]
fn workspace_creation_model_cannot_be_reassigned_by_diagnostic_prepare() {
    use crate::backends::cuda_exec::validate_workspace_model;
    validate_workspace_model(10, 10).unwrap();
    assert!(validate_workspace_model(10, 11).is_err());
    assert!(validate_workspace_model(11, 10).is_err());
    assert!(validate_workspace_model(0, 0).is_err());
}

#[test]
fn malformed_physical_input_lengths_are_rejected_before_launch() {
    use crate::backends::cuda_exec::validate_diagnostic_input_lengths as check;
    check(3, 3 * 22 * 361, 3 * 19).unwrap();
    for (b,s,g) in [(0,0,0),(1,0,19),(1,22*361,18),(1,22*361-1,19),
        (1,22*361+1,19),(usize::MAX,0,0)] {
        assert!(check(b,s,g).is_err());
    }
}

fn group(gate: bool) -> GroupDescriptor {
    GroupDescriptor { id: "trunk.block00.pair00.ffn".into(), rms_layer: 2, ffn_layer: 3,
        end_layer: if gate { 5 } else { 4 }, mid: 384, hidden: 224,
        dual: ProjectionLayout { precision: "int8", n: 448, k: 384, kp: 384, input_stride: 384, output_stride: 448 },
        down: ProjectionLayout { precision: "int8", n: 384, k: 224, kp: 224, input_stride: 224, output_stride: 384 } }
}
fn walk(gate: bool, transitions: &[(usize, usize)]) -> Result<(usize, usize), String> {
    let groups = vec![group(gate)];
    let mut boundary = Boundaries::default();
    let (mut starts, mut ends) = (0, 0);
    for &(li, next) in transitions {
        starts += usize::from(boundary.enter(&groups, li, false)?.is_some());
        ends += usize::from(boundary.leave(&groups, li, next)?.is_some());
    }
    boundary.complete(&groups)?;
    Ok((starts, ends))
}

#[test]
fn normal_rms_ffn_gate_is_one_complete_interval() {
    assert_eq!(walk(true, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]).unwrap(), (1, 1));
}
#[test]
fn either_real_fusion_consumes_the_same_complete_group() {
    assert_eq!(walk(true, &[(0, 1), (1, 2), (2, 4), (4, 5)]).unwrap(), (1, 1));
    assert_eq!(walk(true, &[(0, 1), (1, 2), (2, 3), (3, 5)]).unwrap(), (1, 1));
}
#[test]
fn no_gate_boundary_does_not_absorb_following_layer() {
    assert_eq!(walk(false, &[(0, 1), (1, 2), (2, 4), (4, 5)]).unwrap(), (1, 1));
    assert!(walk(false, &[(0, 1), (1, 2), (2, 3), (3, 5)]).is_err());
}
#[test]
fn pending_splitk_makes_group_non_independent() {
    let groups = vec![group(false)];
    let mut b = Boundaries::default();
    b.enter(&groups, 0, false).unwrap(); b.leave(&groups, 0, 2).unwrap();
    assert!(b.enter(&groups, 2, true).is_err());
}
#[test]
fn skipped_entry_missing_end_and_noncontiguous_layers_fail() {
    assert!(walk(true, &[(0, 3)]).is_err());
    assert!(walk(true, &[(0, 1), (1, 2), (2, 3), (3, 4)]).is_err());
    assert!(walk(true, &[(0, 1), (2, 3)]).is_err());
    assert!(walk(true, &[(0, 1), (1, 2), (2, 6)]).is_err());
}
#[test]
fn duplicate_entry_and_end_without_entry_fail() {
    let groups = vec![group(false)];
    let mut b = Boundaries::default();
    assert!(b.leave(&groups, 0, 1).is_err());
    b.enter(&groups, 0, false).unwrap();
    assert!(b.enter(&groups, 0, false).is_err());
}
#[test]
fn duplicate_overlapping_and_partial_projection_inventories_fail() {
    let valid = group(true);
    validate_groups(&[valid.clone()]).unwrap();
    assert!(validate_groups(&[]).is_err());
    assert!(validate_groups(&[valid.clone(), valid.clone()]).is_err());
    let mut wrong = valid.clone(); wrong.id.push_str(".other");
    assert!(validate_groups(&[valid.clone(), wrong]).is_err());
    let mut wrong = valid.clone(); wrong.dual.n = wrong.hidden; // dual is 2H, not H.
    assert!(validate_groups(&[wrong]).is_err());
    let mut wrong = valid.clone(); wrong.down.input_stride = wrong.down.k - 1;
    assert!(validate_groups(&[wrong]).is_err());
    let mut wrong = valid; wrong.down.precision = "mxfp8";
    assert!(validate_groups(&[wrong]).is_err());
}
fn trace_route(gate: bool, fuse: bool) -> Route {
    start_trace().unwrap();
    role(Role::Rms); launch("rms", "f32-half", [384; 5], None);
    role(Role::Dual); launch("dual", "tn", [448, 384, 384, 384, 448], if fuse { Some(Role::Swiglu) } else { None });
    if !fuse { role(Role::Swiglu); launch("swiglu", "half", [224, 448, 224, 448, 224], None); }
    role(Role::Down); launch("down", "tn", [384, 224, 224, 224, 384], if gate && fuse { Some(Role::Gate) } else { None });
    if gate && !fuse { role(Role::Gate); launch("gate", "f32", [384; 5], None); }
    end_trace().unwrap()
}
#[test]
fn fused_roles_are_included_without_fake_subsegment_times() {
    let _scope = TraceScope::enter().unwrap();
    let fused = trace_route(true, true);
    let unfused = trace_route(true, false);
    fused.validate(true).unwrap(); unfused.validate(true).unwrap();
    assert_eq!(fused.count, 3); assert_eq!(unfused.count, 5);
    assert_ne!(fused, unfused); // Warmup/measurement branch changes cannot alias.
    assert!(fused.validate(false).is_err());
}
#[test]
fn missing_roles_and_trace_overflow_are_rejected() {
    let _scope = TraceScope::enter().unwrap();
    start_trace().unwrap(); role(Role::Rms);
    for _ in 0..=MAX_LAUNCH_OBSERVATIONS { launch("rms", "f32-half", [384; 5], None); }
    let route = end_trace().unwrap();
    assert!(route.overflow); assert!(route.validate(false).is_err());
    assert!(Route::default().validate(false).is_err());
}
#[test]
fn scope_cleanup_nesting_and_disabled_trace_do_not_leak_state() {
    role(Role::Down); setup_observed(); launch("disabled", "none", [0; 5], None);
    assert!(TRACE.with(|cell| cell.borrow().is_none()));
    {
        let _scope = TraceScope::enter().unwrap();
        assert!(TraceScope::enter().is_err());
        start_trace().unwrap(); // Simulate early forward error, without end_trace.
    }
    assert!(TRACE.with(|cell| cell.borrow().is_none()));
    let scope = TraceScope::enter().unwrap();
    assert!(!scope.had_setup().unwrap());
    setup_observed(); assert!(scope.had_setup().unwrap());
}
#[test]
fn measure_requires_warmup_and_rejects_runtime_setup_or_changed_route() {
    let mut cycle = Cycle::default();
    assert!(cycle.start(Phase::Measure, false).is_err());
    cycle.start(Phase::Measure, true).unwrap();
    assert!(cycle.finish(true, true).is_err());
    assert!(cycle.finish(false, false).is_err());
    cycle.finish(false, true).unwrap();
}
#[test]
fn undrained_and_inflight_cycles_cannot_reuse_events() {
    let mut cycle = Cycle::default();
    assert!(cycle.drain().is_err());
    cycle.start(Phase::Warmup, false).unwrap();
    assert!(cycle.start(Phase::Warmup, false).is_err());
    assert!(cycle.drain().is_err());
    cycle.finish(true, false).unwrap(); // Warmup may truthfully report setup.
    assert!(cycle.start(Phase::Warmup, false).is_err());
    cycle.drain().unwrap();
    cycle.start(Phase::Warmup, false).unwrap();
}
#[test]
fn failure_poison_is_terminal() {
    let mut cycle = Cycle::default();
    cycle.start(Phase::Warmup, false).unwrap(); cycle.poison();
    assert!(cycle.finish(false, true).is_err());
    assert!(cycle.drain().is_err());
    assert!(cycle.start(Phase::Warmup, true).is_err());
}
#[test]
fn stream_context_and_physical_batch_must_match() {
    check_owner(11, 11, true, 3, 3, Some(7), 7).unwrap();
    assert!(check_owner(12, 11, true, 3, 3, Some(7), 7).is_err());
    assert!(check_owner(11, 11, false, 3, 3, Some(7), 7).is_err());
    assert!(check_owner(11, 11, true, 1, 3, Some(7), 7).is_err());
}

#[test]
fn workspace_slots_do_not_alias_on_same_stream_and_batch() {
    let first = next_slot_id().unwrap();
    let second = next_slot_id().unwrap();
    assert_ne!(first, second);
    assert!(first > 0 && second > 0);
    check_owner(11, 11, true, 3, 3, Some(first), first).unwrap();
    assert!(check_owner(11, 11, true, 3, 3, Some(second), first).is_err());
    assert!(check_owner(11, 11, true, 3, 3, None, first).is_err());
    assert!(check_owner(11, 11, true, 3, 3, Some(0), 0).is_err());
}
