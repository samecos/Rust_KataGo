//! Single-GPU context isolation. Raw default-stream handles are both zero, so
//! stream-handle comparison alone cannot authorize a device-buffer operation.
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, result, sys};
use kata_nn::backends::cuda::CudaRuntime;
use kata_nn::backends::mxfp8::{Mxfp8Kernels, Mxfp8Output, Mxfp8Weight, Mxfp8Workspace};
use std::sync::Arc;

#[test]
fn equal_default_stream_handles_do_not_authorize_another_context() {
    if std::env::var("KATAGO_TEST_MXFP8").as_deref() != Ok("1") {
        eprintln!("SKIP MXFP8 context test: set KATAGO_TEST_MXFP8=1");
        return;
    }
    let rt = Arc::new(CudaRuntime::new().expect("explicit MXFP8 test requires CUDA"));
    let owner = rt.device.default_stream();
    let kernels = Mxfp8Kernels::load(&rt).unwrap();
    let weight =
        Mxfp8Weight::upload_from_f32(&rt, &owner, &kernels, &vec![1.0f32; 8 * 32], 8, 32, 32)
            .unwrap();
    let mut ws = Mxfp8Workspace::new(rt.clone(), owner.clone(), kernels, 3, 32).unwrap();
    let half_id = ws
        .prepare_projection(weight.clone(), Mxfp8Output::Half)
        .unwrap();
    let float_id = ws.prepare_projection(weight, Mxfp8Output::Float).unwrap();
    let input_host = vec![0x3c00u16; 3 * 32];
    let input = owner.clone_htod(&input_host).unwrap();
    let mut half = owner.alloc_zeros::<u16>(3 * 8).unwrap();
    let mut float = owner.alloc_zeros::<f32>(3 * 8).unwrap();
    owner.synchronize().unwrap();

    // cudarc 0.19.9's safe new_non_primary/create_v4 cfg lists stop at CUDA
    // 13.1, although its generated sys::cuCtxCreate_v4 supports 13.2/13.3.
    // Use that available driver binding and the unconditional owning wrapper;
    // no dependency feature change, fabricated context or second GPU required.
    let foreign_device = result::device::get(0).unwrap();
    let mut raw_context = std::ptr::null_mut();
    unsafe {
        sys::cuCtxCreate_v4(&mut raw_context, std::ptr::null_mut(), 0, foreign_device)
            .result()
            .expect("create independent CUDA context");
    }
    // Ownership transfers to CudaContext; its Drop destroys the non-primary
    // context after all stream/buffer/pinned-allocation Arc owners have dropped.
    let foreign = unsafe { CudaContext::from_raw_context(0, foreign_device, raw_context) }
        .expect("wrap independently created CUDA context");
    assert!(!foreign.is_primary());
    let other = foreign.default_stream();
    assert_eq!(
        owner.cu_stream(),
        other.cu_stream(),
        "both raw default handles must be zero"
    );
    assert_ne!(owner.context().cu_ctx(), other.context().cu_ctx());
    let foreign_input = other.alloc_zeros::<u16>(3 * 32).unwrap();
    let mut foreign_half = other.alloc_zeros::<u16>(3 * 8).unwrap();
    let mut foreign_float = other.alloc_zeros::<f32>(3 * 8).unwrap();
    let mut foreign_status = unsafe { foreign.alloc_pinned::<u32>(2) }.unwrap();
    other.synchronize().unwrap();
    ws.begin_forward().unwrap();
    let context_error = |result: Result<(), String>| {
        let error = result.expect_err("foreign context must be rejected before launch");
        assert!(error.contains("context"), "{error}");
    };
    context_error(ws.validate_owner_stream(&other));
    context_error(ws.project_half(half_id, 0, &foreign_input, 32, &mut half));
    context_error(ws.project_half(half_id, 0, &input, 32, &mut foreign_half));
    context_error(ws.project_f32(float_id, 1, &foreign_input, 32, &mut float));
    context_error(ws.project_f32(float_id, 1, &input, 32, &mut foreign_float));
    context_error(ws.check_final_outputs([&float, &float, &float, &float, &foreign_float]));
    context_error(ws.enqueue_status_copy(&mut foreign_status));

    // Rejected API calls leave the valid owner's forward intact and do not
    // trigger asynchronous CUDA errors. Exercise both output types as control.
    ws.project_half(half_id, 0, &input, 32, &mut half).unwrap();
    ws.project_f32(float_id, 1, &input, 32, &mut float).unwrap();
    ws.check_final_outputs([&float, &float, &float, &float, &float])
        .unwrap();
    ws.complete_warmup().unwrap();
    let half_values = owner.clone_dtoh(&half).unwrap();
    let float_values = owner.clone_dtoh(&float).unwrap();
    owner.synchronize().unwrap();
    assert_eq!(half_values, vec![0x5000u16; 3 * 8]); // FP16 32.0
    assert_eq!(float_values, vec![32.0f32; 3 * 8]);
    eprintln!(
        "PASS_MXFP8_CONTEXT_ISOLATION: input/half-output/float-output/final-output/pinned-status and valid-owner control"
    );
}
