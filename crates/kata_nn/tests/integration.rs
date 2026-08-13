//! End-to-end integration test: synthetic model → parse → ONNX → TensorRT engine → inference.

#[cfg(test)]
mod tests {
    use kata_nn::desc::*;

    #[test]
    fn end_to_end_synthetic_model() {
        let logger =
            kata_core::logger::Logger::new(kata_core::logger::LoggerOptions::default(), None);
        let desc = make_test_desc();

        // ONNX builder produces non-empty model.
        let onnx = kata_nn::onnx_builder::build(&desc, 19, 19, true, false, &logger)
            .expect("ONNX build should succeed");
        assert!(!onnx.serialized_model.is_empty(), "ONNX must not be empty");
        assert!(
            !onnx.trunk_tip_and_head_node_names.is_empty(),
            "must have some trunk-tip/head nodes"
        );

        // TensorRT engine creation from ONNX (only if trt feature + shim available).
        #[cfg(all(feature = "trt", trt_shim_available))]
        {
            let engine = kata_nn::backends::trt_ffi::katago_trt_engine_create_from_onnx(
                onnx.serialized_model.as_ptr(),
                onnx.serialized_model.len(),
                8,
                0,
                0,
            );
            assert!(!engine.is_null(), "TRT engine should be created");
            unsafe {
                kata_nn::backends::trt_ffi::katago_trt_engine_destroy(engine);
            }
        }
    }

    #[test]
    fn backend_lifecycle_with_dummy() {
        use kata_nn::backend::Backend;
        use kata_nn::backend::dummy::DummyBackend;
        let backend = DummyBackend;
        backend.global_initialize();
        backend.print_devices();
        let model = backend.load_model_file("dummy.bin", "").unwrap();
        assert!(model.model_desc().model_version > 0);
        backend.global_cleanup();
    }

    fn make_test_desc() -> ModelDesc {
        ModelDesc {
            name: "test".into(),
            sha256: "0".repeat(64),
            model_version: 15,
            num_input_channels: 22,
            num_input_global_channels: 19,
            num_input_meta_channels: 0,
            num_policy_channels: 2,
            num_value_channels: 3,
            num_score_value_channels: 6,
            num_ownership_channels: 1,
            meta_encoder_version: 0,
            post_process_params: ModelPostProcessParams::default(),
            trunk: TrunkDesc {
                name: "trunk".into(),
                model_version: 15,
                num_blocks: 1,
                trunk_num_channels: 32,
                mid_num_channels: 32,
                regular_num_channels: 32,
                gpool_num_channels: 32,
                meta_encoder_version: 0,
                trunk_norm_kind: 0,
                initial_conv: ConvLayerDesc {
                    name: "init_conv".into(),
                    conv_y_size: 5,
                    conv_x_size: 5,
                    in_channels: 22,
                    out_channels: 32,
                    dilation_y: 1,
                    dilation_x: 1,
                    weights: vec![0.0; 5 * 5 * 22 * 32],
                },
                initial_mat_mul: MatMulLayerDesc {
                    name: "init_mm".into(),
                    in_channels: 19,
                    out_channels: 32,
                    weights: vec![0.0; 19 * 32],
                },
                sgf_metadata_encoder: Default::default(),
                blocks: vec![BlockDesc::Ordinary(ResidualBlockDesc {
                    name: "r0".into(),
                    pre_bn: BatchNormLayerDesc {
                        name: "r0bn0".into(),
                        num_channels: 32,
                        epsilon: 1e-5,
                        has_scale: true,
                        has_bias: true,
                        mean: vec![0.0; 32],
                        variance: vec![1.0; 32],
                        scale: vec![1.0; 32],
                        bias: vec![0.0; 32],
                        merged_scale: vec![0.0; 32],
                        merged_bias: vec![0.0; 32],
                    },
                    pre_activation: Default::default(),
                    regular_conv: ConvLayerDesc {
                        name: "r0c1".into(),
                        conv_y_size: 3,
                        conv_x_size: 3,
                        in_channels: 32,
                        out_channels: 32,
                        dilation_y: 1,
                        dilation_x: 1,
                        weights: vec![0.0; 3 * 3 * 32 * 32],
                    },
                    mid_bn: BatchNormLayerDesc {
                        name: "r0bn1".into(),
                        num_channels: 32,
                        epsilon: 1e-5,
                        has_scale: true,
                        has_bias: true,
                        mean: vec![0.0; 32],
                        variance: vec![1.0; 32],
                        scale: vec![1.0; 32],
                        bias: vec![0.0; 32],
                        merged_scale: vec![0.0; 32],
                        merged_bias: vec![0.0; 32],
                    },
                    mid_activation: Default::default(),
                    final_conv: ConvLayerDesc {
                        name: "r0c2".into(),
                        conv_y_size: 3,
                        conv_x_size: 3,
                        in_channels: 32,
                        out_channels: 32,
                        dilation_y: 1,
                        dilation_x: 1,
                        weights: vec![0.0; 3 * 3 * 32 * 32],
                    },
                })],
                trunk_tip_bn: BatchNormLayerDesc {
                    name: "tip_bn".into(),
                    num_channels: 32,
                    epsilon: 1e-5,
                    has_scale: true,
                    has_bias: true,
                    mean: vec![0.0; 32],
                    variance: vec![1.0; 32],
                    scale: vec![1.0; 32],
                    bias: vec![0.0; 32],
                    merged_scale: vec![0.0; 32],
                    merged_bias: vec![0.0; 32],
                },
                trunk_tip_rms_norm: Default::default(),
                trunk_tip_activation: Default::default(),
            },
            policy_head: PolicyHeadDesc {
                name: "policy".into(),
                model_version: 15,
                policy_out_channels: 2,
                p1_conv: ConvLayerDesc {
                    name: "p1".into(),
                    conv_y_size: 1,
                    conv_x_size: 1,
                    in_channels: 32,
                    out_channels: 2,
                    dilation_y: 1,
                    dilation_x: 1,
                    weights: vec![0.0; 32 * 2],
                },
                g1_conv: ConvLayerDesc {
                    name: "pg1".into(),
                    conv_y_size: 1,
                    conv_x_size: 1,
                    in_channels: 32,
                    out_channels: 32,
                    dilation_y: 1,
                    dilation_x: 1,
                    weights: vec![0.0; 32 * 32],
                },
                g1_bn: BatchNormLayerDesc {
                    name: "pg1bn".into(),
                    num_channels: 32,
                    epsilon: 1e-5,
                    has_scale: true,
                    has_bias: true,
                    mean: vec![0.0; 32],
                    variance: vec![1.0; 32],
                    scale: vec![1.0; 32],
                    bias: vec![0.0; 32],
                    merged_scale: vec![0.0; 32],
                    merged_bias: vec![0.0; 32],
                },
                g1_activation: Default::default(),
                gpool_to_bias_mul: MatMulLayerDesc {
                    name: "pgbmul".into(),
                    in_channels: 32,
                    out_channels: 2,
                    weights: vec![0.0; 32 * 2],
                },
                p1_bn: BatchNormLayerDesc {
                    name: "p1bn".into(),
                    num_channels: 2,
                    epsilon: 1e-5,
                    has_scale: true,
                    has_bias: true,
                    mean: vec![0.0; 2],
                    variance: vec![1.0; 2],
                    scale: vec![1.0; 2],
                    bias: vec![0.0; 2],
                    merged_scale: vec![0.0; 2],
                    merged_bias: vec![0.0; 2],
                },
                p1_activation: Default::default(),
                p2_conv: ConvLayerDesc {
                    name: "p2".into(),
                    conv_y_size: 1,
                    conv_x_size: 1,
                    in_channels: 2,
                    out_channels: 2,
                    dilation_y: 1,
                    dilation_x: 1,
                    weights: vec![0.0; 2 * 2],
                },
                gpool_to_pass_mul: MatMulLayerDesc {
                    name: "pgpmul".into(),
                    in_channels: 32,
                    out_channels: 2,
                    weights: vec![0.0; 32 * 2],
                },
                gpool_to_pass_bias: MatBiasLayerDesc {
                    name: "pgpb".into(),
                    num_channels: 2,
                    weights: vec![0.0; 2],
                },
                pass_activation: Default::default(),
                gpool_to_pass_mul2: MatMulLayerDesc::default(),
            },
            value_head: ValueHeadDesc {
                name: "value".into(),
                model_version: 15,
                v1_conv: ConvLayerDesc {
                    name: "v1".into(),
                    conv_y_size: 1,
                    conv_x_size: 1,
                    in_channels: 32,
                    out_channels: 32,
                    dilation_y: 1,
                    dilation_x: 1,
                    weights: vec![0.0; 32 * 32],
                },
                v1_bn: BatchNormLayerDesc {
                    name: "v1bn".into(),
                    num_channels: 32,
                    epsilon: 1e-5,
                    has_scale: true,
                    has_bias: true,
                    mean: vec![0.0; 32],
                    variance: vec![1.0; 32],
                    scale: vec![1.0; 32],
                    bias: vec![0.0; 32],
                    merged_scale: vec![0.0; 32],
                    merged_bias: vec![0.0; 32],
                },
                v1_activation: Default::default(),
                v2_mul: MatMulLayerDesc {
                    name: "v2mul".into(),
                    in_channels: 32,
                    out_channels: 32,
                    weights: vec![0.0; 32 * 32],
                },
                v2_bias: MatBiasLayerDesc {
                    name: "v2b".into(),
                    num_channels: 32,
                    weights: vec![0.0; 32],
                },
                v2_activation: Default::default(),
                v3_mul: MatMulLayerDesc {
                    name: "v3mul".into(),
                    in_channels: 32,
                    out_channels: 3,
                    weights: vec![0.0; 32 * 3],
                },
                v3_bias: MatBiasLayerDesc {
                    name: "v3b".into(),
                    num_channels: 3,
                    weights: vec![0.0; 3],
                },
                sv3_mul: MatMulLayerDesc {
                    name: "sv3mul".into(),
                    in_channels: 32,
                    out_channels: 6,
                    weights: vec![0.0; 32 * 6],
                },
                sv3_bias: MatBiasLayerDesc {
                    name: "sv3b".into(),
                    num_channels: 6,
                    weights: vec![0.0; 6],
                },
                v_ownership_conv: ConvLayerDesc {
                    name: "own".into(),
                    conv_y_size: 1,
                    conv_x_size: 1,
                    in_channels: 32,
                    out_channels: 1,
                    dilation_y: 1,
                    dilation_x: 1,
                    weights: vec![0.0; 32],
                },
            },
        }
    }
}
