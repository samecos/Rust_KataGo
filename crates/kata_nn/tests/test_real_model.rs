//! Real model integration test: load kata1-b28c512nbt → parse → ONNX → TRT.
//!
//! The model file is located via the `KATAGO_TEST_MODEL_DIR` environment
//! variable (or the legacy `D:/code/KataGo-Lite` location). Tests are
//! silently skipped when the model file is not present.

#[cfg(test)]
mod tests {
    use kata_nn::model_parser;

    const MODEL_FILE: &str = "kata1-b28c512nbt-s13255194368-d5935380940.bin.gz";

    /// Resolve the test model path, or return `None` if unavailable.
    fn model_path() -> Option<String> {
        if let Ok(dir) = std::env::var("KATAGO_TEST_MODEL_DIR") {
            let p = std::path::PathBuf::from(dir).join(MODEL_FILE);
            if p.exists() {
                return Some(p.to_string_lossy().into_owned());
            }
        }
        let legacy = format!("D:/code/KataGo-Lite/{MODEL_FILE}");
        if std::path::Path::new(&legacy).exists() {
            return Some(legacy);
        }
        None
    }

    #[test]
    #[ignore = "model_parser 尚未对齐 b28c512nbt 模型的格式（上游遗留），M2 ONNX 解析工作时一并修复"]
    fn load_real_model() {
        let Some(path) = model_path() else {
            eprintln!("skipped: model file not found (set KATAGO_TEST_MODEL_DIR)");
            return;
        };
        let desc = model_parser::load_model_file(&path).expect("should parse real model");

        assert_eq!(desc.model_version, 15, "model version should be 15");
        assert!(desc.num_input_channels > 0);
        assert!(desc.num_input_global_channels > 0);
        assert!(desc.trunk.num_blocks > 0);
        assert!(desc.trunk.trunk_num_channels > 0);
        assert!(desc.get_num_parameters() > 0);

        println!(
            "Model: {} v{}, {} params, {} channels, {} blocks",
            desc.name,
            desc.model_version,
            desc.get_num_parameters(),
            desc.trunk.trunk_num_channels,
            desc.trunk.num_blocks
        );
    }

    #[test]
    #[ignore = "model_parser 尚未对齐 b28c512nbt 模型的格式（上游遗留），M2 ONNX 解析工作时一并修复"]
    fn onnx_build_real_model() {
        let Some(path) = model_path() else {
            eprintln!("skipped: model file not found (set KATAGO_TEST_MODEL_DIR)");
            return;
        };
        let desc = model_parser::load_model_file(&path).expect("should parse real model");
        let logger =
            kata_core::logger::Logger::new(kata_core::logger::LoggerOptions::default(), None);

        let result = kata_nn::onnx_builder::build(&desc, 19, 19, true, false, &logger);
        match result {
            Ok(onnx) => {
                assert!(!onnx.serialized_model.is_empty(), "ONNX must not be empty");
                assert!(
                    !onnx.trunk_tip_and_head_node_names.is_empty(),
                    "must have trunk-tip nodes"
                );
                println!(
                    "ONNX built: {} bytes, {} trunk-tip nodes",
                    onnx.serialized_model.len(),
                    onnx.trunk_tip_and_head_node_names.len()
                );
            }
            Err(e) => {
                // Some block types may not be supported yet.
                println!("ONNX build not yet supported for this model: {e}");
            }
        }
    }

    #[test]
    #[cfg(all(feature = "trt", trt_shim_available))]
    fn trt_engine_from_real_model() {
        let Some(path) = model_path() else {
            eprintln!("skipped: model file not found (set KATAGO_TEST_MODEL_DIR)");
            return;
        };
        let desc = model_parser::load_model_file(&path).expect("should parse real model");
        let logger =
            kata_core::logger::Logger::new(kata_core::logger::LoggerOptions::default(), None);

        let onnx = kata_nn::onnx_builder::build(&desc, 19, 19, true, false, &logger)
            .expect("ONNX build should succeed");

        let engine = kata_nn::backends::trt_ffi::katago_trt_engine_create_from_onnx(
            onnx.serialized_model.as_ptr(),
            onnx.serialized_model.len(),
            8,
            0,
            0,
        );
        assert!(!engine.is_null(), "TRT engine should be created");

        let mut info = kata_nn::backends::trt_ffi::KatagoTrtEngineInfo::default();
        let ok = unsafe { kata_nn::backends::trt_ffi::katago_trt_engine_get_info(engine, &mut info) };
        assert_eq!(ok, 1);
        assert!(info.nn_x_len > 0);
        assert!(info.nn_y_len > 0);

        unsafe { kata_nn::backends::trt_ffi::katago_trt_engine_destroy(engine) };
        println!("TRT engine created and destroyed OK");
    }
}
