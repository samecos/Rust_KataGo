#[cfg(test)]
mod tests {
    use kata_nn::model_parser;

    /// Resolve the debug model path, or return `None` if unavailable.
    fn debug_model_path() -> Option<String> {
        let file = "kata1-zhizi-b40c768nbt-s11272M-d5935M.bin.gz";
        if let Ok(dir) = std::env::var("KATAGO_TEST_MODEL_DIR") {
            let p = std::path::PathBuf::from(dir).join(file);
            if p.exists() {
                return Some(p.to_string_lossy().into_owned());
            }
        }
        let legacy = format!("D:/code/KataGo-Lite/{file}");
        if std::path::Path::new(&legacy).exists() {
            return Some(legacy);
        }
        None
    }

    #[test]
    fn debug_parse_real_model() {
        let Some(path) = debug_model_path() else {
            eprintln!("skipped: model file not found (set KATAGO_TEST_MODEL_DIR)");
            return;
        };
        let mut file = std::fs::File::open(path).unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut buf).unwrap();

        // Decompress
        use flate2::read::GzDecoder;
        let mut decoder = GzDecoder::new(&buf[..]);
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut data).unwrap();

        // Create a simple parser wrapper to see what's being read
        let mut pos = 0;
        let binary = true;

        fn next_token(data: &[u8], pos: &mut usize) -> String {
            while *pos < data.len() && data[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
            let start = *pos;
            while *pos < data.len() && !data[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
            String::from_utf8(data[start..*pos].to_vec()).unwrap_or_default()
        }

        // Read model header
        let name = next_token(&data, &mut pos);
        println!("name: {}", name);
        let model_version: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("model_version: {}", model_version);
        let num_input_channels: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("num_input_channels: {}", num_input_channels);
        let num_input_global_channels: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("num_input_global_channels: {}", num_input_global_channels);

        // postProcessParams (7 floats)
        for i in 0..7 {
            let v: f32 = next_token(&data, &mut pos).parse().unwrap();
            println!("postProcessParams[{}]: {}", i, v);
        }

        // metaEncoderVersion + 7 unused
        let meta_version: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("metaEncoderVersion: {}", meta_version);
        for i in 0..7 {
            let v: i32 = next_token(&data, &mut pos).parse().unwrap();
            println!("unused[{}]: {}", i, v);
        }

        // Trunk
        let trunk_name = next_token(&data, &mut pos);
        println!("trunk_name: {}", trunk_name);
        let num_blocks: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("num_blocks: {}", num_blocks);
        let trunk_num_channels: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("trunk_num_channels: {}", trunk_num_channels);
        let mid_num_channels: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("mid_num_channels: {}", mid_num_channels);
        let regular_num_channels: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("regular_num_channels: {}", regular_num_channels);
        let dilated_num_channels: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("dilated_num_channels: {}", dilated_num_channels);
        let gpool_num_channels: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("gpool_num_channels: {}", gpool_num_channels);

        // trunkNormKind + 5 unused
        let trunk_norm_kind: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("trunk_norm_kind: {}", trunk_norm_kind);
        for i in 0..5 {
            let v: i32 = next_token(&data, &mut pos).parse().unwrap();
            println!("trunk unused[{}]: {}", i, v);
        }

        // First block
        let conv_name = next_token(&data, &mut pos);
        println!("conv_name: {}", conv_name);
        let conv_y_size: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("conv_y_size: {}", conv_y_size);
        let conv_x_size: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("conv_x_size: {}", conv_x_size);
        let in_channels: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("in_channels: {}", in_channels);
        let out_channels: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("out_channels: {}", out_channels);
        let dilation_y: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("dilation_y: {}", dilation_y);
        let dilation_x: i32 = next_token(&data, &mut pos).parse().unwrap();
        println!("dilation_x: {}", dilation_x);
    }
}
