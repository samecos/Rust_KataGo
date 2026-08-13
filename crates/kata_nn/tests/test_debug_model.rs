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

        // Print first 256 bytes as text
        let text: String = data[..512.min(data.len())]
            .iter()
            .map(|&b| if b.is_ascii() && b != 0 { b as char } else { '.' })
            .collect();
        println!("First 512 bytes as text:\n{}", text);

        // Also print a larger section
        let text2: String = data[..2048.min(data.len())]
            .iter()
            .map(|&b| if b.is_ascii() && b != 0 { b as char } else { '.' })
            .collect();
        println!("\nFirst 2048 bytes as text:\n{}", text2);

        // Print around where the parser fails (around the blocks area)
        let text3: String = data[400..600.min(data.len())]
            .iter()
            .map(|&b| if b.is_ascii() && b != 0 { b as char } else { '.' })
            .collect();
        println!("\nBytes 400-600 as text:\n{}", text3);
    }
}
