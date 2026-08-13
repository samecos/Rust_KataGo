// Build script for kata_nn.
//
// When the `trt` feature is enabled, this script compiles the C++ TensorRT
// shim (`cpp-shim/src/trt_shim.cpp`) into a static library and links against
// CUDA and TensorRT.
//
// If CUDA or TensorRT are not found, the build succeeds without the shim
// and the trt feature stub (which returns compile-time-disabled errors)
// serves as fallback.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Register the custom cfg so that Rust doesn't warn about it.
    println!("cargo::rustc-check-cfg=cfg(trt_shim_available)");

    // Compile the ONNX protobuf into Rust types. This is always done so that
    // the onnx_builder module has types to work with regardless of whether
    // the TRT backend is enabled.
    compile_onnx_proto();

    let is_trt = env::var("CARGO_FEATURE_TRT").is_ok();
    if is_trt {
        build_trt_shim();
    }

    let is_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    if is_cuda {
        compile_cuda_kernels();
    }
}

/// Compile the TensorRT C++ shim and link CUDA/TensorRT.
fn build_trt_shim() {
    // Locate CUDA toolkit root.
    // Search known locations (Windows first, then Linux).
    let cuda_root = find_cuda_root();
    let (cuda_root, is_windows) = match cuda_root {
        Some((path, win)) => (path, win),
        None => {
            println!("cargo:warning=CUDA Toolkit not found — TensorRT shim will not be compiled");
            return;
        }
    };

    let include_dir = cuda_root.join("include");
    if !include_dir.join("cuda_runtime.h").exists() {
        println!(
            "cargo:warning=CUDA include dir {} not found — TensorRT shim skipped",
            include_dir.display()
        );
        return;
    }
    if !include_dir.join("NvInfer.h").exists() {
        println!(
            "cargo:warning=NvInfer.h not found in {} — TensorRT shim skipped",
            include_dir.display()
        );
        return;
    }

    println!("cargo:rerun-if-changed=../../cpp-shim/src/trt_shim.cpp");
    println!("cargo:rerun-if-changed=../../cpp-shim/src/trt_shim.h");

    // Build the C++ shim.
    let shim_cpp = find_shim_cpp();
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(&shim_cpp)
        .include(&include_dir);

    // MSVC-specific flags.
    if is_windows {
        build.flag("/EHsc"); // enable C++ exceptions
        build.flag("/bigobj");
    }

    build.compile("katago_trt_shim");

    // Signal to Rust code that the shim was compiled successfully.
    println!("cargo:rustc-cfg=trt_shim_available");

    // Link libraries.
    let lib_dir = if is_windows {
        cuda_root.join("lib").join("x64")
    } else {
        cuda_root.join("lib64")
    };
    let trt_lib_dir = cuda_root.join("lib");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", trt_lib_dir.display());

    // Link against CUDA runtime.
    println!("cargo:rustc-link-lib=cudart");

    // TensorRT 10 libraries (may be named with version suffix on Windows).
    if is_windows {
        println!("cargo:rustc-link-lib=nvinfer_10");
        println!("cargo:rustc-link-lib=nvonnxparser_10");
    } else {
        println!("cargo:rustc-link-lib=nvinfer");
        println!("cargo:rustc-link-lib=nvonnxparser");
    }
}

/// Find the CUDA toolkit root directory.
fn find_cuda_root() -> Option<(PathBuf, bool)> {
    // 1. Check CUDA_PATH environment variable.
    if let Ok(path) = env::var("CUDA_PATH") {
        let p = PathBuf::from(&path);
        if p.join("include").join("cuda_runtime.h").exists() {
            return Some((p, cfg!(windows)));
        }
    }

    // 2. Check common Windows locations.
    #[cfg(windows)]
    {
        // List subdirectories and pick the highest version.
        let base = PathBuf::from("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
        if base.exists() {
            if let Ok(entries) = std::fs::read_dir(&base) {
                let mut versions: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect();
                versions.sort();
                // Try highest version first.
                for v in versions.iter().rev() {
                    let inc = v.join("include");
                    if inc.join("cuda_runtime.h").exists() && inc.join("NvInfer.h").exists() {
                        return Some((v.clone(), true));
                    }
                }
            }
        }
    }

    // 3. Check common Linux locations.
    #[cfg(not(windows))]
    {
        for candidate in &["/usr/local/cuda", "/opt/cuda"] {
            let p = PathBuf::from(*candidate);
            if p.join("include").join("cuda_runtime.h").exists() {
                return Some((p, false));
            }
        }
    }

    None
}

/// Locate the trt_shim.cpp source file relative to the kata_nn crate.
fn find_shim_cpp() -> PathBuf {
    // The shim is at katago-rs/cpp-shim/src/trt_shim.cpp.
    // kata_nn is at katago-rs/crates/kata_nn/.
    // So from kata_nn/, it's ../../cpp-shim/src/trt_shim.cpp.
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut path = manifest_dir.clone();
    path.pop(); // kata_nn
    path.pop(); // crates
    path.push("cpp-shim");
    path.push("src");
    path.push("trt_shim.cpp");
    if path.exists() {
        return path;
    }
    // Fallback: try relative to workspace root (when building from root).
    let mut path2 = manifest_dir.clone();
    path2.pop();
    path2.pop();
    path2.push("cpp-shim");
    path2.push("src");
    path2.push("trt_shim.cpp");
    if path2.exists() {
        return path2;
    }
    // Last resort: assume relative to manifest dir.
    PathBuf::from("../../cpp-shim/src/trt_shim.cpp")
}

/// Compile the ONNX protobuf schema into Rust types.
fn compile_onnx_proto() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // Vendored copy of KataGo's ONNX protobuf schema (see proto/onnx.proto).
    // Required unconditionally: onnx_builder includes the generated types.
    let proto_path = manifest_dir.join("proto").join("onnx.proto");

    if !proto_path.exists() {
        panic!(
            "ONNX proto not found at {} — required by onnx_builder",
            proto_path.display()
        );
    }

    println!("cargo:rerun-if-changed={}", proto_path.display());

    let proto_dir = proto_path.parent().unwrap().to_path_buf();

    // Set PROTOC to the vendored binary so that prost-build can find it.
    let protoc_path = protoc_bin_vendored::protoc_bin_path().expect("protoc binary not found");
    unsafe { env::set_var("PROTOC", &protoc_path) };

    prost_build::compile_protos(&[proto_path], &[&proto_dir])
        .expect("Failed to compile ONNX protobuf schema");
}

/// Locate the nvcc compiler (CUDA_PATH/CUDA_HOME, else PATH).
fn find_nvcc() -> Option<PathBuf> {
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = env::var(var) {
            let p = PathBuf::from(&root).join("bin").join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
            if p.exists() {
                return Some(p);
            }
        }
    }
    let plain = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    Command::new(plain).arg("--version").output().ok().map(|_| PathBuf::from(plain))
}

/// Compile the hand-written CUDA kernels (`cuda-kernels/*.cu`) to PTX for
/// every target listed in `configs/sm-targets.json`, and emit a Rust table
/// embedding the PTX bytes so the CUDA backend can load them at runtime.
fn compile_cuda_kernels() {
    #[derive(serde::Deserialize)]
    struct SmTarget {
        id: String,
        compute_capability: String,
        nvcc_ptx_arch: String,
    }
    #[derive(serde::Deserialize)]
    struct SmTargets {
        targets: Vec<SmTarget>,
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("cuda-kernels");
    let targets_json = manifest_dir.join("../../configs/sm-targets.json");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    if !kernels_dir.exists() {
        println!("cargo:warning=cuda-kernels dir not found — CUDA backend will have no kernels");
        return;
    }
    let mut cu_files: Vec<PathBuf> = std::fs::read_dir(&kernels_dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "cu"))
                .collect()
        })
        .unwrap_or_default();
    cu_files.sort();
    if cu_files.is_empty() {
        println!("cargo:warning=cuda-kernels dir has no .cu files — CUDA backend will have no kernels");
        return;
    }

    let targets: Vec<SmTarget> = match std::fs::read_to_string(&targets_json) {
        Ok(text) => match serde_json::from_str::<SmTargets>(&text) {
            Ok(t) => t.targets,
            Err(e) => {
                println!("cargo:warning=sm-targets.json parse failed ({e}); using sm_120 only");
                vec![SmTarget {
                    id: "sm_120".to_string(),
                    compute_capability: "12.0".to_string(),
                    nvcc_ptx_arch: "compute_120".to_string(),
                }]
            }
        },
        Err(_) => {
            println!("cargo:warning=sm-targets.json not found; using sm_120 only");
            vec![SmTarget {
                id: "sm_120".to_string(),
                compute_capability: "12.0".to_string(),
                nvcc_ptx_arch: "compute_120".to_string(),
            }]
        }
    };
    println!("cargo:rerun-if-changed={}", targets_json.display());
    for cu in &cu_files {
        println!("cargo:rerun-if-changed={}", cu.display());
    }

    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found — CUDA kernels not compiled");
        return;
    };

    let mut table = String::from("// @generated by build.rs — CUDA kernel PTX per SM target\n");
    table.push_str("pub const CUDA_TARGETS: &[(&str, &str, &[(&str, &[u8])])] = &[\n");
    for t in &targets {
        table.push_str(&format!(
            "    (\"{}\", \"{}\", &[\n",
            t.id, t.compute_capability
        ));
        for cu in &cu_files {
            let name = cu.file_stem().unwrap().to_string_lossy().into_owned();
            let ptx_path = out_dir.join(format!("{}_{}.ptx", t.id, name));
            let mut cmd = Command::new(&nvcc);
            // nvcc needs the MSVC host compiler on Windows; locate it via cc.
            #[cfg(windows)]
            {
                if let Some(ccbin) = msvc_host_dir() {
                    cmd.arg("-ccbin").arg(ccbin);
                }
            }
            let status = cmd
                .args(["-ptx", "-arch", &t.nvcc_ptx_arch, "-O3", "--std=c++17"])
                .arg(cu)
                .arg("-o")
                .arg(&ptx_path)
                .status();
            match status {
                Ok(s) if s.success() => {
                    table.push_str(&format!(
                        "        (\"{name}\", include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}_{}.ptx\"))),\n",
                        t.id, name
                    ));
                }
                other => {
                    println!(
                        "cargo:warning=nvcc failed for kernel {name} (target {}): {other:?}",
                        t.id
                    );
                }
            }
        }
        table.push_str("    ]),\n");
    }
    table.push_str("];\n");
    if let Err(e) = std::fs::write(out_dir.join("cuda_kernels.rs"), table) {
        println!("cargo:warning=failed to write cuda_kernels.rs: {e}");
    }
}

/// Directory containing cl.exe (MSVC host compiler), probed via the cc crate.
#[cfg(windows)]
fn msvc_host_dir() -> Option<PathBuf> {
    let tool = cc::Build::new().get_compiler();
    tool.path().parent().map(|p| p.to_path_buf())
}
