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

struct CutlassBuildInfo {
    version: String,
    commit: String,
}

fn main() {
    // Register the custom cfg so that Rust doesn't warn about it.
    println!("cargo::rustc-check-cfg=cfg(trt_shim_available)");
    println!("cargo::rustc-check-cfg=cfg(katago_dualffn)");

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
        let (compiled_sms, attention_q64, attention_q64_serial) = compile_cuda_kernels();
        let cutlass = compile_dual_ffn();
        emit_cuda_build_info(
            cutlass.as_ref(),
            &compiled_sms,
            attention_q64,
            attention_q64_serial,
        );
    }
}

/// B2:CUTLASS DualGemm(dual FFN + SwiGLU epilogue)的主机侧编译。
/// 与 PTX 嵌入的 kernel 不同:CUTLASS device API 是主机模板代码,必须
/// nvcc -c 编成对象文件再打进静态库链接。CUTLASS 根查找顺序:
/// 环境变量 KATAGO_CUTLASS_ROOT → 仓库 third_party/cutlass → D:/code/cutlass。
/// 找不到或编译失败:跳过(Rust 侧无 katago_dualffn cfg,tactic 回退现有路径)。
fn compile_dual_ffn() -> Option<CutlassBuildInfo> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest_dir.join("cuda-host").join("dual_ffn_cutlass.cu");
    if !src.exists() {
        return None;
    }
    let cutlass_root = env::var("KATAGO_CUTLASS_ROOT")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let p = manifest_dir.join("../../third_party/cutlass");
            p.exists().then_some(p)
        })
        .or_else(|| {
            let p = PathBuf::from("D:/code/cutlass");
            p.exists().then_some(p)
        });
    let Some(cutlass_root) = cutlass_root else {
        println!("cargo:warning=CUTLASS not found (set KATAGO_CUTLASS_ROOT) — dual_ffn skipped");
        return None;
    };
    if !cutlass_root.join("include/cutlass/cutlass.h").exists()
        || !cutlass_root
            .join("examples/45_dual_gemm/device/dual_gemm.h")
            .exists()
    {
        println!(
            "cargo:warning=CUTLASS root {} lacks headers — dual_ffn skipped",
            cutlass_root.display()
        );
        return None;
    }
    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found — dual_ffn skipped");
        return None;
    };
    let Some((cuda_root, is_windows)) = find_cuda_root() else {
        println!("cargo:warning=CUDA root not found — dual_ffn skipped");
        return None;
    };

    let version_path = cutlass_root.join("include/cutlass/version.h");
    let cutlass_version =
        parse_cutlass_version(&version_path).unwrap_or_else(|| "unknown".to_string());
    let cutlass_commit = git_source_id(&cutlass_root).unwrap_or_else(|| "unknown".to_string());

    // gencode 取自 configs/sm-targets.json(与 PTX 路径同表)。
    let targets_json = manifest_dir.join("../../configs/sm-targets.json");
    #[derive(serde::Deserialize)]
    struct SmTarget {
        compute_capability: String,
    }
    #[derive(serde::Deserialize)]
    struct SmTargets {
        targets: Vec<SmTarget>,
    }
    let caps: Vec<String> = std::fs::read_to_string(&targets_json)
        .ok()
        .and_then(|t| serde_json::from_str::<SmTargets>(&t).ok())
        .map(|t| {
            t.targets
                .into_iter()
                .map(|x| x.compute_capability)
                .collect()
        })
        .unwrap_or_else(|| vec!["12.0".to_string()]);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out_dir.join(if is_windows {
        "dual_ffn_cutlass.obj"
    } else {
        "dual_ffn_cutlass.o"
    });
    let mut cmd = Command::new(&nvcc);
    #[cfg(windows)]
    {
        if let Some(ccbin) = msvc_host_dir() {
            cmd.arg("-ccbin").arg(ccbin);
        }
    }
    cmd.arg("-c")
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .arg("-O3")
        .arg("--std=c++17")
        .arg("--expt-relaxed-constexpr")
        .arg("-I")
        .arg(cutlass_root.join("include"))
        .arg("-I")
        .arg(cutlass_root.join("examples/45_dual_gemm"));
    for cap in &caps {
        let digits: String = cap.chars().filter(|c| c.is_ascii_digit()).collect();
        cmd.arg("-gencode")
            .arg(format!("arch=compute_{digits},code=sm_{digits}"));
    }
    if is_windows {
        cmd.arg("-Xcompiler")
            .arg("/Zc:preprocessor /Zc:__cplusplus /EHsc /bigobj /std:c++17");
    }
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", version_path.display());
    println!(
        "cargo:rerun-if-changed={}",
        cutlass_root.join("examples/45_dual_gemm").display()
    );
    println!("cargo:rerun-if-env-changed=KATAGO_CUTLASS_ROOT");
    println!("cargo:rerun-if-env-changed=KATAGO_ALLOW_BROKEN_CUDA_HOST");
    match cmd.status() {
        Ok(s) if s.success() => {}
        other => {
            // CUTLASS 在位但源码编译失败 = 提交了坏代码,必须 fail-loud。
            // (2026-08-17 事故:M2 提交了编译不过的 dual_ffn_cutlass.cu,
            //  此处静默跳过导致 katago_dualffn cfg 缺失、DUALFFN tactic
            //  空转,所有后续构建性能回退且无告警。)
            // 显式逃生门:KATAGO_ALLOW_BROKEN_CUDA_HOST=1(无 CUTLASS 环境
            // 下的降级开发场景仍可构建——那种场景在上面 root 检查已跳过)。
            if env::var("KATAGO_ALLOW_BROKEN_CUDA_HOST").as_deref() == Ok("1") {
                println!(
                    "cargo:warning=nvcc -c failed for dual_ffn_cutlass: {other:?} — skipped (KATAGO_ALLOW_BROKEN_CUDA_HOST=1)"
                );
                return None;
            }
            panic!(
                "nvcc -c failed for dual_ffn_cutlass (CUTLASS root {}): {other:?} — \
                 修复 cuda-host 源码,或设 KATAGO_ALLOW_BROKEN_CUDA_HOST=1 显式降级",
                cutlass_root.display()
            );
        }
    }

    // 对象文件打进静态库(cc 负责 ar/lib.exe 与链接指令)。
    let mut build = cc::Build::new();
    build.object(&obj);
    build.compile("katago_dual_ffn");

    // cudart 静态链接(CUTLASS device API 的主机调用依赖)。
    // 两平台静态库同名:Windows cudart_static.lib / Linux libcudart_static.a
    // (Linux 若写 "cudart" 会找不存在的 libcudart.a,链接失败)。
    let lib_dir = if is_windows {
        cuda_root.join("lib").join("x64")
    } else {
        cuda_root.join("lib64")
    };
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=cudart_static");
    // dual_ffn 的宿主对象是 C++(static-local guard/new),Linux 链接需要
    // libstdc++(Windows 侧 MSVC 运行时由 cudart_static 附带,无需显式)。
    if !is_windows {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    println!("cargo:rustc-cfg=katago_dualffn");
    Some(CutlassBuildInfo {
        version: cutlass_version,
        commit: cutlass_commit,
    })
}

fn parse_cutlass_version(path: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let value = |name: &str| -> Option<String> {
        text.lines()
            .find(|line| line.trim_start().starts_with(&format!("#define {name} ")))
            .and_then(|line| line.split_whitespace().nth(2))
            .map(str::to_string)
    };
    Some(format!(
        "{}.{}.{}",
        value("CUTLASS_MAJOR")?,
        value("CUTLASS_MINOR")?,
        value("CUTLASS_PATCH")?
    ))
}

fn git_source_id(root: &std::path::Path) -> Option<String> {
    use sha2::{Digest, Sha256};

    let commit_out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !commit_out.status.success() {
        return None;
    }
    let commit = String::from_utf8_lossy(&commit_out.stdout)
        .trim()
        .to_string();

    // A plain commit id is insufficient when CUTLASS has local edits. Hash the
    // relevant tracked diff and untracked files so two different dirty trees
    // cannot accidentally share a certified backend build id.
    let diff = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "diff",
            "--binary",
            "HEAD",
            "--",
            "include",
            "examples/45_dual_gemm",
        ])
        .output()
        .ok()?;
    let untracked = Command::new("git")
        .arg("-C")
        .arg(root)
        .args([
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "include",
            "examples/45_dual_gemm",
        ])
        .output()
        .ok()?;
    if diff.stdout.is_empty() && untracked.stdout.is_empty() {
        return Some(commit);
    }

    let mut hasher = Sha256::new();
    hasher.update(&diff.stdout);
    let names = String::from_utf8_lossy(&untracked.stdout);
    for name in names.lines().filter(|line| !line.is_empty()) {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        if let Ok(bytes) = std::fs::read(root.join(name)) {
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        }
    }
    Some(format!("{commit}+dirty.{}", hex::encode(hasher.finalize())))
}

fn nvcc_version_id(output: &str) -> String {
    output
        .lines()
        .find(|line| line.contains("Cuda compilation tools"))
        .and_then(|line| line.split_whitespace().find(|word| word.starts_with('V')))
        .map(|word| {
            word.trim_start_matches('V')
                .trim_end_matches(',')
                .to_string()
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Emit build metadata embedded in the CUDA backend. The build ID hashes only
/// inputs that can change generated device code or the DualGemm host wrapper,
/// so documentation-only Git commits do not invalidate a certified plan.
fn emit_cuda_build_info(
    cutlass: Option<&CutlassBuildInfo>,
    compiled_sms: &[String],
    attention_q64: bool,
    attention_q64_serial: bool,
) {
    use sha2::{Digest, Sha256};

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest_dir.join("../..");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernels_dir = manifest_dir.join("cuda-kernels");
    let host_dir = manifest_dir.join("cuda-host");
    let targets_path = workspace.join("configs/sm-targets.json");

    let nvcc_banner = find_nvcc()
        .and_then(|nvcc| Command::new(nvcc).arg("--version").output().ok())
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let nvcc_version = nvcc_version_id(&nvcc_banner);

    let mut inputs = vec![manifest_dir.join("build.rs"), targets_path.clone()];
    for dir in [&kernels_dir, &host_dir] {
        if let Ok(entries) = std::fs::read_dir(dir) {
            inputs.extend(
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| {
                        path.extension()
                            .is_some_and(|ext| ext == "cu" || ext == "h")
                    }),
            );
        }
    }
    inputs.sort();

    let cutlass_version = cutlass.map(|info| info.version.as_str()).unwrap_or("none");
    let cutlass_commit = cutlass.map(|info| info.commit.as_str()).unwrap_or("none");
    let mut hasher = Sha256::new();
    hasher.update(b"katago-cuda-build-v1\0");
    for path in &inputs {
        if let Ok(bytes) = std::fs::read(path) {
            hasher.update(
                path.strip_prefix(&workspace)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .as_bytes(),
            );
            hasher.update([0]);
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(&bytes);
        }
    }
    hasher.update(b"nvcc\0");
    hasher.update(nvcc_version.as_bytes());
    hasher.update(b"\0target\0");
    hasher.update(
        env::var("TARGET")
            .unwrap_or_else(|_| "unknown".to_string())
            .as_bytes(),
    );
    hasher.update(b"\0cutlass-version\0");
    hasher.update(cutlass_version.as_bytes());
    hasher.update(b"\0cutlass-commit\0");
    hasher.update(cutlass_commit.as_bytes());
    hasher.update(b"\0ptx-flags\0-ptx -O3 --std=c++17");
    hasher.update(b"\0dual-flags\0-c -O3 --std=c++17 --expt-relaxed-constexpr");
    if cfg!(windows) {
        hasher.update(b" -Xcompiler /Zc:preprocessor /Zc:__cplusplus /EHsc /bigobj /std:c++17");
    }
    for sm in compiled_sms {
        hasher.update(b"\0compiled-sm\0");
        hasher.update(sm.as_bytes());
    }
    let build_id = format!("sha256:{}", hex::encode(hasher.finalize()));

    let sms = compiled_sms
        .iter()
        .map(|sm| format!("{sm:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    let generated = format!(
        "// @generated by build.rs\n\
         pub const CUDA_KERNEL_BUILD_ID: &str = {build_id:?};\n\
         pub const CUDA_COMPILER: &str = {nvcc_version:?};\n\
         pub const CUDA_CUTLASS_VERSION: &str = {cutlass_version:?};\n\
         pub const CUDA_CUTLASS_COMMIT: &str = {cutlass_commit:?};\n\
         pub const CUDA_COMPILED_SMS: &[&str] = &[{sms}];\n\
         pub const CUDA_CAP_DUAL_FFN: bool = {};\n\
         pub const CUDA_CAP_ATTENTION_Q64: bool = {attention_q64};\n\
         pub const CUDA_CAP_ATTENTION_Q64_SERIAL: bool = {attention_q64_serial};\n",
        cutlass.is_some()
    );
    std::fs::write(out_dir.join("cuda_build.rs"), generated)
        .expect("failed to write cuda_build.rs");
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
            let p = PathBuf::from(&root).join("bin").join(if cfg!(windows) {
                "nvcc.exe"
            } else {
                "nvcc"
            });
            if p.exists() {
                return Some(p);
            }
        }
    }
    let plain = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    Command::new(plain)
        .arg("--version")
        .output()
        .ok()
        .map(|_| PathBuf::from(plain))
}

/// Compile the hand-written CUDA kernels (`cuda-kernels/*.cu`) to PTX for
/// every target listed in `configs/sm-targets.json`, and emit a Rust table
/// embedding the PTX bytes so the CUDA backend can load them at runtime.
fn compile_cuda_kernels() -> (Vec<String>, bool, bool) {
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
        return (Vec::new(), false, false);
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
        println!(
            "cargo:warning=cuda-kernels dir has no .cu files — CUDA backend will have no kernels"
        );
        return (Vec::new(), false, false);
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
    // 目录级监听：新增 .cu 文件时也触发重编译（文件级列表覆盖不了新文件）。
    println!("cargo:rerun-if-changed={}", kernels_dir.display());
    for cu in &cu_files {
        println!("cargo:rerun-if-changed={}", cu.display());
    }

    let Some(nvcc) = find_nvcc() else {
        println!("cargo:warning=nvcc not found — CUDA kernels not compiled");
        return (Vec::new(), false, false);
    };

    let mut table = String::from("// @generated by build.rs — CUDA kernel PTX per SM target\n");
    table.push_str("pub const CUDA_TARGETS: &[(&str, &str, &[(&str, &[u8])])] = &[\n");
    let mut compiled_sms = Vec::new();
    let q64_kernel_name = "attention_fa2_q64";
    let has_q64_kernel = cu_files
        .iter()
        .any(|path| path.file_stem().is_some_and(|name| name == q64_kernel_name));
    let mut attention_q64 = false;
    let mut attention_q64_serial = false;
    for t in &targets {
        let mut target_table = format!("    (\"{}\", \"{}\", &[\n", t.id, t.compute_capability);
        let mut target_complete = true;
        let mut target_q64 = false;
        let mut target_q64_serial = false;
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
                    if name == q64_kernel_name {
                        target_q64 = true;
                    }
                    if name == "attention_fa2_q64_serial" {
                        target_q64_serial = true;
                    }
                    target_table.push_str(&format!(
                        "        (\"{name}\", include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{}_{}.ptx\"))),\n",
                        t.id, name
                    ));
                }
                other => {
                    target_complete = false;
                    println!(
                        "cargo:warning=nvcc failed for kernel {name} (target {}): {other:?}",
                        t.id
                    );
                }
            }
        }
        target_table.push_str("    ]),\n");
        if target_complete {
            table.push_str(&target_table);
            attention_q64 |= target_q64;
            attention_q64_serial |= target_q64_serial;
            compiled_sms.push(
                t.compute_capability
                    .chars()
                    .filter(char::is_ascii_digit)
                    .collect(),
            );
        } else {
            println!(
                "cargo:warning=SM target {} omitted because one or more kernels failed to compile",
                t.id
            );
        }
    }
    table.push_str("];\n");
    if let Err(e) = std::fs::write(out_dir.join("cuda_kernels.rs"), table) {
        println!("cargo:warning=failed to write cuda_kernels.rs: {e}");
    }
    (
        compiled_sms,
        has_q64_kernel && attention_q64,
        attention_q64_serial,
    )
}

/// Directory containing cl.exe (MSVC host compiler), probed via the cc crate.
#[cfg(windows)]
fn msvc_host_dir() -> Option<PathBuf> {
    let tool = cc::Build::new().get_compiler();
    tool.path().parent().map(|p| p.to_path_buf())
}
