# Rust ↔ C++ FFI 桥接规范

> 用于在 Rust 移植过程中临时调用尚未迁移的 C++ 模块（主要是 CUDA/TensorRT/OpenCL/Metal NN 后端）。

---

## 1. 设计目标

1. **最小化 FFI 表面积**：只暴露必要的结构体和函数，不把整个 C++ 类体系搬过边界。
2. **逐步拆除**：每当 Rust 侧完成一个模块，立即删除对应 shim。
3. **零成本抽象**：FFI 层只做数据转换，不承载业务逻辑。
4. **内存安全**：所有权、生命周期、错误码必须清晰；禁止跨边界传递裸指针做长期借用。

---

## 2. 推荐工具链

| 场景 | 工具 | 说明 |
|------|------|------|
| C++ ↔ Rust 自动生成 | `cxx` | 推荐用于函数/简单 struct 桥接 |
| C API 头生成 | `cbindgen` | 如需把 Rust 暴露给 C++ |
| 构建 C++ shim | `cc` crate in `build.rs` | 编译静态库 |
| 复杂对象生命周期 | opaque pointer (`struct Foo;`) | C++ 侧管理内存，Rust 侧持有 `Box` 或自定义 drop |

---

## 3. 典型使用模式

### 3.1 C++ 暴露给 Rust：NN 后端

#### C++ 侧（`cpp-shim/src/backend_trt.cpp`）
```cpp
#include <cstdint>
#include <memory>
#include <vector>

struct TrtBackend {
    std::vector<float> inputBuffer;
    std::vector<float> outputBuffer;
    // ... 真实 TensorRT 上下文
};

extern "C" {
    TrtBackend* trt_backend_create(const char* model_path, int max_batch_size);
    void trt_backend_destroy(TrtBackend* backend);
    int trt_backend_eval(
        TrtBackend* backend,
        const float* input, int input_size,
        float* policy_out, int policy_size,
        float* value_out, int value_size,
        float* ownership_out, int ownership_size
    );
}
```

#### Rust 侧（`kata_nn/src/backends/trt_ffi.rs`）
```rust
use std::ffi::CString;
use std::os::raw::{c_char, c_float, c_int};
use std::ptr::NonNull;

#[repr(C)]
pub struct TrtBackend {
    _opaque: [u8; 0],
}

extern "C" {
    fn trt_backend_create(model_path: *const c_char, max_batch_size: c_int) -> *mut TrtBackend;
    fn trt_backend_destroy(backend: *mut TrtBackend);
    fn trt_backend_eval(
        backend: *mut TrtBackend,
        input: *const c_float,
        input_size: c_int,
        policy_out: *mut c_float,
        policy_size: c_int,
        value_out: *mut c_float,
        value_size: c_int,
        ownership_out: *mut c_float,
        ownership_size: c_int,
    ) -> c_int;
}

pub struct TrtBackendHandle {
    ptr: NonNull<TrtBackend>,
}

impl TrtBackendHandle {
    pub fn new(model_path: &str, max_batch_size: usize) -> anyhow::Result<Self> {
        let c_path = CString::new(model_path)?;
        let ptr = unsafe { trt_backend_create(c_path.as_ptr(), max_batch_size as c_int) };
        let ptr = NonNull::new(ptr).ok_or_else(|| anyhow::anyhow!("trt_backend_create failed"))?;
        Ok(Self { ptr })
    }

    pub fn eval(&mut self, input: &[f32], policy: &mut [f32], value: &mut [f32], ownership: &mut [f32]) -> anyhow::Result<()> {
        let rc = unsafe {
            trt_backend_eval(
                self.ptr.as_ptr(),
                input.as_ptr(), input.len() as c_int,
                policy.as_mut_ptr(), policy.len() as c_int,
                value.as_mut_ptr(), value.len() as c_int,
                ownership.as_mut_ptr(), ownership.len() as c_int,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(anyhow::anyhow!("trt_backend_eval returned {rc}"))
        }
    }
}

impl Drop for TrtBackendHandle {
    fn drop(&mut self) {
        unsafe { trt_backend_destroy(self.ptr.as_ptr()) };
    }
}
```

---

## 4. 数据类型映射规范

| C++ | Rust FFI | Rust 高层 | 备注 |
|-----|----------|-----------|------|
| `bool` | `u8` / `c_bool` | `bool` | C++ `bool` 大小未定义，建议用 `u8` |
| `int32_t` | `i32` | `i32` | |
| `int64_t` | `i64` | `i64` | |
| `uint32_t` | `u32` | `u32` | |
| `uint64_t` | `u64` | `u64` | |
| `float` | `f32` | `f32` | |
| `double` | `f64` | `f64` | |
| `const char*` | `*const c_char` | `CString` / `&str` | 传入需 `CString`；传出需约定编码与所有权 |
| `std::string` | 不透明指针或 `const char*` + 长度 | `String` | 不推荐直接传递 std::string 对象 |
| `std::vector<T>` | `const T*` + `size_t` | `Vec<T>` / `ndarray` | 数组数据在边界处拷贝 |
| 枚举 | `i32` / `u32` | 自定义枚举 `#[repr(C)]` | 显式指定值 |
| 回调函数 | `extern "C" fn` | `Box<dyn Fn>` | 注意生命周期 |

---

## 5. 错误处理规范

### 5.1 简单错误码
- 返回 `c_int`：0 表示成功，非 0 表示错误码。
- Rust 侧将错误码转换为 `anyhow::Error` 或自定义错误。

### 5.2 带消息的错误
- C++ 侧提供 `int last_error(char* buf, size_t buf_len)`。
- Rust 侧在调用失败后再取一次错误信息。

```cpp
extern "C" int last_error(char* buf, size_t buf_len);
```

```rust
extern "C" { fn last_error(buf: *mut c_char, buf_len: usize) -> c_int; }
```

---

## 6. 所有权与生命周期

| 所有权模式 | 做法 |
|------------|------|
| Rust 创建，Rust 销毁 | C++ 返回裸指针；Rust 用 `Box`/自定义 struct + `Drop` 管理 |
| C++ 创建，C++ 销毁 | 不透明指针；C++ 提供 `*_destroy`，Rust `Drop` 中调用 |
| 借用 | 调用期间有效，禁止跨调用保留指针； Rust 用 `ManuallyDrop` 临时借用 |

**禁止**：在异步上下文中持有 C++ 裸指针跨越 await 点，除非能保证 C++ 对象不会被释放。

---

## 7. 线程安全

- C++ shim 中的对象若会被多线程访问，必须内部加锁或用 TLS。
- Rust 侧可将 `Send + Sync` 标记在包装类型上，**但前提是 C++ 对象确实线程安全**。

```rust
unsafe impl Send for TrtBackendHandle {}
unsafe impl Sync for TrtBackendHandle {}
```

---

## 8. build.rs 示例

```rust
use std::env;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    cc::Build::new()
        .cpp(true)
        .include("../KataGo/cpp")
        .include("../KataGo/cpp/external")
        .file("src/trt_shim.cpp")
        .compile("trt_shim");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=trt_shim");
    // 若链接 TensorRT / CUDA 等系统库，继续添加：
    // println!("cargo:rustc-link-lib=nvinfer");
}
```

---

## 9. 拆除计划

| 阶段 | 拆除动作 |
|------|----------|
| NN CPU 后端完成 | 删除 dummy/eigen shim |
| NN CUDA 后端完成 | 删除 trt/cuda shim |
| NN OpenCL 后端完成 | 删除 opencl shim |
| NN Metal 后端完成 | 删除 metal shim |
| 全部完成 | 删除 `cpp-shim/` 目录 |

---

## 10. 注意事项

- C++ 异常**不得**跨越 FFI 边界，shim 内用 `try/catch` 转成错误码。
- 尽量使用 `cxx` crate 处理复杂 struct，避免手写 `#[repr(C)]` 出现 ABI 不对齐。
- 所有 shim 函数必须加 `extern "C"`。
- 跨边界传递字符串统一使用 UTF-8（C++ 侧 `std::string` 转 UTF-8）。
