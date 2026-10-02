# SM120 MXFP8 独立原型

这是 G3 的独立正确性探针。不进入 RustGo `build.rs`，不读取模型，不修改正式后端，也不代表模型 PTQ 精度或性能验收。

首阶段六个手工已量化矩阵 case 已由主任务在 RTX 5070 Ti 编译并运行，全数通过，FP16/FP32/FP32残差 `bit_mismatches=0`。记录为 cuBLASLt runtime `130600`、算法67、numerical flags `0xc00202`；原始证据在 `target/mxfp8-probe/{build,cpu,gpu}.log`。

新增 GPU block32 量化、转换边界、动态量化与 GEMM 的 Graph 回归在修复流依赖后，已由主任务在同一 RTX 5070 Ti 重新编译并验证：CPU自检通过，GPU **10 PASS / 0 UNSUPPORTED / 0 FAIL**。日志为 `target/mxfp8-probe/quantizer-r2-build.log`、`quantizer-r2-cpu.log`、`quantizer-r2-gpu.log`，没有放宽oracle或数值门。真实 cuBLASLt 执行 E4M3 × E4M3、K32 分组 UE8M0 scale 的 MXFP8 GEMM；不存在 FP16 GEMM 回退。模型接入、RMSNorm/SwiGLU融合与测速尚未实施。

首轮扩展的构建与CPU自检通过，但GPU结果为8 PASS / 2 FAIL，失败日志保留于 `target/mxfp8-probe/quantizer-gpu.log`。K344量化及其Graph前置权重量化在首字节失败，原六个GEMM、796个转换边界和K168完整Graph均通过。这一历史结果不能与修复后的r2验证混用。

该失败的实际byte为253（`0xFD`），与把设备初始化canary `0xCDCDCDCD` 解释为FP32、除以block scale后量化成−416的编码完全相符。代码原先在默认流调用`cudaMemset/cudaMemcpy`，量化却使用nonblocking stream，缺少生产者→消费者依赖。现在每个DeviceBuffer显式绑定同一条流：初始化、上传、下载和guard读取全部在owner stream排队；上传返回前同步该流，保证临时host向量存活到传输结束；下载同样完成后才访问host结果。Graph内device-to-device拷贝仍在捕获流内异步执行。r2复验中此前两个失败项均已通过，其余八项继续通过；量化bytes/scales逐位门、GEMM误差界、Graph A→B→A与fresh direct逐位门均保持原值。NVIDIA明确说明pageable H2D `cudaMemcpy`返回时DMA可能仍未完成，`cudaMemset`对host也可异步，不能仅由函数名推断跨流依赖。[CUDA 13.3同步语义](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-runtime-api/api-sync-behavior.html)

在有 MSVC 编译环境、CUDA 13.3 和库搜索路径的终端中单独编译，例如 PowerShell：

```powershell
New-Item -ItemType Directory -Force target/mxfp8-probe | Out-Null
& "$env:CUDA_PATH/bin/nvcc.exe" -std=c++17 -arch=sm_120 -O2 --ftz=false --prec-div=true experiments/mxfp8_probe.cu -lcublasLt -lcublas -o target/mxfp8-probe/mxfp8_probe_quantizer.exe
```

不要传 `--use_fast_math`。本机 13.3 的 DLL 位于 `$env:CUDA_PATH/bin/x64`；运行进程的 PATH 应包含该目录。原型打印实际 CUDA runtime/driver API、cuBLASLt runtime、编译头文件/nvcc 版本和 Windows 已加载库路径。

```powershell
& target/mxfp8-probe/mxfp8_probe_quantizer.exe --cpu-only
& target/mxfp8-probe/mxfp8_probe_quantizer.exe --device 0 2>&1 | Tee-Object target/mxfp8-probe/quantizer-gpu.log
```

`--cpu-only` 只检查独立 CPU 解码、half RNE、正有限 half 穷举往返、E4M3 RNE/SATFINITE、E8M0上取整和零块/padding；不初始化 GPU，不代表量化kernel或GEMM通过。完整运行原六个case，再执行转换边界、量化器、两种输出的Graph回归，共10个顶层结果；每项输出 `PASS`、`UNSUPPORTED` 或 `FAIL`。返回码0表示全部通过，1表示执行/数值/内存保护等失败，2表示至少一项不支持且没有其他失败。不支持的FP32 case仍单独保留结果，不能从日志中删除后声称全部通过。

| case | tokens | 输出通道 | 原 K → GEMM K | 输出 | beta |
|---|---:|---:|---:|---|---:|
| 跨 scale tile | 361 | 136 | 160 → 160 | FP16 | 0 |
| B11 投影形状 | 361 | 384 | 384 → 384 | FP16 | 0 |
| B15 风格剪枝 K | 17 | 384 | 344 → 352 | FP16 | 0 |
| 小通道与尾块 | 129 | 24 | 72 → 96 | FP16 | 0 |
| FP32 输出 | 361 | 136 | 160 → 160 | FP32 | 0 |
| FP32 原位残差 | 361 | 136 | 160 → 160 | FP32 | 1 |

布局采用现有引擎的转置映射：行主序 `X[tokens,K] × W[channels,K]ᵀ` 在 cuBLAS 中是 `m=channels,n=tokens,k=Kpad`，A=权重且转置，B=激活且不转置。C/D 使用同一指针及列主序描述符，最终存储仍为引擎的行主序。K 补至32，FP8尾值清零。Scale 按128×4 tile排列，完整tile分配、越界字节清零；A/B scale指针设置后保持稳定，设备内存存活时间覆盖描述符和全部GPU执行。

CPU oracle 独立解码手工 E4M3 数据和**逻辑 scale 表**，不读取 GPU 使用的 swizzle表。它计算 FP32 FMA，同时用 double 检查所选二进制精确数据确实没有 FP32 累加舍入；因此不同归约顺序不构成放宽误差的理由。FP16 输出参考由独立 CPU IEEE half RNE 转换得到，FP16/FP32 都要求逐位相同且有限。另报告相对存储参考和 FP32 参考的最大绝对误差；FP16相对FP32的舍入差允许存在，不能误读为GEMM错误。

每个 case 先请求至多32个heuristic，再做 `cublasLtMatmulAlgoCheck`；选择第一个满足workspace及Tensor Core/FP32累加/E4M3 numerical flags的候选。记录算法ID、tile、split-K、reduction及flags。只执行一次，不在候选间按输出选择或测速；选择后失败直接记录。数值flags配合显式block-scale描述符用于API层路径审计，未来kernel/SASS验证应另行进行。

A/B/scale/C/D/workspace均有256字节首尾guard，检查只读A/B/scale内容不变。检查不能代替Compute Sanitizer。这份验证不能证明真实模型误差、Tensor Core利用率或端到端收益。

## 新增量化器与 Graph 验证

量化内核按一个warp处理一个K32块。激活读取真实half存储后转FP32；权重读取原始FP32，不先经过half舍入。这是本原型明确固定的语义，与以FP16权重为起点的量化存在区别。FP32 absmax除以448，使用显式 `__fdiv_rn`，然后E8M0向正无穷取整；数据使用E4M3 RNE + SATFINITE。全零块固定 `q=+0, scale=1`；其他块保留负零的符号。数据K尾补+0，packed scale越界tile槽清零。Scale byte0是 `2^-127`，不是数值零。

CPU参考枚举127个正有限E4M3值，选择最近值并按编码最低位做ties-even；不调用CUDA转换intrinsic。Scale参考对**已经舍入为FP32的商**使用`frexp`判上取整，再夹到UE8M0范围。GPU独立生成packed scale，CPU数学参考继续读取逻辑scale表。

覆盖包括：

- 所有126个E4M3相邻值midpoint、每个midpoint的正负上下相邻FP32值；次正规数、448边界、极大有限数的SATFINITE；多个E8M0 power-of-two边界。
- FP32与half零块、FP32极小/极大值、half次正规值、`amax/448`下溢及scale最低边界。
- K31/32/33、K65、剪枝风格K344→352；输入stride故意大于K且不按16字节对齐；行padding为NaN，必须忽略。
- 逻辑区域NaN、+Inf、-Inf由CPU拒绝，GPU错误位分别标记NaN=1、Inf=2，独立负例不发射GEMM。

先逐位比较完整FP8值数组与packed scale数组（含尾部和越界tile槽），另检查source未被写入。`quantization_max_abs`/`quantization_rmse`单独报告，不是算子执行误差。极大有限FP32权重可能量化成数学值`2^128`，超过FP32可表示范围；该边界只验证量化编码并以double报告量化误差，不混入有限输出GEMM用例。

两个新增完整管线：

| case | tokens | 输出通道 | 原K→GEMM K | 输入stride | 权重stride | 输出/beta |
|---|---:|---:|---:|---:|---:|---|
| 动态激活Graph | 361 | 136 | 344→352 | 351 | 357 | FP16/0 |
| 动态激活Graph残差 | 129 | 24 | 168→192 | 175 | 181 | FP32/1 |

两个case使用确定性随机有限输入。权重量化一次，激活每次动态量化；先比较量化bytes/scales，再执行GEMM。CPU分别计算已量化矩阵的顺序FP32 FMA、FP64参考及`sum(abs(a*b))`。执行误差预设为FP32 `gamma_(K+2) * sum_abs`，包含beta残差并补充FP64参考求和误差；FP16输出再将区间两端按独立half RNE转换。该界只覆盖浮点计算与存储，**不把量化误差加入容限**，不根据GPU结果事后放宽。相对量化前FP32权重/half激活的输出误差另列 `separate_quantization_output_max_abs`。

Graph捕获“清错误位/scale padding→激活量化→恢复C→MXFP8 GEMM”，描述符scale指针始终固定。A→B→A过程中两组输入的FP8值和scale都改变；每轮对照独立source/FP8/scale/status/output/workspace的fresh direct管线，要求最终输出逐位一致。beta=1每次从独立只读device residual恢复C，避免重复累加。另捕获重放一次NaN输入：GEMM节点仍会运行，但主机检查错误位并丢弃结果，绝不把它当作有效推理；下一次合法A须清除错误位并逐位恢复输出。原型没有实现设备内提前取消整个Graph，未来生产接入必须保留拒绝与丢弃契约。

实现依据：[CUDA 13.3 cuBLAS narrow precision](https://docs.nvidia.com/cuda/archive/13.3.0/cublas/index.html#narrow-precision-data-types-usage)、[官方 MXFP8 sample](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtMxfp8Matmul)。本机核对为 CUDA SDK 13.3.1 / cuBLAS 13.6.0.2；实际运行以日志为准。

## 整网接入契约（尚未实现）

以下是下一阶段的具体设计，不表示已接入生产后端。第一版只承接现有 FFN dual/down、attention QKV/out 的投影边界，RMSNorm、SwiGLU、attention核心和输出头继续使用现有实现。先验证原始投影管线，再另行评价融合与性能。

建议最小文件拆分为 `crates/kata_nn/src/backends/mxfp8.rs`（资源、权重、工作区、私有Lt描述符和算法准备）、`crates/kata_nn/cuda-kernels/mxfp8.cu`（half/FP32 block32量化、状态与必要的有限性检查）、`crates/kata_nn/tests/test_mxfp8.rs`（独立CPU codec/oracle与算子/Graph测试）。现有 cudarc 0.19.9 的cuBLASLt绑定已经包含E4M3、VEC32_UE8M0和scale属性，无需为了这些API新增C++ host shim；是否实际可运行仍须单独做能力检查。第一版不抽取或重构共用INT8基础设施。

Rust API草案如下；名字和签名用于约束所有权、调用顺序，尚非已提供接口：

```rust
Mxfp8Kernels::load(rt: &CudaRuntime) -> Result<Self, String>;
Mxfp8Weight::upload_from_f32(
    rt: &CudaRuntime, stream: &Arc<CudaStream>, kernels: &Mxfp8Kernels,
    source: &[f32], n: usize, k: usize, stride: usize,
) -> Result<Arc<Self>, String>;

enum Mxfp8Output { Half, Float, FloatResidual }
// 工作区持有runtime/stream；rows固定为该工作区的物理batch × 361。
Mxfp8Workspace::new(
    rt: Arc<CudaRuntime>, stream: Arc<CudaStream>, kernels: Mxfp8Kernels,
    rows: usize, max_k: usize,
) -> Result<Self, String>;
Mxfp8Workspace::prepare_projection(
    &mut self, weight: Arc<Mxfp8Weight>, output: Mxfp8Output,
) -> Result<PreparedProjectionId, String>;
Mxfp8Workspace::begin_forward(&mut self) -> Result<(), String>;
Mxfp8Workspace::project_half(
    &mut self, id: PreparedProjectionId, layer_id: u32,
    input: &CudaSlice<u16>, stride: usize, output: &mut CudaSlice<u16>,
) -> Result<(), String>;
Mxfp8Workspace::project_f32(
    &mut self, id: PreparedProjectionId, layer_id: u32,
    input: &CudaSlice<u16>, stride: usize, output: &mut CudaSlice<f32>,
) -> Result<(), String>;
Mxfp8Workspace::enqueue_status_copy(
    &self, destination: &mut PinnedHostSlice<u32>,
) -> Result<(), String>;
validate_completed_status(status: &[u32]) -> Result<(), String>;
```

`PreparedProjectionId`应为带工作区身份的私有字段类型；其他工作区的ID、错误输出类型、越界尺寸或stride都拒绝。`prepare_projection`在捕获外完成描述符、heuristic、AlgoCheck和实际warmup所需准备；`project_*`只执行已准备路径，不创建描述符、分配缓冲或选择算法。独立模块先可测试，后续另行修改生产注册、build、执行器、配方及身份绑定。

必须保持以下数值与资源契约：

1. **原FP32静态权重。** 从模型原始FP32 tensor读取，完成gate/up拼接或张量转置后直接量化，不经FP16权重上传/舍入。建议复用已验证的FP32 GPU量化kernel，在加载时仅执行一次；原FP32临时设备输入、host上传源一直存活至该流完成，检查加载专用状态后才释放。CPU枚举codec保持独立测试oracle，避免把每元素127值枚举用作正式加载器。存储语义固定为E4M3 RNE/SATFINITE、FP32商后的UE8M0向上取整、零块q=+0/S=1及当前scale布局，并纳入独立量化身份。非有限原权重直接拒绝；极端有限权重产生不可表示的FP32反量化值也必须明确拒绝，不能因编码合法就默认为可执行模型。

2. **half激活与输出边界。** 运行时量化器只接受已有half激活；RMSNorm/SwiGLU的FP32计算和原half落盘边界保留。以后融合也须先作等价half舍入再量化，不能直接量化尚未舍入的FP32中间值。GEMM固定E4M3×E4M3、FP32累加、`FAST_ACCUM=0`、FP32 alpha/beta。Half模式beta=0直接写half；Float模式beta=0写FP32；FloatResidual模式beta=1写原位FP32残差。后两者不能先转half再加残差。beta=1读取本次forward上游产生的当前残差，不能复制原型为了独立对拍而保留的固定初始C。

3. **scale指针随资源固定。** 权重对象拥有A的FP8值/packed scale；工作区拥有B的动态FP8值/packed scale。每个已准备投影持有对应权重的Arc，并绑定该工作区稳定的B-scale地址。INT8现有`(rows,n,kp)`描述符cache没有嵌入逐层scale指针，不能直接用于MXFP8：两个同形状权重也可能有完全不同的A-scale地址。第一版每个绑定投影独立保存LtPlan，避免错误复用；将来若去重，key至少包括权重身份、工作区身份、形状、输出类型和beta。移动Rust对象不会移动CudaSlice的设备地址；重新分配、替换或跨流复用其缓冲则必须销毁对应Graph/plan并重新准备。

4. **Graph与同流所有权。** 工作区、量化、GEMM、状态回拷均在同一owner stream；权重加载若使用另一流，加载返回前必须完成显式同步。加载阶段可以同步，forward仅同流异步排队，不把原型每次host上传的同步带入热路径。runtime保活到Graph及全部设备执行结束，Graph先销毁，再销毁plan/缓冲。复用`CudaRuntime::cublaslt_workspace_ptr/len`时，在捕获前触发分配并验证非零、大小与256B对齐；同流串行共享可行，不允许其他流共用该执行工作区。权重/激活scale地址、quantization缓冲和算法在每个Graph生存期固定，捕获前先用真实已准备管线warmup。现有runtime关闭自动event tracking，更不能依赖隐式跨流排序。

5. **每次forward仅清一次状态。** `begin_forward`在整次forward入口清设备状态并被捕获进Graph；每层量化仅`atomicOr`非有限错误位，不能沿用原型“每个量化调用都清状态”的单算子习惯。可附首个出错layer ID。量化发现错误时写确定的安全占位，Graph后续节点可以继续，但整批输出必须丢弃；建议对MXFP8 GEMM输出做有限性检查并累计单独错误位，覆盖有限输入产生FP16/FP32溢出的情况。没有主机逐层同步。

6. **服务双槽必须保留自己的状态快照。** 当前`CudaBatchState`两个slot同流串行、共享device工作区，`CudaSlot`各自拥有pinned输出。每次submit在本次Graph/direct之后，将sticky状态同流回拷到该slot独立的pinned状态，再record其done_event；`finish_output`等事件后，先检查该slot状态再解码、缓存或发送输出。不能finish时读取共享device状态，因为下一次forward可能已经将它清零。测试用`CudaWorkspace::to_host`也须检查；只改该helper不能保护真实Worker链路。

7. **B15尾部必须分清数据与scale。** 逻辑K保持模型值，GEMM K补至32，权重与激活K尾填+0；输入stride可大于逻辑K，padding不参与amax/非有限检测。输出N保持实际剪枝宽度，不为scale tile补回128或原始网络宽度。当前合法FFN hidden是8对齐：例如H=344时dual N=688无需补数据N，down的K=344补到352；两者scale的outer维仍分配完整128 tile。FP16输出LD要求8对齐、FP32输出LD要求4对齐；不满足这一最小实现范围的N应明确拒绝。若将来扩展到任意N，必须另加Npad、残差padding与输出裁剪的专门测试，不能只改描述符尺寸。scale内层块数补至4，其tile外字节每层可靠清零；多层共享缓冲、K大小来回变化必须纳入回归。

8. **能力与失败拒绝。** 初期明确限SM120；检查真实设备、cuBLASLt handle/runtime版本、编译出的量化符号。使用TN映射、VEC32_UE8M0和稳定scale指针，每个实际形状/输出类型均做heuristic与AlgoCheck，要求Tensor Core/E4M3/FP32累加flags，并验证全部指针、leading dimension、workspace约束。首次实际执行仍须成功，AlgoCheck并不保证实际指针对齐正确。无候选、属性不支持或运行失败均返回错误，不以FP16执行后仍报告MXFP8。记录实选算法配置、库/驱动/build与量化语义身份；本机algo67和runtime130600的成功不代表其他库版本或其他SM120设备已认证。[cuBLAS FP8约束与AlgoCheck边界](https://docs.nvidia.com/cuda/archive/13.3.0/cublas/index.html#cublasltmatmul)

下一阶段最小验收应保留独立CPU量化byte/scale门、原六个精确dyadic GEMM门及随机GEMM舍入误差界，并增加：两个**同形状不同scale权重**顺序调用；不同K/stride共享scratch且大小来回切换；非首层NaN/Inf后继续合法层仍报错；无效forward后合法forward恢复；两个slot在途、先后不同错误状态与物理batch切换；Graph A→B→A相对独立fresh direct逐位一致。算子门通过后，才能对B11/B15做整图和真实Worker的独立精度验收，随后决定是否进行性能实验。
