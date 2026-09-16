# Strict B14 attention — local development bundle, revision 1

This is an opt-in development asset, **not a production certification or a
published redistributable package**. The measured strict r2 CUBIN and no-vcopy
run-r1 helper PTX are copied byte for byte. No compiler or GPU was run while
preparing this bundle. `manifest.json` inventories the source and evidence;
`abi.json` is the separate immutable runtime contract.

| Runtime file | SHA256 |
| --- | --- |
| `attention.sm120.cubin` | `90341b6d5c54a69e983a7e7d8b62956748785ad44c3e4e1b2734b92f3da46749` |
| `rope_no_vcopy.sm120.ptx` | `3a5448f8cb639fca97bf13ce17b66f12bdc83c4875b8f146447e7c861f33191a` |
| `abi.json` | `8c8ef96685ce5dceceb1aaf0d5c1b8aa30183defd1affbd08c983c4ff5f152ae` |

The helper is 24,529 bytes and retains its other diagnostic entries. Load only
`probe_rope_no_vcopy`. Its measured geometry is grid `[948,1,1]`, block
`[1024,1,1]`, zero dynamic shared memory. The unchanged attention entry is named
in `abi.json`; it uses grid `[3,12,14]`, block `[128,1,1]`, 20,480 shared bytes.
This artifact is fixed to SM120, B14, S361, H12, D32 and the recorded packed
strides. It has no runtime batch/shape/stride parameters.

Q/K use the original explicit FP32 multiply/subtract/FMA sequence and half RN
stores. The helper does not write V; attention reads V at original input +1536
bytes while Q/K read the separate scratch buffer. Input and scratch must stay
alive until the same stream completes. QK and PV accumulate in FP32; softmax
uses natural FP32 scale, the ordinary expf corrective sequence, FP32 sums and
`rcp.rn.f32`. P and the final output retain their recorded half boundaries.
Different attention tiling/online reduction order can still change output bits.

The copied run-r1 evidence passed the attention numerical guards. Its attention
comparisons passed the registered ABBA gates; helper-only seed 1 was unstable
and remains diagnostic. These results do not replace whole-model all-head
correctness, graph/partial-batch handling, or Worker performance validation.
Historical `EXPORTED_UNVERIFIED` manifests are retained unchanged.

## Identity and CPU verification

The runtime fingerprint is `sha256:` of the concatenation below, in this exact
order. Lengths are byte lengths; integers use little-endian encoding:

1. `b"Rust_KataGo strict-attention AOT\0"`, then revision `u32(1)`.
2. For each of `attention.sm120.cubin`, `rope_no_vcopy.sm120.ptx`, `abi.json`:
   `u64(name_length)`, UTF-8 name, `u64(data_length)`, actual file bytes.

The provenance manifest is not part of that runtime fingerprint. Changing any
runtime file creates a new artifact identity; never update a certified identity
by copying its old hash. This identity is separate from the pre-existing core
CUDA kernel identity and must be explicitly bound by a plan enabling the new
attention tactic.

From the repository root, this command only reads and hashes files:

```powershell
./.venv/Scripts/python.exe scripts/probe/fa4_strict_bundle.py verify
```

## Source and dependency reproduction

`source/build_fa4_strict.py` is the exact measured generator; `source/reference`
contains its original inputs, and `source/generated` contains the strict
outputs. `patches` shows the softmax/import/scale changes without changing the
historical files. `source/helpers.cu` and `source/probe_fork_fa4.rs` are the exact
no-vcopy run-r1 sources, not re-created approximations. The original helper
compiler options, source hashes and available compiler-file observations are
recorded in `locks/helper-toolchain.json`.

The AOT source provenance is FlashAttention commit
`145b1010051dbfd4bdc41a0ae55d495b08d7a458`, as recorded by the original WSL
source-build manifest. Its custom wheel hash is
`9ce90d0c89282558f53df3b0a44844c32050f4f4a1dc59f2d0ebd7a578f8366c`.
The reference Fork checkout was `5dfd8cb16bc0393518bdadcd1fe55ee1252da1a8` from
`https://github.com/doomoooo/KataGomo_fork.git`. The separate Rust DualFFN
CUTLASS 3.9.2 installation is not this CuTe DSL toolchain.

`locks/toolchain.lock.json` records the measured WSL environment's Python,
all installed distribution versions, 499 actual codegen package files
(including the CuTe compiler shared objects), and CUDA compiler/libdevice
files. `locks/requirements.versions.txt` is the explicit installed version set.
**Quack 0.6.4 declares DSL 4.6.2, while this tested environment uses DSL 4.7.0.**
Do not let dependency resolution silently replace the tested combination.
Reconstruction uses the explicit version set with `--no-deps` and then the
byte-lock verifier. This package contains no compiler wheels or shared objects.

Only the custom FlashAttention wheel archive hash is currently available here;
this lock is not a self-contained wheelhouse or a `pip --require-hashes` file.
Exact licensed packages must first be obtained for another machine. Version
matching alone is insufficient: the verifier also checks the locked codegen
files. The other dependencies are version-bound, not all byte-hashed.

The following only verifies the existing WSL toolchain and bundle:

```powershell
wsl.exe -d Ubuntu-24.04 -- /root/katagomo-fullflow/.final-migration-env/venv/bin/python /mnt/d/code/Rust_KataGo/scripts/probe/fa4_strict_bundle.py verify-toolchain
```

An explicit future build window can regenerate the attention export with the
same unchanged generator and bundled inputs. It uses fake tensors/stream and
hides CUDA devices. This command was **not run** when packaging:

```powershell
wsl.exe -d Ubuntu-24.04 -- /root/katagomo-fullflow/.final-migration-env/venv/bin/python /mnt/d/code/Rust_KataGo/scripts/probe/fa4_strict_bundle.py rebuild --output /mnt/d/code/Rust_KataGo/target/fa4-strict-rebuild-fresh
```

The output must be fresh. Original PTX debug locations contain absolute source
and venv paths, so relocation can change PTX/CUBIN bytes. Bitwise regeneration
has not been demonstrated. A changed export is a new candidate requiring codegen
semantics, ABI, numerical and performance validation; the script never replaces
this measured bundle. It regenerates attention only: helper PTX remains the
original measured NVRTC export, with its complete source/options retained for
a separately scheduled regeneration.

## Licenses and notices

This directory is not uniformly BSD-licensed. Original notices are preserved:

- `licenses/LICENSE.KataGo`: Fork generator/source terms and upstream notices.
- `licenses/LICENSE.FlashAttention`, `LICENSE.FlashAttention.CuTe`, and
  `AUTHORS.FlashAttention`: FlashAttention source copyright/BSD terms.
- `licenses/LICENSE.CUTLASS`: CUTLASS terms, including its separate DSL notice.
- `licenses/LICENSE.CuTeDSL`: the actual installed NVIDIA DSL software license.
- `licenses/LICENSE.Quack`: the actual installed Quack Apache 2.0 license.

The NVIDIA compiler/runtime packages themselves are external tools and are not
included. Keep their independent terms and the application/source notices when
reviewing any future distribution; do not relabel them under the repository's
general license. See the original
[NVIDIA DSL terms](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html).
