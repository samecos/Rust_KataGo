# Portable native TF3 FP32 reference

`tf3-b11c768-fp32.json.gz` contains the existing, fully collected C++ CUDA FP32
reference for the original model SHA256 listed in `index.json`. It contains
outputs, not model weights. The 128 cases are 16 semantic positions under all
8 symmetries from `../worker_positions.json`; ownership is always included.

Source: `target/worker-parity-tf3-cuda-fp32/cpp/outputs.json`, documented in
`docs/Go-Server-Worker.md`. Source C++ binary SHA256, commit/backend declaration,
exact FP32 configuration, protocol/fixture/request digests and result identities
are retained in the compressed JSON. Export used `autotune_reference.py --source`
and checked the entire corpus; it did not regenerate or alter NN outputs.

The index pins compressed bytes. Loading also validates the corpus itself,
including request identities, shapes, finite values, normalization, model v17
and explicit `cudaUseFP16=false`. No model SHA, protocol or fixture substitution
is permitted. This is a trusted FP32 golden data artifact, not a new GPU tactic
certificate. A different model needs its own reference export.
