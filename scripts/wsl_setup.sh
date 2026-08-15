#!/bin/bash
# WSL 环境安装：Rust + CUDA toolkit(不装驱动,用 Windows 透传)
set -e
export DEBIAN_FRONTEND=noninteractive
echo "=== apt update + base tools ==="
apt-get update -qq
apt-get install -y -qq curl build-essential pkg-config libssl-dev ca-certificates python3 python3-pip

echo "=== rustup ==="
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh
  sh /tmp/rustup.sh -y --default-toolchain stable --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version

echo "=== cuda toolkit (wsl-ubuntu repo) ==="
if ! command -v nvcc >/dev/null 2>&1; then
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update -qq
  # 只装 nvcc + 运行时库(不装驱动/完整 toolkit 省时)
  apt-get install -y -qq cuda-nvcc-13-2 cuda-cudart-13-2 || apt-get install -y -qq cuda-nvcc cuda-cudart
fi
nvcc --version | tail -2

echo "=== all done ==="
which cargo nvcc
