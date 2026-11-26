#!/usr/bin/env bash
# Self-contained local GPU environment (Linux / WSL2), no sudo needed:
# micromamba -> python 3.10 + PyTorch 2.1 (CUDA 12.1) + nvcc 12.1 -> GauHuman CUDA extensions.
# Usage: bash tools/setup_local_env.sh [env_prefix]   (default ~/vpskin-env)
set -euo pipefail
PREFIX="${1:-$HOME/vpskin-env}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MM="$HOME/.local/bin/micromamba"

if [ ! -x "$MM" ]; then
  mkdir -p "$HOME/.local/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$HOME/.micromamba"

if [ ! -d "$PREFIX" ]; then
  "$MM" create -y -p "$PREFIX" -c pytorch -c nvidia -c conda-forge \
    python=3.10 pytorch=2.1.2 torchvision=0.16.2 pytorch-cuda=12.1 \
    cuda-nvcc=12.1 cuda-cudart-dev=12.1 cuda-libraries-dev=12.1 "mkl<2024.1" ninja
fi
PY="$PREFIX/bin/python"
export CUDA_HOME="$PREFIX"
export PATH="$PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.9}"   # T4 (Kaggle) + RTX 40xx
# conda CUDA keeps cub / thrust / libcu++ (CCCL) under targets/: make them visible to nvcc
export CPATH="$PREFIX/targets/x86_64-linux/include/cccl:$PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"

"$PY" -m pip install -q "setuptools<70" "numpy<2" "opencv-python-headless<4.11" "plyfile<1.1" lpips==0.1.4 tqdm "imageio<3" pytest scipy pandas matplotlib h5py tensorboard
cd "$REPO"
"$PY" -m pip install -q --no-build-isolation submodules/diff-gaussian-rasterization submodules/simple-knn
"$PY" -c "import torch, diff_gaussian_rasterization, simple_knn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "OK: activate with  export PATH=$PREFIX/bin:\$PATH"
