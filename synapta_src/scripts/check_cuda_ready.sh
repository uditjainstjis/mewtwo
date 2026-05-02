#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/cuda-path.sh

echo "== NVIDIA-SMI =="
nvidia-smi
echo

echo "== NVCC =="
nvcc --version
echo

echo "== Build CUDA smoke test =="
nvcc -O2 -o /archive/old_top_level/old_tmp/cuda_smoke_test /home/learner/Desktop/mewtwo/synapta_src/synapta_src/scripts/cuda_smoke_test.cu
echo

echo "== Run CUDA smoke test =="
/archive/old_top_level/old_tmp/cuda_smoke_test
