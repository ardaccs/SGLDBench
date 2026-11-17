#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
GPUS="${1:-2}"
nvcc -O2 -std=c++17 -Xcompiler -fopenmp test_global_reductions.cu -lcublas -lnccl -o test_global_reductions
./test_global_reductions "$GPUS"
