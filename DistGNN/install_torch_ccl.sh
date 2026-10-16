#!/bin/bash
set -e

cd /workspace/torch-ccl

export Torch_DIR=$(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/share/cmake/Torch')")

export CMAKE_C_COMPILER=gcc
export CMAKE_CXX_COMPILER=g++

python setup.py install
