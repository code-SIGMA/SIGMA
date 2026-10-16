#!/bin/bash

set -e
mkdir -p sub407 && cd sub407 || exit 1
# update following line to setup gcc 8.3.0 compiler
#source /swtools/gcc/8.3.0/gcc_vars.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda3

echo "Accept Channels Term of Service for main and r..."
miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

miniconda3/bin/conda create -y -n sub407 python=3.7.10

echo "Activating conda env..."
source miniconda3/bin/activate sub407

echo "Installing packages...."
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi jemalloc tqdm future pydot scikit-learn
conda install -y -c conda-forge gperftools onnx tensorboardx libunwind libpq psycopg

#echo "Install pytorch..."
#conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch

# Explicitly install torch without CUDA (should probably be fixed) {
conda config --env --set channel_priority strict
export CC=gcc
export CXX=g++
conda install -y pip
pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
echo $?
# }

# Explicit additional libs {
pip install "networkx==2.6.3" "urllib3<2.0" "requests<2.32" psutil
# }


echo "Install torch-ccl..."
export Torch_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake', 'Torch'))")

(
  git clone https://github.com/ddkalamk/torch-ccl.git -b working_1.7 torch-ccl-Working_1.7
  cd torch-ccl-Working_1.7
  git checkout 633a77e
  git submodule sync --recursive
  git submodule update --init --recursive
  CC=gcc CXX=g++ python setup.py install
)

echo "Install DGL..."
(
  git clone --recursive https://github.com/sanchit-misra/dgl.git -b xeon-optimizations dgl_XEON
  cd dgl_XEON
  git checkout 7558ffa
  git submodule sync --recursive
  git submodule update --init --recursive
  rm -rf build && mkdir build && cd build
  CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j"$(nproc)"
  cd ../python
  python setup.py clean --all
  CC=gcc CXX=g++ python setup.py install
)

echo "Installing few more packages.."
pip install psycopg_pool
pip install nvidia-ml-py
pip install ogb

echo "All installations done !!!"
