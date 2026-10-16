FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/miniconda3 \
    PYTHON_VERSION=3.10 \
    NUMPY_VERSION=1.25 \
    PANDAS_VERSION=2.3 \
    DGLBACKEND=pytorch \
    PYTORCH_VERSION=2.3.0 \
    TORCHVISION_VERSION=0.18.0 \
    TORCHAUDIO_VERSION=2.3.0 \
    DGL_VERSION=2.5.0 \
    DGL_CUDA_SUFFIX=cu121 \
    DGL_TORCH_TAG=torch-2.3 \
    PATH=/opt/miniconda3/bin:$PATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda and must match with PYTHON_VERSION
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_25.11.1-1-Linux-x86_64.sh \
        -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy && \
    find $CONDA_DIR -follow -type f -name '*.a' -delete && \
    find $CONDA_DIR -follow -type f -name '*.pyc' -delete

# Accept Anaconda Terms of Service and install Python packages
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda install -y python=$PYTHON_VERSION && \
    conda install -y -c conda-forge numpy=$NUMPY_VERSION && \
    conda install -y -c conda-forge pandas=$PANDAS_VERSION && \
    conda clean -afy && \
    find $CONDA_DIR -follow -type f -name '*.a' -delete && \
    find $CONDA_DIR -follow -type f -name '*.pyc' -delete

# Install PyTorch 2.3 with CUDA 12.1 support
# Pinned to match the training image so saved .dgl files load identically.
RUN pip install --no-cache-dir \
        torch==$PYTORCH_VERSION \
        torchvision==$TORCHVISION_VERSION \
        torchaudio==$TORCHAUDIO_VERSION \
        --index-url https://download.pytorch.org/whl/cu121

# Additional deps needed by the download scripts (requests for twitch script)
RUN pip install --no-cache-dir \
        pyyaml \
        "urllib3<2.0" \
        "requests<2.32" \
        "scikit-learn>=0.20.0" \
        "outdated>=0.2.0"

RUN pip install --no-cache-dir --no-deps ogb \
    && pip install --no-cache-dir six

# make the 'decide_download' from ogb return True as default.
RUN python_package_path=$(python -c "import site; print(site.getsitepackages()[0])") && \
    sed -i '/def decide_download(url):/a\    return True' \
        "$python_package_path/ogb/utils/url.py"

# Install DGL 2.5.0 for cu121 / torch 2.3 via pip
# (dglteam conda channel has no 2.5 builds; pip is the only route.)
RUN pip install --no-cache-dir \
        dgl==${DGL_VERSION}+${DGL_CUDA_SUFFIX} \
        -f https://data.dgl.ai/wheels/${DGL_TORCH_TAG}/${DGL_CUDA_SUFFIX}/repo.html

# Verification: fail the build if versions drift
RUN python - <<'EOF'
import torch, dgl
assert torch.__version__.startswith('2.3.'),  f'torch is {torch.__version__}'
assert dgl.__version__.startswith('2.5.'),    f'dgl is {dgl.__version__}'
assert torch.version.cuda.startswith('12.1'), f'cuda is {torch.version.cuda}'
print('OK: torch', torch.__version__, 'dgl', dgl.__version__)
EOF

WORKDIR /workspace
COPY ./ /tools
CMD ["/bin/bash"]
