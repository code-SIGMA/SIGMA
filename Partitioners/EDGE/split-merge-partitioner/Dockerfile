# -------------------------
# Base Image
# -------------------------
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        git \
        build-essential \
        g++ \
        gcc \
        make \
        cmake \
        pkg-config \
        libgoogle-glog-dev \
        libgflags-dev \
        libboost-all-dev \
        libopenmpi-dev \
        openmpi-bin \
        libsparsehash-dev \
        libgomp1 \
        python3 \
        python3-pip && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# Python dependencies
# -------------------------
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    networkx \
    ogb \
    PyYAML \
    pydantic

RUN pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
        dgl==2.5.0+cu121 \
        -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# -------------------------
# CMake latest version (3.26.4)
# -------------------------
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    sh cmake-3.26.4-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.26.4-linux-x86_64.sh

# -------------------------
# Working directory
# -------------------------
WORKDIR /app

# -------------------------
# Copy FSM source code
# -------------------------
COPY . /sources

# -------------------------
# Build FSM partitioner
# -------------------------
RUN --mount=type=bind,source=.,target=/sources,readonly=0 \
    sh -c " cd /sources && \
            mkdir build && \
            cd build && \
            cmake .. && \
            make -j\$(nproc) && \
            cp /sources/build/main /app/"

# -------------------------
# Runtime
# -------------------------
WORKDIR /app
CMD ["/bin/bash"]
