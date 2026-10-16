FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ARG CMAKE_VERSION=3.26.4
ARG TORCH_VERSION=2.0.1
ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Debug-Tools
        gdb \
        # Build-Tools
        ca-certificates \
        wget \
        git \
        g++ \
        gcc \
        make \
        pkg-config \
        # Runtime-Libraries
        libgomp1 \
        libgoogle-glog0v5 \
        libgflags2.2 \
        libboost-mpi1.74.0 \
        libopenmpi3 \
        # Development-Libraries
        libgoogle-glog-dev \
        libgflags-dev \
        libboost-all-dev \
        libopenmpi-dev \
        libargtable2-dev \
        libsparsehash-dev \
        openmpi-bin \
        # Python
        python3 \
        python3-pip \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    chmod +x cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    ./cmake-${CMAKE_VERSION}-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
        "numpy<2.0" \
        ogb \
        PyYAML \
        pydantic

RUN pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
        dgl==2.5.0+cu121 \
        -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

WORKDIR /app

COPY . /sources

RUN cd /sources && \
    mkdir -p build && \
    cd build && \
    #cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_FLAGS="-w" .. && \
    cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg .. && \
    make -j$(nproc) && \
    cp verge /app/ && \
    cd /app && \
    rm -rf /sources/build

CMD ["/bin/bash"]]
