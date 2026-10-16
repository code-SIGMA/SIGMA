FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libmetis-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python3 -m pip install --upgrade pip

# Python dependencies including torchdata
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    networkx \
    metis

RUN pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
        dgl==2.5.0+cu121 \
        -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

WORKDIR /workspace

COPY ./run_partition.py /workspace/

CMD ["/bin/bash"]
