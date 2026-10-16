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
        libomp-dev \
        python3 \
        python3-pip && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# Python dependencies
# -------------------------
RUN pip install --no-cache-dir \
        "numpy<2.0" \
        "ogb" \
        "PyYAML" \
        "pydantic"

RUN pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
        dgl==2.5.0+cu121 \
        -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# -------------------------
# Latest CMake (optional but kept from original)
# -------------------------
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    sh cmake-3.26.4-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.26.4-linux-x86_64.sh

# -------------------------
# Working directory
# -------------------------
WORKDIR /app

# -------------------------
# Copy source code
# -------------------------
COPY . /sources

# -------------------------
# Build CUTTANA partitioner
# -------------------------
#   RUN --mount=type=bind,source=.,target=/sources,readonly=0 \
#       sh -c "cd /sources && \
#           make clean && \
#           make CFLAGS='-g -O1 -fsanitize=thread -fno-omit-frame-pointer -std=c++20 -I utils -I queue -fopenmp' \
#                   LDFLAGS='-fsanitize=thread' && \
#           cp ogpart /app/ && \
#           cp /sources/*.py /app/ 2>/dev/null || true"
RUN --mount=type=bind,source=.,target=/sources,readonly=0 \
    sh -c "cd /sources && \
        make ogpartsinglethread CFLAGS='-g -O0 -fno-omit-frame-pointer -std=c++20 -I utils -I queue' && \
        cp ogpart /app/"


CMD ["/bin/bash"]
