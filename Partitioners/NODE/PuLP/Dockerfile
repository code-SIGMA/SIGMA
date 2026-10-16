FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        gcc \
        make \
        pkg-config \
        wget \
        git \
        libgomp1 \
        autoconf \
        automake \
        libtool \
        flex \
        hwloc \
        python3 \
        python3-pip \
        libhwloc-dev && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# Python dependencies
# -------------------------
RUN pip install --no-cache-dir \
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

# Set the DGL backend explicitly to suppress the runtime warning
ENV DGLBACKEND=pytorch

# Build CUDA-aware OpenMPI from source (the default Ubuntu package is NOT CUDA-aware)
ENV OPENMPI_VERSION=4.1.6
RUN wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar xzf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local --with-cuda=/usr/local/cuda && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf openmpi-${OPENMPI_VERSION} openmpi-${OPENMPI_VERSION}.tar.gz && \
    ldconfig

ENV PATH="/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

# Run the build
RUN --mount=type=bind,source=.,target=/sources,readonly=0 \
    sh -c " cd /sources && \
            chmod +x install && \
            ./install && \
            mkdir -p /app/lib /app/include /app/bin && \
            cp -r lib/. /app/lib/ 2>/dev/null || true && \
            cp -r include/. /app/include/ 2>/dev/null || true && \
            find . -maxdepth 3 -type f -executable \( -name 'xtrapulp' -o -name 'pulp*' \) -exec cp {} /app/bin/ \; 2>/dev/null || true"

ENV PATH="/app/bin:${PATH}"
ENV LD_LIBRARY_PATH="/app/lib:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]
