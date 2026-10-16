FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# System packages including build tools (for any native dependencies in artifact.yml)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        gcc \
        make \
        pkg-config \
        wget \
        git \
        libgoogle-glog-dev \
        libgflags-dev \
        libboost-all-dev \
        libopenmpi-dev \
        openmpi-bin \
        libsparsehash-dev \
        libgomp1 \
        libgoogle-glog0v5 \
        libgflags2.2 \
        libboost-mpi1.74.0 \
        libopenmpi3 && \
    rm -rf /var/lib/apt/lists/*

# Miniconda installieren
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app

RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# --- Environment 1: "partitioner" with the old DGL version from artifact.yml ---
COPY artifact.yml .
RUN conda env create -f artifact.yml -n partitioner && \
    conda clean -afy


RUN conda init
# --- Environment 2: "modern" with the current DGL version for general use ---
RUN conda create -y -n modern python=3.11 && \
    conda run -n modern pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu118 && \
    conda run -n modern pip install --no-cache-dir \
        dgl -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html && \
    conda run -n modern pip install --no-cache-dir \
        pandas numpy scipy networkx ogb PyYAML pydantic && \
    conda clean -afy


# The interactive shell activates the MODERN environment by default
RUN echo "conda activate modern" >> ~/.bashrc

# Restliche Projektdateien kopieren
COPY SDT_GNN/ ./SDT_GNN/
COPY run_partitioner.py /app/run_partitioner.py


# Wrapper script to ensure the partitioner code uses the old environment,
# without having to switch manually each time
RUN printf '#!/bin/bash\nexec conda run --no-capture-output -n partitioner python "$@"\n' \
        > /usr/local/bin/partitioner-python && \
    chmod +x /usr/local/bin/partitioner-python

# RUN/CMD layers use the modern environment by default
SHELL ["conda", "run", "--no-capture-output", "-n", "modern", "/bin/bash", "-c"]

CMD ["/bin/bash", "-l"]
