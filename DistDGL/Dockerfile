FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

#     System environment
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

#     System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl git \
        build-essential make cmake \
        ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        python3-dev \
        vim gdb \
        openssh-server net-tools iputils-ping \
    && rm -rf /var/lib/apt/lists/*

#     Miniforge (conda)
RUN wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda init bash \
    && /opt/conda/bin/conda update -n base -c defaults conda -y \
    && /opt/conda/bin/conda clean -afy

#     Conda environment
RUN conda create -n dgl_env python=3.11 pip -y \
    && conda clean -afy

# Use the env's Python/pip for all subsequent RUN commands
ENV PATH=/opt/conda/envs/dgl_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=dgl_env

# ---- Scientific stack FIRST ----
RUN conda install -n dgl_env -y \
        numpy ninja pyyaml \
        mkl mkl-include \
        setuptools cmake cffi \
        jemalloc tqdm future pydot \
        scikit-learn \
    && conda install -n dgl_env -y -c conda-forge \
        gperftools onnx tensorboardx \
        libunwind libpq psycopg \
    && conda clean -afy

# ---- PyTorch 2.3 + CUDA 12.1 (pinned) ----
# Downgraded from 2.4 because DGL 2.5 (which fixes a distributed-sampler bug
# in 2.4) only has wheels for torch 2.3 on cu121. Pin torch<2.4 so later pip
# commands don't silently upgrade it.
RUN pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121

# ---- DGL 2.5.0 for cu121 / torch 2.3 ----
# DGL 2.4.0 has a bug in the distributed neighbor sampler: block destination
# indices are written with values from the src index-space, producing NaN
# losses during distributed training. 2.5.0 resolves this.
RUN pip install --no-cache-dir \
        dgl==2.5.0+cu121 \
        -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html

# ---- Additional pip packages (guard against torch reinstall) ----
RUN pip install --no-cache-dir \
        pandas \
        psycopg_pool \
        nvidia-ml-py \
        "networkx==2.6.3" \
        "urllib3<2.0" \
        "requests<2.32" \
        psutil \
    && pip install --no-cache-dir --no-deps ogb \
    && pip install --no-cache-dir outdated six

# ---- Verification: fail the build if versions drift ----
RUN python - <<'EOF'
import torch, dgl

assert torch.__version__.startswith('2.3.'),    f'torch is {torch.__version__}'
assert dgl.__version__.startswith('2.5.'),      f'dgl is {dgl.__version__}'
assert torch.version.cuda.startswith('12.1'),   f'cuda is {torch.version.cuda}'

print('OK: torch', torch.__version__, 'dgl', dgl.__version__)
EOF

ENV DGLBACKEND=pytorch
ENV device=gpu

# Also persist in .bashrc for interactive shells (debugging)
RUN echo "conda activate dgl_env" >> /root/.bashrc

#----------------------------------------------------------------------------
# DGL source tree
# tools/launch.py is NOT shipped in the pip wheel -- we need it from source.
# Pin a version that matches the pip-installed dgl to avoid API drift.
# Note: DGL tags use a 'v' prefix (v2.4.0), not bare semver.
#----------------------------------------------------------------------------
ENV DGL_GIT_TAG=v2.4.0
RUN git clone --depth 1 --branch ${DGL_GIT_TAG} https://github.com/dmlc/dgl.git /dgl_source \
    && ls /dgl_source/tools/launch.py

# Handy env var so run_dgl_dist.sh can find it
ENV LAUNCH_PY=/dgl_source/tools/launch.py

#----------------------------------------------------------------------------
# SSH server setup (for DGL's launcher to ssh into worker containers)
# The image-internal key pair lets any container from this image ssh into
# any other container from this image. In Swarm, DO NOT mount ~/.ssh over
# /root/.ssh from the host -- it would clobber these keys.
#----------------------------------------------------------------------------
RUN mkdir -p /var/run/sshd /root/.ssh \
    && ssh-keygen -t rsa -f /root/.ssh/id_rsa -N '' \
    && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/authorized_keys \
    && chmod 644 /root/.ssh/id_rsa.pub \
    # harden sshd
    && sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/'           /etc/ssh/sshd_config \
    && sed -i 's/#\?PermitRootLogin.*/PermitRootLogin yes/'                        /etc/ssh/sshd_config \
    && sed -i 's/#\?ChallengeResponseAuthentication.*/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/#\?UsePAM.*/UsePAM no/'                                           /etc/ssh/sshd_config \
    # Let env vars propagate to non-interactive SSH sessions (belt & braces,
    # ENV above already does the heavy lifting)
    && echo "PermitUserEnvironment yes"    >> /etc/ssh/sshd_config \
    && echo "AcceptEnv *"                   >> /etc/ssh/sshd_config \
    && echo "StrictHostKeyChecking no"      >> /etc/ssh/ssh_config \
    # Pre-populate /root/.ssh/environment so every SSH session sees our PATH.
    # Python resolves to /opt/conda/envs/dgl_env/bin/python in launched commands.
    && echo "PATH=/opt/conda/envs/dgl_env/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" > /root/.ssh/environment \
    && echo "DGLBACKEND=pytorch"    >> /root/.ssh/environment \
    && echo "CONDA_DEFAULT_ENV=dgl_env" >> /root/.ssh/environment

#     Working directory
WORKDIR /workspace

#----------------------------------------------------------------------------
# Healthcheck
# Softer than before: only checks that python + dgl import, not CUDA.
# A CUDA-dependent healthcheck fails spuriously when drivers are being
# initialized or when running CPU-only for debugging.
#----------------------------------------------------------------------------
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import dgl, torch" || exit 1

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
