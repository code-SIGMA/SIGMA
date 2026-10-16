FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install required packages and GCC-8
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y gcc-8 g++-8 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80

RUN apt-get -y install git wget nano expect make libomp-dev libopenblas-dev vim \
    openssh-server openssh-client bc gdb

# Intel MPI Runtime via APT installieren
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor \
    | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    > /etc/apt/sources.list.d/oneapi.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-mpi-devel intel-oneapi-mpi && \
    apt-get update && apt-get install -y --no-install-recommends libjemalloc2 && \
    rm -rf /var/lib/apt/lists/*

# Set GCC-8 as default
RUN update-alternatives --set gcc /usr/bin/gcc-8 && \
    update-alternatives --set g++ /usr/bin/g++-8

# Setup SSH
RUN mkdir -p /var/run/sshd ~/.ssh && \
    ssh-keygen -f ~/.ssh/id_rsa -N '' && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 600 ~/.ssh/authorized_keys && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -ri 's/^#?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#ChallengeResponseAuthentication yes/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/UsePAM yes/UsePAM no/' /etc/ssh/sshd_config && \
    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# Intel MPI in PATH setzen
ENV I_MPI_ROOT=/opt/intel/oneapi/mpi/latest
ENV PATH=${I_MPI_ROOT}/bin:$PATH
ENV LD_LIBRARY_PATH=${I_MPI_ROOT}/lib:$LD_LIBRARY_PATH

ENV I_MPI_FABRICS=shm:ofi

# Setup-Skripte kopieren
COPY ./setup_env.sh /workspace/setup_env.sh
COPY ./env.sh /workspace/env.sh
RUN chmod +x ./setup_env.sh && ./setup_env.sh

# Entrypoint for SSH
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 22
CMD ["/entrypoint.sh"]
