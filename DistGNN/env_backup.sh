#!/bin/bash

# Intel MPI Umgebung aktivieren – stellt mpiexec.hydra bereit
source /opt/intel/oneapi/mpi/latest/env/vars.sh

# OneCCL Umgebung
export CCL_ROOT=/opt/oneccl
export LD_LIBRARY_PATH=/opt/oneccl/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/opt/oneccl/bin:$PATH

# Conda Umgebung aktivieren
source sub407/miniconda3/bin/activate sub407

# Torch-CCL aktivieren
torch_ccl_path=$(python -c "import torch_ccl, os; print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
source "$torch_ccl_path/env/setvars.sh"

# Thread/CCL Konfiguration
NUM_THREADS=$(lscpu | grep "Core(s) per socket" | awk '{print $NF}')
export OMP_NUM_THREADS=$NUM_THREADS
export KMP_AFFINITY=compact,1,granularity=fine,verbose
export KMP_BLOCKTIME=1
export I_MPI_PIN_DOMAIN=core
export CCL_WORKER_COUNT=2
export CCL_WORKER_AFFINITY=0,1

# Conda-spezifische Bibliotheken
#export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so
#export LD_PRELOAD=$LD_PRELOAD:$(find $CONDA_PREFIX/lib -name "libmkl_rt.so" | head -n 1)

# jemalloc-Pfad ermitteln (robust, funktioniert egal woher es kommt)
JEMALLOC_LIB="$(find /usr /opt ${CONDA_PREFIX:-/nonexistent} -name 'libjemalloc.so*' 2>/dev/null | head -n 1)"

if [ -z "$JEMALLOC_LIB" ]; then
    echo "WARN: jemalloc not found, falling back to glibc malloc" >&2
else
    echo "INFO: using jemalloc from $JEMALLOC_LIB" >&2
    export LD_PRELOAD="$JEMALLOC_LIB"
fi

# THEN append libiomp5 and libmkl_rt (order: jemalloc first!)
export LD_PRELOAD="$LD_PRELOAD:${CONDA_PREFIX}/lib/libiomp5.so"
export LD_PRELOAD="$LD_PRELOAD:$(find $CONDA_PREFIX/lib -name 'libmkl_rt.so' | head -n 1)"

# jemalloc tuning for CPU training workloads
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:5000,muzzy_decay_ms:5000,narenas:4"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export PYTHONPATH=$PYTHONPATH:/workspace/dgl_XEON/examples/pytorch/graphsage/
