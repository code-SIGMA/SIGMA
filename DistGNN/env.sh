#!/bin/bash
# =============================================================================
#  env.sh  —  DistGNN / oneCCL Toolchain Environment
#
#  Purpose: Intel MPI, oneCCL, Conda, torch-ccl, allocator, networking.
#  NOT responsible for: CPU pinning. This is handled by run_dist.sh per rank,
#  because the containers have different CPU sets
#  (nodeA 4-25,52-73 / nodeB 26-47,74-95).
#
#  Usage:  source /workspace/env.sh
#
#  Overridable:
#    CONDA_ROOT=/workspace/sub407/miniconda3
#    CONDA_ENV=sub407
#    DGL_HOME=/workspace/sub407/dgl_XEON
#    NET_IFACE=eth0
# =============================================================================

[ -n "${GNN_ENV_LOADED:-}" ] && return 0

NET_IFACE="${NET_IFACE:-eth0}"

_err() { echo "env.sh ERROR: $*" >&2; }

# =============================================================================
#  1) Find Conda installation
# =============================================================================
# setup_env.sh runs 'mkdir -p sub407 && cd sub407' and then installs into
# ./miniconda3 — i.e. <working-directory>/sub407/miniconda3. Its location
# therefore depends on where setup_env.sh was started. /opt/miniconda3 was
# only a guess and caused the "No such file or directory" error.

CONDA_ENV="${CONDA_ENV:-sub407}"

if [ -z "${CONDA_ROOT:-}" ]; then
    for _cand in \
        /workspace/sub407/miniconda3 \
        /workspace/miniconda3 \
        "$HOME/sub407/miniconda3" \
        /opt/sub407/miniconda3 \
        /opt/miniconda3
    do
        [ -f "$_cand/bin/activate" ] && CONDA_ROOT="$_cand" && break
    done
fi

if [ -z "${CONDA_ROOT:-}" ]; then
    _found=$(find /workspace /opt "$HOME" -maxdepth 5 -type f -path '*/miniconda3/bin/activate' 2>/dev/null | head -n1)
    [ -n "$_found" ] && CONDA_ROOT="${_found%/bin/activate}"
fi

if [ -z "${CONDA_ROOT:-}" ] || [ ! -f "$CONDA_ROOT/bin/activate" ]; then
    _err "Conda not found."
    echo "  Check:  ls -d /workspace/*/miniconda3" >&2
    echo "  Set:    export CONDA_ROOT=/path/to/miniconda3" >&2
    return 1
fi

# =============================================================================
#  2) Find DGL source tree (for PYTHONPATH)
# =============================================================================
# setup_env.sh also clones dgl_XEON into sub407/, next to miniconda3.
# The original env.sh pointed to /workspace/dgl_XEON — one directory too high.

if [ -z "${DGL_HOME:-}" ]; then
    for _cand in \
        "$(dirname "$CONDA_ROOT")/dgl_XEON" \
        /workspace/dgl_XEON \
        /workspace/sub407/dgl_XEON \
        /dgl
    do
        [ -d "$_cand/examples/pytorch/graphsage" ] && DGL_HOME="$_cand" && break
    done
fi

# =============================================================================
#  3) Intel MPI + oneCCL
# =============================================================================
_impi_vars=/opt/intel/oneapi/mpi/latest/env/vars.sh
if [ -f "$_impi_vars" ]; then
    set +u
    source "$_impi_vars"
else
    _err "Intel MPI vars.sh not found ($_impi_vars)"
    return 1
fi

export CCL_ROOT="${CCL_ROOT:-/opt/oneccl}"
export PATH="$CCL_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CCL_ROOT/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# =============================================================================
#  4) Activate Conda
# =============================================================================
if ! source "$CONDA_ROOT/bin/activate" "$CONDA_ENV"; then
    _err "Could not activate conda environment '$CONDA_ENV'"
    echo "  Available environments:" >&2
    ls "$CONDA_ROOT/envs" 2>/dev/null >&2
    return 1
fi

if ! command -v python >/dev/null 2>&1; then
    _err "python is not in PATH after conda activate"
    return 1
fi

# =============================================================================
#  5) torch-ccl
# =============================================================================
torch_ccl_path=$(python -c "import torch_ccl, os; print(os.path.dirname(os.path.abspath(torch_ccl.__file__)))" 2>/dev/null)

if [ -z "$torch_ccl_path" ]; then
    torch_ccl_path=$(python -c "import oneccl_bindings_for_pytorch as t, os; print(os.path.dirname(os.path.abspath(t.__file__)))" 2>/dev/null)
fi

if [ -n "$torch_ccl_path" ] && [ -f "$torch_ccl_path/env/setvars.sh" ]; then
    source "$torch_ccl_path/env/setvars.sh"
else
    _err "torch_ccl cannot be imported or setvars.sh is missing"
    echo "  Test:  python -c 'import torch_ccl'" >&2
    return 1
fi

# =============================================================================
#  6) Threading — fallback values only
# =============================================================================
# The authoritative values are set by run_dist.sh for each rank. This only
# provides defaults in case env.sh is sourced interactively without mpiexec.
#
# Pure Bash counting: 'python3' does not exist in the container (the conda
# environment uses Python 3.7 and the executable is called 'python'), and
# lscpu reports the HOST topology with 48 cores instead of the 22 available
# in the cpuset.

_count_phys_cores() {
    local cal part start end c sib_file sib key n=0
    declare -A seen=()
    cal=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
    [ -z "$cal" ] && { echo 1; return; }
    local oldifs=$IFS
    IFS=','
    for part in $cal; do
        if [ "${part#*-}" != "$part" ]; then
            start=${part%%-*}; end=${part##*-}
        else
            start=$part; end=$part
        fi
        for (( c=start; c<=end; c++ )); do
            sib_file="/sys/devices/system/cpu/cpu${c}/topology/thread_siblings_list"
            if [ -r "$sib_file" ]; then
                sib=$(cut -d, -f1 < "$sib_file"); key=${sib%%-*}
            else
                key=$c
            fi
            if [ -z "${seen[$key]:-}" ]; then seen[$key]=1; n=$((n+1)); fi
        done
    done
    IFS=$oldifs
    echo "$n"
}

N_PHYSICAL=$(_count_phys_cores)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$N_PHYSICAL}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
#export KMP_BLOCKTIME=1
export OMP_WAIT_POLICY=passive
# Do NOT set KMP_AFFINITY: any value (including 'disabled') disables the
# OMP_PLACES/OMP_PROC_BIND control provided by libiomp5, which the wrapper uses.
unset KMP_AFFINITY
unset KMP_HW_SUBSET

# AMD Zen 4 supports AVX-512; otherwise MKL may not select this code path.
# Verify the effect once with MKL_VERBOSE=1.
export MKL_ENABLE_INSTRUCTIONS=AVX512

# =============================================================================
#  7) Networking: force the overlay interface
# =============================================================================
# Containers on the overlay have both eth0 (overlay) AND eth1 (docker_gwbridge).
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-mpi}"
export FI_PROVIDER="${FI_PROVIDER:-tcp}"
export FI_TCP_IFACE="$NET_IFACE"
export I_MPI_OFI_PROVIDER=tcp
export I_MPI_FABRICS=shm:ofi
export I_MPI_HYDRA_IFACE="$NET_IFACE"
export I_MPI_HYDRA_BOOTSTRAP=ssh
export I_MPI_HYDRA_BOOTSTRAP_EXEC_EXTRA_ARGS="-o StrictHostKeyChecking=no"
export GLOO_SOCKET_IFNAME="$NET_IFACE"

# Leave pinning to the wrapper
export I_MPI_PIN=0
unset I_MPI_PIN_DOMAIN
unset I_MPI_PIN_PROCESSOR_LIST

# =============================================================================
#  8) Allocator
# =============================================================================
JEMALLOC_LIB="$(find /usr /opt "$CONDA_PREFIX" -name 'libjemalloc.so*' 2>/dev/null | head -n1)"
_PRELOAD=""
if [ -n "$JEMALLOC_LIB" ]; then
    _PRELOAD="$JEMALLOC_LIB"
else
    echo "env.sh WARN: jemalloc not found, glibc malloc will be used" >&2
fi

#_IOMP="$(find "$CONDA_PREFIX/lib" -name 'libiomp5.so' 2>/dev/null | head -n1)"
# PyTorch ships its own GNU OpenMP runtime (libgomp-<hash>.so.1) inside
# libtorch_cpu.so, and DGL is built with gcc. Preloading libiomp5 puts a
# second OpenMP runtime into the same process: whichever initialises first
# binds the master thread to place[0], the other then sees a single-CPU
# affinity mask and rejects every other proc ID (OMP: Warning #124).
_MKLRT="$(find "$CONDA_PREFIX/lib" -name 'libmkl_rt.so*' 2>/dev/null | head -n1)"
[ -n "$_MKLRT" ] && _PRELOAD="${_PRELOAD:+$_PRELOAD:}$_MKLRT"
[ -n "$_PRELOAD" ] && export LD_PRELOAD="$_PRELOAD"

# Without this, libmkl_rt loads libiomp5 at runtime and reintroduces the
# second runtime through the back door.
export MKL_THREADING_LAYER=GNU

_MKLRT="$(find "$CONDA_PREFIX/lib" -name 'libmkl_rt.so*' 2>/dev/null | head -n1)"
for _lib in "$_IOMP" "$_MKLRT"; do
    [ -n "$_lib" ] && _PRELOAD="${_PRELOAD:+$_PRELOAD:}$_lib"
done
[ -n "$_PRELOAD" ] && export LD_PRELOAD="$_PRELOAD"

export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:5000,muzzy_decay_ms:5000,narenas:${OMP_NUM_THREADS}"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

if [ -n "${DGL_HOME:-}" ]; then
    export PYTHONPATH="${PYTHONPATH:-}:${DGL_HOME}/examples/pytorch/graphsage/"
else
    echo "env.sh WARN: dgl_XEON not found, PYTHONPATH not set" >&2
fi

export DGLBACKEND=pytorch
export GNN_ENV_LOADED=1

# =============================================================================
#  9) Status output
# =============================================================================
cat >&2 <<EOF
--- env.sh loaded: $(hostname) / $(hostname -I 2>/dev/null | awk '{print $1}') ---
  CONDA_ROOT      : $CONDA_ROOT   (env: $CONDA_ENV)
  python          : $(command -v python)  [$(python -V 2>&1)]
  torch           : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '??')
  dgl             : $(python -c 'import dgl; print(dgl.__version__)' 2>/dev/null || echo '??')
  DGL_HOME        : ${DGL_HOME:-<not found>}
  cpuset          : $(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)
  phys. cores     : $N_PHYSICAL
  OMP_NUM_THREADS : $OMP_NUM_THREADS   [fallback — overridden by run_dist.sh]
  Interface       : $NET_IFACE
  LD_PRELOAD      : ${LD_PRELOAD:-<empty>}
------------------------------------------------------------
EOF

unset _cand _found _lib _PRELOAD _IOMP _MKLRT _impi_vars
