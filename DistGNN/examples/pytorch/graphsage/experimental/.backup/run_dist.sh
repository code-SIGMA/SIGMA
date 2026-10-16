#!/bin/bash

#HARDCODED_THREADS=10

#    Environment Setup

if [[ ! -d "/workspace" ]]; then
  echo "error: /workspace directory not found" >&2
  exit 1
fi

pushd "/workspace" > /dev/null || { echo "error: cannot cd to /workspace" >&2; exit 1; }
source env.sh
popd > /dev/null || exit 1

export KMP_BLOCKTIME=0

#    Helpers

print_vars() {
  local var

  for var in \
    ${!CCL@} \
    ${!I_MPI@} \
    ${!KMP_@} \
    ${!OMP_@} \
    ${!MPI@} \
    ${!dlrm_@} \
    ${!pytorch_@} \
    ${!pcl_@} \
    ${!args_@} \
    ${!slurm_@} \
    ${!CUDA@} \
    ${!NCCL@} \
    ${!DGL@} \
    LD_PRELOAD \
    LD_LIBRARY_PATH \
    VIRTUAL_ENV \
    PATH \
    PYTHONPATH \
    MASTER_ADDR \
    MASTER_PORT \
    WORLD_SIZE \
    RANK \
    LOCAL_RANK \
    DGLBACKEND \
    "$@"
  do
    [[ -n "${!var:-}" ]] && printf 'using %s=%q\n' "$var" "${!var}"
  done
}

#    Argument Parsing

args_ntasks=
args_ppn=
args_hostfile=

while (( "$#" )); do
  case "$1" in
    -n | -np) args_ntasks=$2; shift 2 ;;
    -ppn)     args_ppn=$2;    shift 2 ;;
    -f)       args_hostfile=$2; shift 2 ;;
    --)       shift; break ;;
    -*|--*=)  echo "error: unsupported flag $1" >&2; exit 1 ;;
    *)        break ;;
  esac
done

#    Node / Task Configuration

nnodes=1
np=1
prefix=
opt_hostfile=
opt_ppn=

[[ -n "${SLURM_JOB_ID:-}"  ]] && prefix="srun -n 1 "
[[ -n "${SLURM_NNODES:-}"  ]] && nnodes=$SLURM_NNODES
[[ -n "${SLURM_NTASKS:-}"  ]] && np=$SLURM_NTASKS
[[ -n "${args_ntasks:-}"   ]] && np=$args_ntasks

if [[ -n "${args_hostfile:-}" ]]; then
  if [[ ! -f "$args_hostfile" ]]; then
    echo "error: hostfile '$args_hostfile' does not exist" >&2
    exit 1
  fi
  nnodes=$(sort -u "$args_hostfile" | wc -l)
  opt_hostfile="-f $args_hostfile"
  prefix="mpiexec.hydra -np 1 -f $args_hostfile"
fi

if [[ -n "${args_ppn:-}" ]]; then
  opt_ppn="-ppn $args_ppn"
  real_nnodes=$(( (np + args_ppn - 1) / args_ppn ))
  (( real_nnodes < nnodes )) && nnodes=$real_nnodes
fi

echo "running $np tasks on $nnodes nodes"

#    CPU Topology Detection

num_threads=$(    $prefix lscpu | awk '/Core\(s\) per socket/  { print $NF }')
num_sockets=$(    $prefix lscpu | awk '/Socket\(s\):/          { print $NF }')
num_numa_nodes=$( $prefix lscpu | awk '/NUMA node\(s\):/       { print $NF }')
threads_per_core=$($prefix lscpu | awk '/Thread\(s\) per core/ { print $NF }')

physcpu="0-$(( num_threads - 1 ))"

[[ -n "${args_hostfile:-}" && -z "${args_ppn:-}" ]] && opt_ppn="-ppn $num_sockets"

#    CCL Worker Configuration

if [[ "$np" -eq 1 || -n "${dlrm_use_mpi:-}" ]]; then
  export CCL_WORKER_COUNT=1
else
  export CCL_WORKER_COUNT=${CCL_WORKER_COUNT:-2}
fi
export CCL_WORKER_COUNT=1
echo "ccl count: ${CCL_WORKER_COUNT}"

# FIX: compute ranks per node and reserve CCL_WORKER_COUNT cores *per rank*,
#      not once for the whole node.
ranks_per_node=$(( np / nnodes ))
total_ccl_cores=$(( CCL_WORKER_COUNT * ranks_per_node ))

if [[ -n "${HARDCODED_THREADS:-}" ]]; then
    cores_per_rank=$HARDCODED_THREADS
else
    cores_per_rank=$(( (num_threads - total_ccl_cores) / ranks_per_node ))
fi

total_needed_cores=$(( ranks_per_node * (cores_per_rank + CCL_WORKER_COUNT) ))

if (( total_needed_cores > num_threads )); then
  echo "ERROR: Not enough CPU cores!"
  echo "Needed: $total_needed_cores, Available: $num_threads"
  echo "Reduce HARDCODED_THREADS or ranks_per_node"
  exit 1
fi


# Build explicit per-rank application core lists for I_MPI_PIN_PROCESSOR_LIST.
# Each rank's slot: [slot_start .. slot_start + CCL_WORKER_COUNT - 1] = CCL workers
#                   [slot_start + CCL_WORKER_COUNT .. slot_end]        = app (OMP) threads
# This is consistent with CCL_WORKER_AFFINITY and avoids the hex-mask approach,
# which only works correctly for exactly one rank per socket.
app_core_ranges=()
for (( r = 0; r < ranks_per_node; r++ )); do
  slot_start=$(( r * (cores_per_rank + CCL_WORKER_COUNT) ))
  app_start=$(( slot_start + CCL_WORKER_COUNT ))
  app_end=$(( app_start + cores_per_rank - 1 ))
  app_core_ranges+=("${app_start}-${app_end}")
done

export I_MPI_PIN_PROCESSOR_LIST="$(IFS=:; echo "${app_core_ranges[*]}")"
# Disable I_MPI_PIN_DOMAIN so it does not override the processor list.
unset I_MPI_PIN_DOMAIN

# FIX: CCL_WORKER_AFFINITY must list the CCL worker cores (the reserved cores
#      at the start of each rank's slot), NOT the application cores.
#      Layout per rank r (node-local): slot starts at r*(cores_per_rank+CCL_WORKER_COUNT).
#      First CCL_WORKER_COUNT cores of that slot → CCL workers.
#      Remaining cores_per_rank cores → application (OMP) threads.
ccl_worker_affinity=""
for (( r = 0; r < ranks_per_node; r++ )); do
  slot_start=$(( r * (cores_per_rank + CCL_WORKER_COUNT) ))
  for (( p = 0; p < CCL_WORKER_COUNT; p++ )); do
    ccl_worker_affinity+="$(( slot_start + p )),"
  done
done
# Remove trailing comma
ccl_worker_affinity=${ccl_worker_affinity%,}

export CCL_WORKER_AFFINITY=$ccl_worker_affinity
#export I_MPI_PIN_DOMAIN="[$(IFS=,; echo "${masks[*]}")]"

# FIX: OMP_NUM_THREADS should be cores_per_rank (per-rank allocation),
#      not (num_threads - CCL_WORKER_COUNT) which is a node-wide figure.
export OMP_NUM_THREADS=$cores_per_rank
export MKL_NUM_THREADS=$cores_per_rank

echo "ranks_per_node:          $ranks_per_node"
echo "cores_per_rank:          $cores_per_rank"
echo "CCL_WORKER_AFFINITY:     ${CCL_WORKER_AFFINITY}"
echo "I_MPI_PIN_PROCESSOR_LIST:${I_MPI_PIN_PROCESSOR_LIST}"
echo "OMP_NUM_THREADS:         ${OMP_NUM_THREADS}"

#    Environment Summary

which python icc gcc mpicc mpiexec.hydra 2>/dev/null

echo "#### initial env ####"
print_vars
echo "#### initial env ####"

echo "pytorch version: $(python -c "import torch; print(torch.__version__)" 2>/dev/null)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun hostname | sort -u
fi

#    MPI Launch

export DGLBACKEND=pytorch
export MASTER_PORT=12345

# Determine MASTER_ADDR deterministically.
# Priority: explicit hostfile > Slurm allocation > local hostname.
if [[ -n "$args_hostfile" ]]; then
  export MASTER_ADDR=$(head -n 1 "$args_hostfile")
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
else
  export MASTER_ADDR=$(hostname)
fi
echo "MASTER_ADDR=$MASTER_ADDR"

cmd=$1
shift
cmd_args=("$@")

export I_MPI_PIN_PROCESSOR_LIST="$I_MPI_PIN_PROCESSOR_LIST"

mpiexe_args=(
  -np "$np"
  $opt_ppn
  $opt_hostfile
  -l
  -genv MASTER_ADDR="$MASTER_ADDR"
  -genv MASTER_PORT="$MASTER_PORT"
  -genvexcl I_MPI_PIN_DOMAIN
  -genv I_MPI_PIN_PROCESSOR_LIST="$I_MPI_PIN_PROCESSOR_LIST"
  -genv CCL_WORKER_AFFINITY="$CCL_WORKER_AFFINITY"
  -genv CCL_WORKER_COUNT="$CCL_WORKER_COUNT"
  -genv OMP_NUM_THREADS="$OMP_NUM_THREADS"
  -genv MKL_NUM_THREADS="$OMP_NUM_THREADS"
  -genv OMP_WAIT_POLICY=passive
  -genv KMP_BLOCKTIME=0
  -genv OMP_PROC_BIND=close
  -genv OMP_PLACES=cores
  -genv KMP_AFFINITY=granularity=fine,compact,1,0
  # --- Limit RAM usage ---
  # glibc malloc: aggressively release memory instead of caching it on the heap
  -genv MALLOC_ARENA_MAX=2
  -genv MALLOC_TRIM_THRESHOLD_=131072
  -genv MALLOC_MMAP_THRESHOLD_=131072
  -genv MALLOC_TOP_PAD_=131072
  -genv MALLOC_MMAP_MAX_=65536
  # Intel MPI / libfabric: no large shared heaps and no MR cache
  -genv I_MPI_SHM_HEAP_VSIZE=0
  -genv FI_MR_CACHE_MAX_SIZE=0
  # CCL: ring-allreduce uses less memory than rabenseifner and similar algorithms
  -genv CCL_ATL_TRANSPORT=mpi
  -genv CCL_ALLREDUCE=ring
)

echo "running mpiexec.hydra ${mpiexe_args[*]} $cmd ${cmd_args[*]}"

mpiexec.hydra "${mpiexe_args[@]}" hostname | sort

echo "start time: $(date)"
mpiexec.hydra "${mpiexe_args[@]}" "$cmd" "${cmd_args[@]}"
echo "end time:   $(date)"
