#!/bin/bash
# =============================================================================
#  run_dist.sh  —  MPI launcher for DistGNN in containers with cpuset
#
#  Self-contained: no separate rank_wrapper.sh required. The script calls
#  itself via '--rank-exec' so that CPU pinning is calculated LOCALLY inside
#  each container. This is necessary because the containers have different
#  cpusets (nodeA: 4-25,52-73 / nodeB: 26-47,74-95) — centrally calculated
#  pin lists would be wrong for half of the ranks.
#
#  Usage:
#    bash experimental/run_dist.sh \
#         -f /root/host_config.txt \
#         -n 8 [-ppn 1] \
#         python experimental/train_dist_sym.py --dataset ... 2>&1
#
#  Environment variables:
#    CCL_WORKER_COUNT=2    dedicated communication cores per rank
#    MASTER_PORT=29500     fixed port
#    NET_IFACE=eth0        overlay interface
#    ALLOCATOR=jemalloc    jemalloc | glibc
#    LOG_DIR=/workspace/logs/run1   per-rank log files instead of console output
#    ENV_SH=/workspace/env.sh
# =============================================================================

set -uo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ENV_SH="${ENV_SH:-/workspace/env.sh}"
NET_IFACE="${NET_IFACE:-eth0}"
CCL_WORKER_COUNT="${CCL_WORKER_COUNT:-2}"

# =============================================================================
#  Shared helpers
# =============================================================================

# "4-6,9" -> 4 5 6 9   (one ID per line)
expand_cpu_list() {
	local spec=$1 part start end c
	local -a parts=()
	IFS=',' read -ra parts <<<"$spec"
	for part in "${parts[@]}"; do
		if [[ $part == *-* ]]; then
			start=${part%%-*}
			end=${part##*-}
			for ((c = start; c <= end; c++)); do echo "$c"; done
		else
			echo "$part"
		fi
	done
}

# One thread per physical core (remove SMT siblings)
physical_cores_in_cpuset() {
	local cal c sf sib key
	local -a allowed=() phys=()
	declare -A seen=()
	cal=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
	[[ -z $cal ]] && return 1
	mapfile -t allowed < <(expand_cpu_list "$cal" | sort -n)
	for c in "${allowed[@]}"; do
		sf="/sys/devices/system/cpu/cpu${c}/topology/thread_siblings_list"
		if [[ -r $sf ]]; then sib=$(<"$sf"); else sib=$c; fi
		key=${sib%%,*}
		key=${key%%-*}
		if [[ -z ${seen[$key]:-} ]]; then
			seen[$key]=1
			phys+=("$c")
		fi
	done
	printf '%s\n' "${phys[@]}"
}

load_env_sh() {
	[[ -n ${GNN_ENV_LOADED:-} ]] && return 0
	[[ -f $ENV_SH ]] || {
		echo "error: $ENV_SH not found" >&2
		exit 1
	}
	local dir
	dir=$(dirname "$ENV_SH")
	set +u
	pushd "$dir" >/dev/null || exit 1
	# shellcheck disable=SC1090
	source "$ENV_SH"
	popd >/dev/null || exit 1
}

# =============================================================================
#  MODE 1: Rank mode  (called by mpiexec.hydra for each rank)
# =============================================================================
if [[ "${1:-}" == "--rank-exec" ]]; then
	shift

	# The toolchain is normally inherited via -genvall; if not, load it here.
	# GNN_ENV_LOADED prevents sourcing it twice.
	if ! command -v python >/dev/null 2>&1; then
		load_env_sh
	fi

	mapfile -t PHYS < <(physical_cores_in_cpuset) || {
		echo "[rank] ERROR: Unable to read CPU affinity mask" >&2
		exit 1
	}

	n_phys=${#PHYS[@]}
	n_ccl=${CCL_WORKER_COUNT:-2}
	n_app=$((n_phys - n_ccl))

	if ((n_app < 1)); then
		echo "[rank] ERROR: $n_phys cores, $n_ccl CCL workers -> not enough cores" >&2
		exit 1
	fi

	APP=("${PHYS[@]:0:$n_app}")
	CCL=("${PHYS[@]:$n_app:$n_ccl}")

	# --- Pin OpenMP exactly ---------------------------------------------
	# Explicit list instead of OMP_PLACES=cores: 'cores' is numbered relative
	# to the visible system, not to the container's cpuset.
	places=""
	for c in "${APP[@]}"; do places+="{${c}},"; done
	#  export OMP_PLACES="${places%,}"
	#  export OMP_PROC_BIND=close
	export OMP_NUM_THREADS=$n_app
	export MKL_NUM_THREADS=$n_app
	export NUMEXPR_NUM_THREADS=$n_app

	# KMP_AFFINITY MUST be removed: any value (including 'disabled') overrides
	# the OMP_PLACES/OMP_PROC_BIND control of libiomp5.
	unset KMP_AFFINITY
	unset KMP_HW_SUBSET

	# --- Pin oneCCL workers to the reserved cores ------------------------
	export CCL_WORKER_COUNT=$n_ccl
	((n_ccl > 0)) && export CCL_WORKER_AFFINITY="$(
		IFS=,
		echo "${CCL[*]}"
	)"

	# --- Disable MPI pinning ---------------------------------------------
	# With one rank per container, the cpuset is already the domain.
	export I_MPI_PIN=0
	unset I_MPI_PIN_DOMAIN
	unset I_MPI_PIN_PROCESSOR_LIST

	# Tie jemalloc arenas to the actual number of threads
	if [[ -n ${MALLOC_CONF:-} ]]; then
		export MALLOC_CONF="$(echo "$MALLOC_CONF" | sed -E "s/narenas:[0-9]+/narenas:${n_app}/")"
	fi

	printf '[rank] %-10s %-14s phys=%d app=%s..%s ccl=%s OMP=%d\n' \
		"$(hostname)" "$(hostname -I 2>/dev/null | awk '{print $1}')" \
		"$n_phys" "${APP[0]}" "${APP[-1]}" "${CCL_WORKER_AFFINITY:-none}" "$n_app" >&2

	# Generate the rank prefix locally (instead of 'mpiexec -l') so the Hydra
	# label demultiplexer is not modified.
	rank="${PMI_RANK:-${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-?}}}"
	exec > >(stdbuf -oL sed -u "s/^/[$rank] /") 2>&1

	exec "$@"
fi

# =============================================================================
#  MODE 2: Launcher  (normal invocation)
# =============================================================================

MASTER_PORT="${MASTER_PORT:-29500}"
ALLOCATOR="${ALLOCATOR:-jemalloc}"

load_env_sh
export DGLBACKEND=pytorch
#export KMP_BLOCKTIME=0
export OMP_WAIT_POLICY=passive

# ----------------------------------------------------------------- Args
args_ntasks=
args_ppn=
args_hostfile=

while (($#)); do
	case "$1" in
	-n | -np)
		args_ntasks=$2
		shift 2
		;;
	-ppn)
		args_ppn=$2
		shift 2
		;;
	-f)
		args_hostfile=$2
		shift 2
		;;
	--)
		shift
		break
		;;
	-*)
		echo "error: unknown option $1" >&2
		exit 1
		;;
	*) break ;;
	esac
done

if (($# == 0)); then
	echo "usage: $0 -f <hostfile> -n <ntasks> [-ppn <ranks/host>] <cmd> [args...]" >&2
	exit 1
fi

cmd=$1
shift
cmd_args=("$@")

# ----------------------------------------------------------------- Hostfile
np="${args_ntasks:-1}"
nnodes=1
opt_hostfile=()

if [[ -n $args_hostfile ]]; then
	[[ -f $args_hostfile ]] || {
		echo "error: hostfile '$args_hostfile' missing" >&2
		exit 1
	}
	# DO NOT use 'sort -u': every line is a separate MPI endpoint.
	nnodes=$(grep -cvE '^[[:space:]]*(#|$)' "$args_hostfile")
	((nnodes > 0)) || {
		echo "error: hostfile is empty" >&2
		exit 1
	}
	opt_hostfile=(-f "$args_hostfile")
fi

ppn="${args_ppn:-1}"

expected=$((nnodes * ppn))
if ((expected != np)); then
	cat >&2 <<EOF
ERROR: Rank calculation does not match.
  hostfile lines : $nnodes   ($args_hostfile)
  -ppn           : $ppn
  -n             : $np
  expected       : $nnodes x $ppn = $expected

With one container per hostfile line, -ppn must be 1.
For 8 partitions: 8 lines (container IPs) + -n 8 + -ppn 1.
EOF
	exit 1
fi

echo "=== $np ranks on $nnodes endpoints ($ppn rank/endpoint) ==="

# ----------------------------------------------------------------- Sanity
echo "cpuset (launcher container): $(awk '/^Cpus_allowed_list:/{print $2}' /proc/self/status)"

shm_mb=$(df -m --output=size /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
if [[ -n $shm_mb ]] && ((shm_mb < 51200)); then
	echo "WARNING: /dev/shm is only ${shm_mb} MB — start containers with shm_size: 200gb." >&2
fi

if [[ -r /sys/fs/cgroup/memory.max ]]; then
	echo "cgroup memory.max: $(cat /sys/fs/cgroup/memory.max)"
	oom=$(awk '/^oom_kill /{print $2}' /sys/fs/cgroup/memory.events 2>/dev/null)
	[[ -n ${oom:-} ]] && echo "cgroup oom_kill so far: $oom"
fi

# ----------------------------------------------------------------- Master
if [[ -n $args_hostfile ]]; then
	MASTER_ADDR=$(grep -vE '^[[:space:]]*(#|$)' "$args_hostfile" | head -n1 | cut -d: -f1 | tr -d '[:space:]')
else
	MASTER_ADDR=$(hostname -I | awk '{print $1}')
fi
export MASTER_ADDR MASTER_PORT
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# ----------------------------------------------------------------- Allocator
# jemalloc (via LD_PRELOAD from env.sh) ignores the glibc MALLOC_* variables.
# Enable only one of the two configurations.
alloc_env=()
if [[ $ALLOCATOR == glibc ]]; then
	export LD_PRELOAD="$(echo "${LD_PRELOAD:-}" | tr ':' '\n' | grep -v jemalloc | paste -sd: -)"
	unset MALLOC_CONF
	alloc_env=(
		-genv MALLOC_ARENA_MAX=2
		-genv MALLOC_TRIM_THRESHOLD_=131072
		-genv MALLOC_MMAP_THRESHOLD_=131072
		-genv MALLOC_TOP_PAD_=131072
		-genv MALLOC_MMAP_MAX_=65536
	)
	echo "Allocator: glibc"
else
	echo "Allocator: jemalloc"
fi

# ----------------------------------------------------------------- Logging
log_env=()
if [[ -n ${LOG_DIR:-} ]]; then
	mkdir -p "$LOG_DIR"
	log_env=(-outfile-pattern "$LOG_DIR/out.%r.log" -errfile-pattern "$LOG_DIR/err.%r.log")
	echo "Logs: $LOG_DIR/{out,err}.<rank>.log"
fi
# Do not use '-l': label mode routes every line through the Hydra demuxer and
# is the code path where stdoe_cb() asserts when ranks terminate abruptly.

# ----------------------------------------------------------------- MPI args
mpiexe_args=(
	-np "$np"
	-ppn "$ppn"
	"${opt_hostfile[@]}"
	-wdir "$PWD"
	"${log_env[@]}"

	-genv MASTER_ADDR="$MASTER_ADDR"
	-genv MASTER_PORT="$MASTER_PORT"

	# Only the setting is passed here; the actual CPU IDs are calculated locally
	# by rank mode.
	-genv CCL_WORKER_COUNT="$CCL_WORKER_COUNT"

	# Containers on the overlay have eth0 (overlay) AND eth1 (docker_gwbridge).
	# Without these variables, libfabric often selects eth1 -> the job hangs.
	-genv FI_PROVIDER=tcp
	-genv FI_TCP_IFACE="$NET_IFACE"
	-genv I_MPI_OFI_PROVIDER=tcp
	-genv I_MPI_FABRICS=shm:ofi
	-genv I_MPI_HYDRA_IFACE="$NET_IFACE"
	-genv I_MPI_PIN="$I_MPI_PIN"
	-genv GLOO_SOCKET_IFNAME="$NET_IFACE"

	-genv OMP_WAIT_POLICY=passive
	-genv KMP_BLOCKTIME=0
	-genv PYTHONUNBUFFERED=1
	-genv PYTHONFAULTHANDLER=1

	"${alloc_env[@]}"

	-genv I_MPI_SHM_HEAP_VSIZE=0
	-genv FI_MR_CACHE_MAX_SIZE=0
	-genv CCL_ATL_TRANSPORT=mpi
	-genv CCL_ALLREDUCE=ring

	#  -genv KMP_AFFINITY=verbose,granularity=fine,proclist=[4,5,...,23],explicit
	#  -genv OMP_DISPLAY_ENV=TRUE
)

# ----------------------------------------------------------------- Preflight
echo "--- Preflight: checking reachability of all endpoints ---"
if ! mpiexec.hydra -np "$np" -ppn "$ppn" "${opt_hostfile[@]}" \
	bash -c 'echo "$(hostname) $(hostname -I 2>/dev/null | awk "{print \$1}")"' | sort; then
	echo "ERROR: Preflight failed — check SSH or the overlay." >&2
	exit 1
fi

# ----------------------------------------------------------------- Launch
echo "#### env ####"
for v in CCL_WORKER_COUNT LD_PRELOAD MALLOC_CONF MASTER_ADDR MASTER_PORT \
	PYTHONPATH DGLBACKEND CONDA_PREFIX; do
	[[ -n ${!v:-} ]] && printf 'using %s=%q\n' "$v" "${!v}"
done
echo "#############"

echo "start time: $(date)"
# Use 'bash "$SELF"' instead of executing directly: no execute permission is
# required, and rank mode calculates pinning inside the respective container.
mpiexec.hydra "${mpiexe_args[@]}" bash "$SELF" --rank-exec "$cmd" "${cmd_args[@]}"
rc=$?
echo "end time:   $(date)"
echo "exit code:  $rc"
exit $rc
