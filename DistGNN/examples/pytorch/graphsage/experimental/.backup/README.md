In the paper(https://arxiv.org/pdf/2308.15602), **"lf" stands for "latency factor"**. It is used as a configuration parameter in the experiments to evaluate the performance of different graph partitioning strategies.

### Context and Usage:

The term appears specifically in **Section 5 (EXPERIMENTAL EVALUATION)**, particularly in **Section 5.4**, which focuses on "Performance with Computation Offloading."

* **Definition**: The authors define $lf$ as a factor that scales the computation time of the GNN layers. It is used to simulate different hardware capabilities or workload intensities.
* **Purpose**: By varying $lf$, the researchers can study how the ratio between communication and computation affects the training time and the effectiveness of different partitioning methods.
* **Example from Text**: The paper states, *"To vary the workload, we use a latency factor $lf$ that scales the compute time of a layer."*

### Impact on Training:

* When $lf$ is low, the system is more sensitive to communication overhead (network latency), making high-quality partitioning (which minimizes edge cuts) more important.
* When $lf$ is high, the system becomes compute-bound, meaning the time spent on calculations outweighs the time spent moving data across the network. In this scenario, the specific partitioning strategy may have less of an impact on overall training speed compared to a low $lf$ scenario.

In summary, "lf" is not a standard type of feature data (like a node attribute), but rather a **control variable** used in the study to simulate various computational loads during the GNN training process.

# Wrapper Script from scratch information(From LLM)

## What a launcher wrapper actually does

Your training script (Python/PyTorch) is just a single process. The wrapper's job is to **set up the environment correctly and then spawn N copies of that process** across cores/nodes so they can communicate with each other. It does this in layers:

---

### Layer 1 — Environment sourcing

```bash
source env.sh
```

`env.sh` is where you store project-specific paths, module loads, virtual environments, and any variables your training script reads. The wrapper sources it first so everything downstream inherits those values. You write `env.sh` yourself — it's just a flat shell file full of `export VAR=value` lines.

---

### Layer 2 — Hardware discovery

Before you can pin threads or assign workers, you need to know what you're running on:

```bash
lscpu   # → cores per socket, sockets, NUMA nodes, hyperthreading
free    # → RAM available
/proc/meminfo, /sys/fs/cgroup/memory.*   # → cgroup-limited RAM (containers)
du -sb  # → dataset size on disk
```

These are all standard Linux tools. The wrapper just parses their output with `awk` and `grep` to get plain numbers.

---

### Layer 3 — Process count resolution

You need to know **how many processes to launch** and **on how many nodes**. There are three sources for this, in priority order:

1. **SLURM** — if you're on a cluster, `$SLURM_NTASKS` and `$SLURM_NNODES` are set automatically
2. **Hostfile** — a plain text file, one hostname per line, listing which machines to use
3. **CLI flags** — `-np 4` passed directly to the wrapper

The wrapper resolves them in that order so the same script works both on a laptop and on a 100-node cluster.

---

### Layer 4 — The process launcher

This is the core of the wrapper. You pass your training command to one of these launchers:

| Launcher | Use case |
|---|---|
| `mpiexec.hydra` (Intel MPI) | CPU training, what your current script uses |
| `torchrun` | PyTorch native, easiest for GPU training |
| `mpirun` (OpenMPI) | General MPI, common on HPC clusters |
| `srun` | SLURM-native, works on any SLURM cluster |

They all do the same thing: spawn N copies of your command and inject environment variables so each process knows its rank (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`) and where the master process is (`MASTER_ADDR`, `MASTER_PORT`). Your Python script reads those to set up `torch.distributed`.

---

### Layer 5 — Thread and memory tuning

PyTorch uses two thread pools under the hood:

- **OpenMP** (`OMP_NUM_THREADS`) — controls how many CPU threads each process uses for tensor ops
- **CCL workers** (`CCL_WORKER_COUNT`) — communication threads for collective ops (AllReduce etc.) when using Intel oneCCL

If you don't set these, PyTorch will spawn as many threads as there are cores, and with N processes you'll massively oversubscribe the CPU. The formula is roughly:

```
OMP_NUM_THREADS = (cores_per_socket - CCL_workers) / processes_per_socket
```

For GPU training you generally set `OMP_NUM_THREADS` much lower (4–8) because the GPU does the heavy lifting.

---

### Layer 6 — CPU affinity

On multi-socket machines, you want each process pinned to its own socket so it doesn't cross NUMA boundaries. This is done via bitmasks:

```bash
I_MPI_PIN_DOMAIN   # Intel MPI: which cores each rank owns
numactl --cpunodebind=0  # alternative: bind a process to a NUMA node
taskset -c 0-15    # alternative: bind to specific core range
```

For GPU training you usually skip this and let CUDA handle device affinity.

---

## What you need to build your own

### Minimal skeleton

```bash
#!/bin/bash
source /your/project/env.sh

# --- Hardware ---
NUM_CORES=$(lscpu | grep "Core(s) per socket" | awk '{print $NF}')
NUM_SOCKETS=$(lscpu | grep "Socket(s):" | awk '{print $NF}')
MEM_GB=$(grep MemTotal /proc/meminfo | awk '{print $2/1024/1024}')

# --- How many processes ---
NP=${1:-1}   # first arg, default 1

# --- Thread tuning ---
export OMP_NUM_THREADS=$(( NUM_CORES / NP ))
if [ $OMP_NUM_THREADS -lt 1 ]; then export OMP_NUM_THREADS=1; fi

# --- Launch ---
export MASTER_ADDR=localhost
export MASTER_PORT=12345
torchrun --nproc_per_node=$NP your_train_script.py
```

That's a working wrapper for single-node CPU or GPU training.

---

### GPU-specific additions

For GPU training with `torchrun` you swap out the launcher and add:

```bash
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

# Let CUDA handle device assignment
export CUDA_VISIBLE_DEVICES=0,1,2,3   # or auto-detect

# Lower CPU threads — GPU does the work
export OMP_NUM_THREADS=4

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  your_train_script.py
```

For **multi-node GPU** you add:
```bash
torchrun \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \    # 0 on master, 1,2,3... on workers
  --nproc_per_node=$NUM_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  your_train_script.py
```

---

### Features you can add incrementally

| Feature | What it needs |
|---|---|
| GPU memory tuning | `nvidia-smi --query-gpu=memory.total` → same ratio logic as RAM |
| Mixed precision awareness | Flag `--fp16` / `--bf16` → reduce memory reserve (less RAM per activation) |
| Gradient checkpointing | Flag → reduce memory pressure tier by one level |
| Automatic batch size scaling | `suggested_batch = headroom_gb * 1024 / bytes_per_sample` |
| NUMA binding | `numactl --cpunodebind=$SOCKET --membind=$SOCKET` wrapping each rank |
| Profiling mode | `--profile` flag → set `TORCH_CPP_LOG_LEVEL`, launch with `perf` or `vtune` |
| Fault tolerance | Wrap `torchrun` with `--max_restarts=3` |

---

### The one thing that must be in your Python training script

The wrapper only works if your training script calls `torch.distributed.init_process_group()` and reads the injected env vars:

```python
import torch.distributed as dist
import os

dist.init_process_group(backend="nccl")   # GPU
# or
dist.init_process_group(backend="gloo")   # CPU

local_rank = int(os.environ["LOCAL_RANK"])
```

The launcher injects `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` automatically — your script just reads them.

---

### Recommended backends to know

| Backend | Use case |
|---|---|
| `nccl` | GPU-to-GPU, fastest for NVIDIA |
| `gloo` | CPU or fallback GPU, works everywhere |
| `mpi` | HPC clusters, requires MPI install |
| `ccl` (Intel) | CPU on Intel hardware, what your current script uses |

Start with `torchrun` + `nccl` for GPU work — it's the simplest path and well documented by PyTorch directly.
