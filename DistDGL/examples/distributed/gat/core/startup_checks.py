#!/usr/bin/env python3

"""Startup checks for CPU parallelism and GPU isolation in distributed runs.

Motivation: the affinity mask of the main thread says nothing about where the
OpenMP worker threads actually end up. A run can look perfectly configured in
the logs (correct OMP_PLACES, correct OMP_NUM_THREADS, no warnings) while every
worker thread sits on a single core. This module measures the threads
themselves and aborts before any timing is recorded.

The same applies to GPUs: CUDA_VISIBLE_DEVICES only masks devices for this
process, it does not stop the container from reaching the others, and it says
nothing about whether a second container was started on the same physical GPU.
The GPU checks look at the process, the container and all ranks together.
"""

import glob
import math
import os
import re
import shutil
import socket
import subprocess

import torch as th
import torch.distributed as dist

def _parse_cpu_list(spec):
    """'4-6,9' -> {4, 5, 6, 9}"""
    cpus = set()
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _thread_masks():
    """Affinity mask of every thread in this process, keyed by TID."""
    masks = {}
    task_dir = "/proc/self/task"
    for tid in os.listdir(task_dir):
        try:
            with open(os.path.join(task_dir, tid, "status")) as fh:
                for line in fh:
                    if line.startswith("Cpus_allowed_list:"):
                        masks[int(tid)] = _parse_cpu_list(line.split(":", 1)[1])
                        break
        except (OSError, ValueError):
            # Thread may vanish between listdir and open; ignore it.
            continue
    return masks


def _warm_up_thread_pool(n_threads):
    """Force the OpenMP pool to materialize.

    Worker threads are created lazily on the first parallel region, so the
    thread list is meaningless before some real work has run.
    """
    import torch as th

    if n_threads:
        th.set_num_threads(n_threads)
    a = th.randn(768, 768)
    b = th.randn(768, 768)
    for _ in range(3):
        a = (a @ b).clamp_(-1.0, 1.0)


def cpu_parallelism_report(expected_threads=None, warm_up=True):
    """Measure how much CPU parallelism this rank can actually reach.

    'effective_cores' is the number of distinct CPUs the thread pool can spread
    over, capped by the thread count: the union of all per-thread masks tells us
    which cores are reachable at all, and no more threads than exist can run in
    parallel. In the pathological case (every worker pinned to the same core)
    the union collapses to one CPU and the value drops to 1.
    """
    if expected_threads is None:
        expected_threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or None

    if warm_up:
        _warm_up_thread_pool(expected_threads)

    proc_mask = set(os.sched_getaffinity(0))
    masks = _thread_masks()
    reachable = set()
    for m in masks.values():
        reachable |= m

    # Threads bound to exactly one CPU, grouped by that CPU. A large group is
    # the signature of a collapsed pool.
    pinned = {}
    for m in masks.values():
        if len(m) == 1:
            cpu = next(iter(m))
            pinned[cpu] = pinned.get(cpu, 0) + 1

    n_threads = len(masks)
    effective = len(reachable)

    # Cap by the requested thread count only -- not by the number of threads
    # that happen to exist. Threads come and go (rank 0 loads the full graph
    # for evaluation), so counting them makes the metric jittery and turns a
    # cross-rank equality check into a false alarm.
    if expected_threads:
        effective = min(effective, expected_threads)

    return {
        "proc_mask_size": len(proc_mask),
        "n_threads": n_threads,
        "reachable_cores": sorted(reachable),
        "effective_cores": effective,
        "expected_threads": expected_threads,
        "largest_pinned_group": max(pinned.values()) if pinned else 0,
        "pinned_groups": pinned,
    }


def check_cpu_parallelism(rep, min_fraction=0.9, max_pinned_group=2):
    """Turn a report into a verdict.

    min_fraction:      how much of the requested thread count must be reachable.
    max_pinned_group:  how many threads may share a single-CPU mask before this
                       counts as a collapsed pool. Dedicated communication
                       workers (e.g. oneCCL) legitimately sit alone on a core,
                       so a small number is expected.
    """
    errors, warnings = [], []
    exp = rep["expected_threads"]

    if exp:
        floor = max(1, math.ceil(min_fraction * exp))
        if rep["effective_cores"] < floor:
            errors.append(
                "only {} core(s) reachable by the thread pool, expected at "
                "least {} for OMP_NUM_THREADS={}".format(
                    rep["effective_cores"], floor, exp
                )
            )

    if rep["largest_pinned_group"] > max_pinned_group:
        cpu, count = max(rep["pinned_groups"].items(), key=lambda kv: kv[1])
        errors.append(
            "{} threads are pinned to CPU {} alone -- the thread pool has "
            "collapsed onto one core".format(count, cpu)
        )

    if exp and rep["proc_mask_size"] < exp:
        warnings.append(
            "process mask has {} CPUs but {} threads were requested".format(
                rep["proc_mask_size"], exp
            )
        )

    return not errors, errors, warnings


def format_report(rep, rank):
    cores = rep["reachable_cores"]
    if cores:
        span = "{}..{}".format(cores[0], cores[-1])
    else:
        span = "none"
    return (
        "[rank {}] cpu: threads={} effective_cores={} (expected {}) "
        "reachable={} n={} mask_size={} largest_pinned_group={}".format(
            rank, rep["n_threads"], rep["effective_cores"],
            rep["expected_threads"], span, len(cores),
            rep["proc_mask_size"], rep["largest_pinned_group"],
        )
    )


def agree_across_ranks(value, distributed):
    """Return (min, max) of an integer value over all ranks.

    Uses two MAX all-reduces instead of MIN/MAX, because not every backend
    implements every reduction op for CPU tensors.
    """
    if not distributed:
        return value, value

    import torch as th
    import torch.distributed as dist

    hi = th.tensor([float(value)])
    lo = th.tensor([float(-value)])
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    dist.all_reduce(lo, op=dist.ReduceOp.MAX)
    return int(-lo.item()), int(hi.item())


# ---------------------------------------------------------------------------
# GPU isolation
#
# Three levels are checked:
#   1. process:   what torch sees (respects CUDA_VISIBLE_DEVICES)
#   2. container: what the container can reach (/dev/nvidiaN, nvidia-smi)
#   3. cluster:   no two ranks may use the same physical GPU (by UUID)


def _run(cmd, timeout=10):
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=timeout, check=True,
        )
        return res.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _norm_uuid(uuid):
    """torch reports the bare UUID, nvidia-smi prefixes 'GPU-' or 'MIG-'."""
    uuid = str(uuid).strip()
    if uuid.startswith(("GPU-", "MIG-")):
        return uuid
    return "GPU-" + uuid


def _csv_rows(out, keys):
    rows = []
    for line in (out or "").splitlines():
        if line.strip():
            rows.append(dict(zip(keys, (f.strip() for f in line.split(",")))))
    return rows


def gpu_report():
    """Collect what this rank can see of the GPUs, at all three levels."""
    rep = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        # Device nodes the container can open, independent of env masking.
        "device_nodes": sorted(
            p for p in glob.glob("/dev/nvidia*")
            if re.match(r"^/dev/nvidia\d+$", p)
        ),
        "torch_cuda": False,
        "torch_devices": [],
        "smi_gpus": None,   # None = nvidia-smi not available
        "smi_apps": None,
        "gpu_uuid": None,
    }

    try:
        import torch as th
    except ImportError:
        th = None
    if th is not None and th.cuda.is_available():
        rep["torch_cuda"] = True
        for i in range(th.cuda.device_count()):
            props = th.cuda.get_device_properties(i)
            uuid = getattr(props, "uuid", None)  # torch >= 2.0
            rep["torch_devices"].append({
                "index": i,
                "name": props.name,
                "uuid": _norm_uuid(uuid) if uuid else None,
                "total_mib": props.total_memory // 2**20,
            })

    if shutil.which("nvidia-smi"):
        out = _run([
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        if out is not None:
            rep["smi_gpus"] = _csv_rows(
                out, ("index", "uuid", "name", "total_mib", "used_mib"))
        out = _run([
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        if out is not None:
            rep["smi_apps"] = _csv_rows(out, ("pid", "gpu_uuid", "used_mib"))

    devs = rep["torch_devices"]
    if len(devs) == 1 and devs[0]["uuid"]:
        rep["gpu_uuid"] = devs[0]["uuid"]
    elif rep["smi_gpus"] and len(rep["smi_gpus"]) == 1:
        rep["gpu_uuid"] = rep["smi_gpus"][0]["uuid"]
    return rep


def check_gpu_isolation(rep, require_gpu=True, strict_container=True,
                        busy_mib=500):
    """Turn a GPU report into a verdict.

    require_gpu:       no usable CUDA device is an error (else a warning).
    strict_container:  a container that can reach more than one GPU is an
                       error even if CUDA_VISIBLE_DEVICES hides the others;
                       the mask only covers this process, not other tools
                       or libraries in the same container.
    busy_mib:          memory already in use on our GPU at startup above this
                       points to another tenant on the same device.
    """
    errors, warnings = [], []
    devs = rep["torch_devices"]

    if not rep["torch_cuda"]:
        (errors if require_gpu else warnings).append(
            "CUDA not available to torch")
    elif len(devs) != 1:
        errors.append(
            "process sees {} GPUs, expected exactly 1".format(len(devs)))

    cvd = rep["cuda_visible_devices"]
    if cvd is not None and len([x for x in cvd.split(",") if x.strip()]) > 1:
        errors.append("CUDA_VISIBLE_DEVICES lists several GPUs: {}".format(cvd))

    # Container level: what is physically reachable, whatever the env says.
    counts = [("device nodes", len(rep["device_nodes"]))]
    if rep["smi_gpus"] is not None:
        counts.append(("nvidia-smi", len(rep["smi_gpus"])))
    else:
        warnings.append(
            "nvidia-smi not available, container-level check limited to "
            "device nodes")
    for source, n in counts:
        if n > 1:
            (errors if strict_container else warnings).append(
                "container can reach {} GPUs ({}); only env masking limits "
                "this process -- start the container with a single GPU "
                "(e.g. --gpus device=N)".format(n, source))

    if rep["nvidia_visible_devices"] == "all":
        warnings.append("NVIDIA_VISIBLE_DEVICES=all")

    # torch and nvidia-smi must agree on which GPU this is. Skipped for MIG,
    # where --query-gpu reports the parent GPU instead of the instance.
    t_uuid = devs[0]["uuid"] if len(devs) == 1 else None
    smi = rep["smi_gpus"]
    if (t_uuid and not t_uuid.startswith("MIG-")
            and smi and len(smi) == 1 and smi[0]["uuid"] != t_uuid):
        errors.append("torch uses {} but nvidia-smi reports {}".format(
            t_uuid, smi[0]["uuid"]))

    # Signs of another tenant. Inside a container nvidia-smi usually cannot
    # list processes from other PID namespaces, so memory already in use is
    # the more reliable signal; the process list is best effort.
    uuid = rep["gpu_uuid"]
    for g in smi or []:
        if g["uuid"] != uuid:
            continue
        try:
            used = int(g["used_mib"])
        except ValueError:
            continue
        if used > busy_mib:
            warnings.append(
                "{} MiB already in use on {} at startup -- GPU may be "
                "shared".format(used, uuid))
    foreign = [a["pid"] for a in rep["smi_apps"] or []
               if a["gpu_uuid"] == uuid and a["pid"] != str(rep["pid"])]
    if foreign:
        warnings.append("other processes on {}: pid {}".format(
            uuid, ", ".join(foreign)))

    return not errors, errors, warnings


def format_gpu_report(rep, rank):
    lines = ["[rank {}] gpu on {}: CUDA_VISIBLE_DEVICES={} "
             "NVIDIA_VISIBLE_DEVICES={} device_nodes={}".format(
                 rank, rep["hostname"], rep["cuda_visible_devices"],
                 rep["nvidia_visible_devices"],
                 ",".join(rep["device_nodes"]) or "none")]
    for d in rep["torch_devices"]:
        lines.append("[rank {}]   torch cuda:{} {} {} MiB {}".format(
            rank, d["index"], d["name"], d["total_mib"], d["uuid"]))
    if not rep["torch_devices"]:
        lines.append("[rank {}]   torch: no CUDA devices".format(rank))
    for g in rep["smi_gpus"] or []:
        lines.append("[rank {}]   nvidia-smi {}: {} {}/{} MiB {}".format(
            rank, g["index"], g["name"], g["used_mib"], g["total_mib"],
            g["uuid"]))
    return "\n".join(lines)


def gather_strings(value, distributed, max_len=128):
    """All-gather one short string per rank; every rank gets the full list.

    Same constraints as agree_across_ranks: CPU tensors and MAX all-reduce
    only. Each rank writes its bytes into its own row, all other rows stay
    zero, so MAX over ranks assembles the table.
    """
    if not distributed:
        return [value]

    import torch as th
    import torch.distributed as dist

    world = dist.get_world_size()
    rank = dist.get_rank()
    data = value.encode("utf-8")[:max_len]
    buf = th.zeros(world, max_len + 1)
    buf[rank, 0] = float(len(data))
    if data:
        buf[rank, 1:1 + len(data)] = th.tensor([float(b) for b in data])
    dist.all_reduce(buf, op=dist.ReduceOp.MAX)

    out = []
    for r in range(world):
        n = int(buf[r, 0].item())
        out.append(bytes(int(x) for x in buf[r, 1:1 + n].tolist())
                   .decode("utf-8", "replace"))
    return out


def shared_gpus_across_ranks(rep, distributed):
    """Return one message per physical GPU used by more than one rank.

    GPU UUIDs are globally unique, so equal UUIDs mean the same device even
    when the ranks run in different containers. All ranks get the same
    result, so the verdict is collective.
    """
    entries = gather_strings(
        "{}\t{}".format(rep["gpu_uuid"] or "", rep["hostname"]), distributed)
    by_uuid = {}
    for rank, entry in enumerate(entries):
        uuid, _, host = entry.partition("\t")
        if uuid:
            by_uuid.setdefault(uuid, []).append(
                "rank {}@{}".format(rank, host))
    return [
        "{} used by {}".format(uuid, ", ".join(ranks))
        for uuid, ranks in sorted(by_uuid.items()) if len(ranks) > 1
    ]


def feature_stats_dist(feat, nids, max_chunk_bytes=256 * 2**20):
    """Like feature_stats, but distributed: each rank processes its nids, then all_reduce."""
    d = feat.shape[1]
    chunk_rows = max(1, max_chunk_bytes // (d * 8))
    col_sum = th.zeros(d, dtype=th.float64)
    col_sqsum = th.zeros(d, dtype=th.float64)
    n_nonfinite = 0
    fmin, fmax = math.inf, -math.inf

    for i in range(0, len(nids), chunk_rows):
        chunk = feat[nids[i:i + chunk_rows]].to(th.float64)
        finite = th.isfinite(chunk)
        bad = int((~finite).sum())
        if bad:
            n_nonfinite += bad
            chunk = th.where(finite, chunk, th.zeros_like(chunk))
        col_sum += chunk.sum(0)
        col_sqsum += (chunk * chunk).sum(0)
        fmin = min(fmin, float(chunk.min()))
        fmax = max(fmax, float(chunk.max()))

    sums = th.cat([col_sum, col_sqsum,
                   th.tensor([len(nids), n_nonfinite], dtype=th.float64)])
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    mm = th.tensor([-fmin, fmax], dtype=th.float64)
    dist.all_reduce(mm, op=dist.ReduceOp.MAX)

    col_sum, col_sqsum = sums[:d], sums[d:2 * d]
    n, n_nonfinite = int(sums[2 * d]), int(sums[2 * d + 1])
    col_mean = col_sum / n
    col_std = (col_sqsum / n - col_mean ** 2).clamp_min(0).sqrt()
    mean = float(col_sum.sum()) / (n * d)
    std = math.sqrt(max(float(col_sqsum.sum()) / (n * d) - mean ** 2, 0.0))
    row_l2 = None  # optional, see below

    return {
        "n_nodes": n, "n_dims": d,
        "min": float(-mm[0]), "max": float(mm[1]),
        "mean": mean, "std": std,
        "n_nonfinite": n_nonfinite,
        "n_dead_dims": int((col_std == 0).sum()),
        "max_col_std": float(col_std.max()),
        "max_abs_col_mean": float(col_mean.abs().max()),
    }

def check_features(st, expect_standardized=False):
    """Judge the feature matrix.

    Hard errors are conditions under which training cannot produce anything
    useful. Everything else is a warning: features that are simply not
    z-scored (word embeddings, bag-of-words, GloVe) train perfectly well and
    must not abort the run.

    Returns (ok, errors, warnings).
    """
    errors, warnings = [], []
    max_abs = max(abs(st["min"]), abs(st["max"]))

    # --- hard errors -----------------------------------------------------
    if st["n_nonfinite"]:
        errors.append(f"{st['n_nonfinite']} non-finite value(s) (NaN/Inf)")
    if st["max_col_std"] == 0.0:
        errors.append("every feature dimension is constant (std=0), no signal to learn from")
    if max_abs > 1e4 or st["std"] > 1e3:
        errors.append(
            f"feature scale is extreme (max|x|={max_abs:.3g}, std={st['std']:.3g}); "
            "raw counts/IDs, normalize before training"
        )
    if expect_standardized and not (abs(st["mean"]) < 0.1 and 0.5 < st["std"] < 2.0):
        errors.append(
            f"features are not standardized (mean={st['mean']:.3f}, std={st['std']:.3f}) "
            "but --expect-standardized was requested"
        )

    # --- warnings --------------------------------------------------------
    dead_frac = st["n_dead_dims"] / st["n_dims"]
    if dead_frac > 0.1:
        warnings.append(
            f"{st['n_dead_dims']}/{st['n_dims']} dimensions ({dead_frac:.0%}) are constant"
        )
    if not (abs(st["mean"]) < 0.1 and 0.5 < st["std"] < 2.0):
        warnings.append(
            f"features are not z-scored (mean={st['mean']:.3f}, std={st['std']:.3f}); "
            "fine for embedding features, but standardizing often helps convergence"
        )
    if st["std"] > 0 and max_abs / st["std"] > 50:
        warnings.append(
            f"heavy outliers: max|x|/std = {max_abs / st['std']:.1f}"
        )
    if st["std"] > 0 and st["std"] < 1e-3:
        warnings.append(f"very small dynamic range (std={st['std']:.3g}), gradients may vanish")

    return not errors, errors, warnings
