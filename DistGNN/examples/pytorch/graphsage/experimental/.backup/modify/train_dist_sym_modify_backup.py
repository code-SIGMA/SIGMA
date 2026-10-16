#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.append("../")

import torch  # noqa: E402

import argparse
import csv
import gc
import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.data import register_data_args
from dgl.data.utils import load_graphs, load_tensors
from dgl.distgnn.drpa_sym import drpa
from dgl.nn.pytorch.conv import SAGEConv

# Shared modules -- ONE definition of the DRPA metadata format and of
# the feature-name aliases for both the vertex-cut converter
# (edge_partition_dgl_format_saver.py) and this script. Both files
# ship next to this one.
from drpa_meta import (  # noqa: E402
    attach_drpa_meta_to_graph,
    validate_leaders_consistent_dist,
)
from graph_prep import resolve_feature_key  # noqa: E402

try:
    import psycopg  # noqa: F401
    import psycopg_pool
    _PSYCOPG_AVAILABLE = True
except Exception:
    _PSYCOPG_AVAILABLE = False

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import torch_ccl  # noqa: F401
except ImportError as e:
    print(f"[startup] torch_ccl not available: {e}", flush=True)

# One interop thread is the usual choice for DDP + many worker threads;
# the BLAS thread count is left to OMP_NUM_THREADS so the launcher can
# tune it per node.
th.set_num_interop_threads(1)

# th.inference_mode was added in PyTorch 1.9; the DGL_XEON fork that this
# script targets often ships with an older PyTorch. Fall back to no_grad.
_inference_mode = getattr(th, "inference_mode", th.no_grad)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
DEBUG = False
_RANK_FOR_LOG: Optional[int] = None


def _log(level: str, msg: str, rank: Optional[int] = None) -> None:
    if level == "DEBUG" and not DEBUG:
        return
    ts = time.strftime("%H:%M:%S")
    rank = rank if rank is not None else _RANK_FOR_LOG
    prefix = f"[{level:<5} {ts}]"
    if rank is not None:
        prefix += f"[rank {rank}]"
    print(f"{prefix} {msg}", flush=True)


def _dbg(msg, rank=None):  _log("DEBUG", msg, rank)
def _info(msg, rank=None): _log("INFO",  msg, rank)
def _warn(msg, rank=None): _log("WARN",  msg, rank)


def _rss_gb() -> float:
    """Current RSS of this process in GB. Cheap; safe to call often."""
    try:
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return -1.0


def _mem(label: str, rank: Optional[int] = None) -> None:
    """Print this rank's RSS + system free RAM. Always on (cheap)."""
    try:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        _info(f"MEM[{label}] rss={_rss_gb():.2f}GB "
              f"sys_avail={vm.available / (1024 ** 3):.1f}GB "
              f"sys_used={vm.percent:.0f}% swap={sw.percent:.0f}%",
              rank=rank)
    except Exception as e:
        _info(f"MEM[{label}] (psutil error: {e})", rank=rank)


# Peak RSS sampler. Discrete _mem() prints only catch the steady state --
# they MISS transient allocations inside a single forward call. The
# sampler runs in a daemon thread and remembers the highest RSS it has
# seen. Call _peak_rss_report(label) to print and reset.
_peak_rss_lock = threading.Lock()
_peak_rss_gb = 0.0
_peak_sampler_started = False


def _peak_rss_sampler(interval_s: float = 0.05):
    global _peak_rss_gb
    proc = psutil.Process()
    while True:
        try:
            rss = proc.memory_info().rss / (1024 ** 3)
            with _peak_rss_lock:
                if rss > _peak_rss_gb:
                    _peak_rss_gb = rss
        except Exception:
            pass
        time.sleep(interval_s)


def start_peak_rss_sampler():
    global _peak_sampler_started
    if _peak_sampler_started:
        return
    threading.Thread(target=_peak_rss_sampler, daemon=True).start()
    _peak_sampler_started = True


def _peak_rss_report(label: str, rank: Optional[int] = None) -> None:
    global _peak_rss_gb
    with _peak_rss_lock:
        peak = _peak_rss_gb
        _peak_rss_gb = 0.0
    _info(f"MEM_PEAK[{label}] peak_rss={peak:.2f}GB "
          f"current_rss={_rss_gb():.2f}GB", rank=rank)


# ---------------------------------------------------------------------------
# DB / env
# ---------------------------------------------------------------------------
run_id: Optional[uuid.UUID] = None
db_config: Optional[Dict[str, str]] = None

if os.path.exists("/root/env.json"):
    try:
        with open("/root/env.json") as f:
            env_json = json.load(f)
        run_id = uuid.UUID(env_json["run_id"])
        db_config = {k: env_json[k] for k in
                     ("db_host", "db_port", "db_name", "db_user", "db_password")}
    except Exception as e:
        print(f"[startup] Failed to load /root/env.json: {e}", flush=True)

_pool: Optional["psycopg_pool.ConnectionPool"] = None


def _get_pool():
    global _pool
    if db_config is None or not _PSYCOPG_AVAILABLE:
        return None
    if _pool is None:
        conninfo = (f"host={db_config['db_host']} port={db_config['db_port']} "
                    f"dbname={db_config['db_name']} user={db_config['db_user']} "
                    f"password={db_config['db_password']}")
        _pool = psycopg_pool.ConnectionPool(conninfo, min_size=1, max_size=4, open=True)
    return _pool


# ---------------------------------------------------------------------------
# Background sampler + DB writer (single threads, bounded queue)
# ---------------------------------------------------------------------------
SAMPLE_INTERVAL_S = 0.5
_metrics_cache: Dict[str, object] = {}
_metrics_lock = threading.Lock()
_sampler_started = False


def _sample_gpu():
    if not _NVML_AVAILABLE:
        return [], [], []
    util, mem_gb, mem_pct = [], [], []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            util.append(float(u.gpu))
            mem_gb.append(m.used / (1024 ** 3))
            mem_pct.append(100.0 * m.used / m.total)
    except Exception:
        pass
    return util, mem_gb, mem_pct


def _sampler_loop(proc_pid):
    proc = psutil.Process(proc_pid) if proc_pid else psutil.Process()
    psutil.cpu_percent(interval=0.5)
    psutil.cpu_percent(interval=None, percpu=True)
    psutil.cpu_times_percent(interval=0.5)
    proc.cpu_percent(interval=0.5)

    # BUGFIX: the previous version reported *cumulative* byte counters
    # under "_mb_s" names. Track previous counters + timestamp and emit
    # true rates.
    prev_disk = psutil.disk_io_counters()
    prev_net = psutil.net_io_counters()
    prev_t = time.time()

    while True:
        try:
            gpu_util, gpu_mem_gb, gpu_mem_pct = _sample_gpu()
            disk = psutil.disk_io_counters()
            net = psutil.net_io_counters()
            now = time.time()
            dt = max(now - prev_t, 1e-6)

            snap = {
                "proc_rss_gb":       proc.memory_info().rss / (1024 ** 3),
                "proc_threads":      proc.num_threads(),
                "proc_cpu_percent":  proc.cpu_percent(interval=None),
                "cpu_total_percent": psutil.cpu_percent(interval=None),
                "cpu_per_core":      psutil.cpu_percent(interval=None, percpu=True),
                "cpu_iowait":        psutil.cpu_times_percent(interval=None).iowait,
                "ram_percent":       psutil.virtual_memory().percent,
                "ram_available_gb":  psutil.virtual_memory().available / (1024 ** 3),
                "swap_percent":      psutil.swap_memory().percent,
                "disk_read_mb_s":    (disk.read_bytes - prev_disk.read_bytes) / dt / (1024 ** 2),
                "disk_write_mb_s":   (disk.write_bytes - prev_disk.write_bytes) / dt / (1024 ** 2),
                "net_recv_mb_s":     (net.bytes_recv - prev_net.bytes_recv) / dt / (1024 ** 2),
                "net_sent_mb_s":     (net.bytes_sent - prev_net.bytes_sent) / dt / (1024 ** 2),
                "gpu_util_percent":  gpu_util,
                "gpu_mem_used_gb":   gpu_mem_gb,
                "gpu_mem_percent":   gpu_mem_pct,
            }
            prev_disk, prev_net, prev_t = disk, net, now
            with _metrics_lock:
                _metrics_cache.update(snap)
        except Exception as e:
            print(f"[sampler] {e}", flush=True)
        time.sleep(SAMPLE_INTERVAL_S)


def start_background_sampler(proc_pid=None):
    global _sampler_started
    if _sampler_started:
        return
    t = threading.Thread(target=_sampler_loop, args=(proc_pid,), daemon=True)
    t.start()
    _sampler_started = True
    time.sleep(SAMPLE_INTERVAL_S * 2)


def _get_cached_metrics():
    with _metrics_lock:
        return dict(_metrics_cache)


# Bounded so a slow DB cannot leak unbounded memory.
_db_queue: "queue.Queue" = queue.Queue(maxsize=2048)
_db_worker_started = False


def _db_worker_loop():
    while True:
        item = _db_queue.get()
        if item is None:
            break
        fn, args = item
        try:
            fn(*args)
        except Exception as e:
            print(f"[db-worker] {fn.__name__}: {e}", flush=True)


def _ensure_db_worker():
    global _db_worker_started
    if _db_worker_started:
        return
    threading.Thread(target=_db_worker_loop, daemon=True).start()
    _db_worker_started = True


def _insert_epoch(db_config, run_id, rank, epoch, train_loss,
                  total_time, forward_time, backward_time, update_time,
                  train_acc, val_acc, test_acc):
    try:
        pool = _get_pool()
        if pool is None:
            return
        with pool.connection() as conn:
            conn.execute("""
                INSERT INTO epoch_metric (
                    run_id, rank, epoch, train_loss,
                    total_time, forward_time, backward_time, update_time,
                    train_accuracy, val_accuracy, test_accuracy
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (run_id, rank, epoch, train_loss,
                  total_time, forward_time, backward_time, update_time,
                  train_acc, val_acc, test_acc))
    except Exception as e:
        print(f"epoch insert failed: {e}", flush=True)


def _insert_system_metric(db_config, run_id, rank, epoch, m):
    try:
        pool = _get_pool()
        if pool is None:
            return
        with pool.connection() as conn:
            conn.execute("""
                INSERT INTO system_metric (
                    run_id, rank, epoch,
                    cpu_total_percent, cpu_iowait, cpu_per_core,
                    ram_percent, ram_available_gb, swap_percent,
                    disk_read_mb_s, disk_write_mb_s,
                    net_recv_mb_s, net_sent_mb_s,
                    proc_rss_gb, proc_threads, proc_cpu_percent,
                    gpu_util_percent, gpu_mem_used_gb, gpu_mem_percent
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                run_id, rank, epoch,
                m.get("cpu_total_percent"), m.get("cpu_iowait"), m.get("cpu_per_core"),
                m.get("ram_percent"), m.get("ram_available_gb"), m.get("swap_percent"),
                m.get("disk_read_mb_s"), m.get("disk_write_mb_s"),
                m.get("net_recv_mb_s"), m.get("net_sent_mb_s"),
                m.get("proc_rss_gb"), m.get("proc_threads"), m.get("proc_cpu_percent"),
                m.get("gpu_util_percent", []),
                m.get("gpu_mem_used_gb", []),
                m.get("gpu_mem_percent", []),
            ))
    except Exception as e:
        print(f"system_metric insert failed: {e}", flush=True)


def _enqueue_db(fn, args):
    try:
        _db_queue.put_nowait((fn, args))
    except queue.Full:
        _warn("DB queue full; dropping metric")


def send_epoch_info_to_database(*args):
    if run_id is None or db_config is None:
        return
    _ensure_db_worker()
    _enqueue_db(_insert_epoch, (db_config, run_id) + tuple(args))


def send_system_metrics_to_database(rank, epoch):
    if run_id is None or db_config is None:
        return
    m = _get_cached_metrics()
    if not m:
        return
    _ensure_db_worker()
    _enqueue_db(_insert_system_metric, (db_config, run_id, rank, epoch, m))


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
_CSV_COLS = ["rank", "epoch", "total_time", "forward_time", "backward_time",
             "update_time", "train_loss", "train_acc", "val_acc", "test_acc"]


def save_epoch_info_in_csv(rank, epoch, total_time, forward_time, backward_time,
                           update_time, train_loss, train_acc, val_acc, test_acc,
                           filename):
    new = not os.path.isfile(filename) or os.path.getsize(filename) == 0
    with open(filename, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(_CSV_COLS)
        w.writerow([rank, epoch, total_time, forward_time, backward_time,
                    update_time, train_loss, train_acc, val_acc, test_acc])


# ---------------------------------------------------------------------------
# Feature-key resolution (converter writes "_N/feat", "_N:_E:_N/...")
# ---------------------------------------------------------------------------
# The alias table and resolve_feature_key() moved to graph_prep.py, shared
# with the download scripts -- which now canonicalise the names ON DISK
# (feat / label / *_mask). The import at the top of this file keeps this
# script working with artefacts prepared by OLDER download scripts; for
# new artefacts the resolver hits the canonical name on its first probe,
# so the fallback is effectively free.


# ---------------------------------------------------------------------------
# Partition feature loader
#   * Pops the source tensor from node_feats so each row is held once.
#   * Allocates the mirror-padded tensor in one shot.
#   * NEW: optionally downcasts float64 features to float32 -- halves the
#     largest tensor in the job and speeds up every SpMM, with no
#     accuracy impact for GraphSAGE-style training.
# ---------------------------------------------------------------------------
def load_partition_feature(graph, node_feats, role: str, n_master: int,
                           rank: int, required: bool = True,
                           downcast_f64: bool = False) -> Optional[Tuple[int, int]]:
    key = resolve_feature_key(node_feats, role)
    if key is None:
        if required:
            raise KeyError(
                f"Could not resolve role '{role}'. Available keys: "
                f"{list(node_feats.keys())}")
        return None

    # pop() releases the original dict reference immediately.
    master_feat = node_feats.pop(key)
    if master_feat.shape[0] != n_master:
        raise RuntimeError(
            f"Feature '{role}' (key '{key}') has {master_feat.shape[0]} rows, "
            f"expected {n_master} masters.")

    if downcast_f64 and master_feat.dtype == th.float64:
        master_feat = master_feat.to(th.float32)
        if rank == 0:
            _info(f"'{role}': downcast float64 -> float32", rank=rank)

    n_total = graph.num_nodes()
    if n_total == n_master:
        graph.ndata[role] = master_feat
    else:
        shape = (n_total,) + tuple(master_feat.shape[1:])
        full = th.empty(shape, dtype=master_feat.dtype)
        full[:n_master].copy_(master_feat)
        full[n_master:].zero_()
        del master_feat
        graph.ndata[role] = full

    if rank == 0:
        t = graph.ndata[role]
        size_mb = t.element_size() * t.nelement() / (1024 ** 2)
        _info(f"loaded '{role}' from key '{key}' shape={tuple(t.shape)} "
              f"dtype={t.dtype} size={size_mb:.1f} MB", rank=rank)
    return n_master, n_total


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 activation, dropout, aggregator_type):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        for _ in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))

    def forward(self, graph, inputs, inf: bool = False):
        h = inputs
        last = len(self.layers) - 1
        for l, layer in enumerate(self.layers):
            h_new = layer(graph, h)
            del h
            h = h_new
            if l != last:
                h = F.relu(h, inplace=True)
                if not inf:
                    h = self.dropout(h)
        return h


def _split_acc(logits_master, labels_master, mask_master) -> Tuple[int, int]:
    if mask_master.numel() == 0 or not bool(mask_master.any()):
        return 0, 0
    sel = mask_master.nonzero(as_tuple=True)[0]
    pred = logits_master[sel].argmax(dim=1)
    correct = int((pred == labels_master[sel]).sum().item())
    return correct, int(sel.numel())


# Cached per (graph, n_master) empty mask so we don't allocate a fresh
# zero-bool tensor on every eval when val/test masks are absent.
_EMPTY_MASK_CACHE: Dict[int, th.Tensor] = {}


def _empty_mask(n: int) -> th.Tensor:
    m = _EMPTY_MASK_CACHE.get(n)
    if m is None:
        m = th.zeros(n, dtype=th.bool)
        _EMPTY_MASK_CACHE[n] = m
    return m


def evaluate_distributed(model, gobj, distributed: bool,
                         rank: int) -> Tuple[float, float, float, float]:
    """
    Returns (train_acc, val_acc, test_acc, fwd_time_s). Each rank returns
    the same numbers (collective all_reduce).
    """
    t0 = time.time()
    model.eval()

    features = gobj.ndata["feat"]
    labels = gobj.ndata["label"]
    inner = gobj.ndata["inner_node"].bool()
    n_master = int(inner.sum().item())

    with _inference_mode():
        logits = model(gobj, features, True)

    # Masters occupy [0, n_master) by construction (converter invariant).
    logits_m = logits[:n_master]
    labels_m = labels[:n_master]
    train_m = gobj.ndata["train_mask"][:n_master].bool()
    val_m   = gobj.ndata.get("val_mask")
    test_m  = gobj.ndata.get("test_mask")
    val_m   = val_m[:n_master].bool()  if val_m  is not None else _empty_mask(n_master)
    test_m  = test_m[:n_master].bool() if test_m is not None else _empty_mask(n_master)

    tc, tt = _split_acc(logits_m, labels_m, train_m)
    vc, vt = _split_acc(logits_m, labels_m, val_m)
    sc, st = _split_acc(logits_m, labels_m, test_m)

    del logits, logits_m
    fwd_time = time.time() - t0

    stats = th.tensor([tc, tt, vc, vt, sc, st], dtype=th.int64)
    if distributed:
        dist.all_reduce(stats)

    def _div(c, t):
        return float(c) / float(t) if t > 0 else 0.0

    train_acc = _div(int(stats[0]), int(stats[1]))
    val_acc   = _div(int(stats[2]), int(stats[3]))
    test_acc  = _div(int(stats[4]), int(stats[5]))
    _info(f"eval: train={train_acc*100:.4f}% val={val_acc*100:.4f}% "
          f"test={test_acc*100:.4f}% (fwd {fwd_time:.3f}s)", rank=rank)
    return train_acc, val_acc, test_acc, fwd_time


def evaluate_full_graph(model, gr_full, rank: int, distributed: bool
                        ) -> Tuple[float, float, float, float]:
    """
    Single-machine eval on rank 0 against the full (non-partitioned)
    graph. Used when --eval-mode=full. Other ranks idle through this
    via barriers (no collective ops here -- gr_full only exists on r0).
    """
    if rank != 0:
        if distributed:
            dist.barrier()
        return 0.0, 0.0, 0.0, 0.0

    t0 = time.time()
    model.eval()
    with _inference_mode():
        logits = model(gr_full, gr_full.ndata["feat"], True)
    fwd_time = time.time() - t0

    def _one(mask_key):
        m = gr_full.ndata.get(mask_key)
        if m is None:
            return 0.0
        m = m.bool()
        if not bool(m.any()):
            return 0.0
        sel = m.nonzero(as_tuple=True)[0]
        pred = logits[sel].argmax(dim=1)
        return float((pred == gr_full.ndata["label"][sel]).sum().item()) / float(sel.numel())

    train_acc = _one("train_mask")
    val_acc = _one("val_mask")
    test_acc = _one("test_mask")
    del logits
    gc.collect()
    _info(f"eval(full): train={train_acc*100:.4f}% val={val_acc*100:.4f}% "
          f"test={test_acc*100:.4f}% (fwd {fwd_time:.3f}s)", rank=rank)

    if distributed:
        dist.barrier()
    return train_acc, val_acc, test_acc, fwd_time


def evaluate(model, gobj, gr_full, eval_mode: str, distributed: bool, rank: int
             ) -> Tuple[float, float, float, float]:
    """Dispatch wrapper. Picks the path based on --eval-mode."""
    if eval_mode == "full":
        # Broadcast rank-0 accs so the caller's per-rank numbers agree.
        result = evaluate_full_graph(model, gr_full, rank, distributed)
        if distributed:
            t = th.tensor(list(result), dtype=th.float64)
            dist.broadcast(t, src=0)
            result = tuple(float(x) for x in t.tolist())
        return result
    return evaluate_distributed(model, gobj, distributed, rank)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def sync_model_state(model, distributed: bool):
    if not distributed:
        return
    with th.no_grad():
        # Coalesce parameters into one flat buffer -> ONE broadcast
        # instead of one collective per tensor. Buffers are broadcast
        # separately (dtypes may differ from params).
        params = [p.data for p in model.parameters()]
        if params:
            flat = th.cat([p.reshape(-1) for p in params])
            dist.broadcast(flat, src=0)
            off = 0
            for p in params:
                n = p.numel()
                p.copy_(flat[off:off + n].view_as(p))
                off += n
        for b in model.buffers():
            dist.broadcast(b.data, src=0)


class GradReducer:
    """
    Coalesced gradient all-reduce.

    The previous implementation issued one dist.all_reduce per parameter
    tensor: for a 3-layer GraphSAGE that is ~8-12 collectives per step,
    each paying full CCL latency. This class flattens all grads into a
    single persistent buffer (allocated once, reused every epoch), does
    ONE all_reduce, divides, and scatters back. On small models the
    latency saving dominates; on any model it removes per-step allocation
    churn.
    """

    def __init__(self, model: nn.Module):
        self._params: List[th.Tensor] = [p for p in model.parameters()
                                         if p.requires_grad]
        self._numels = [p.numel() for p in self._params]
        self._flat: Optional[th.Tensor] = None

    def all_reduce_mean(self, world_size: int) -> None:
        grads = [p.grad for p in self._params if p.grad is not None]
        if not grads:
            return
        total = sum(g.numel() for g in grads)
        if self._flat is None or self._flat.numel() != total or \
           self._flat.dtype != grads[0].dtype:
            self._flat = th.empty(total, dtype=grads[0].dtype)
        off = 0
        for g in grads:
            n = g.numel()
            self._flat[off:off + n].copy_(g.reshape(-1))
            off += n
        dist.all_reduce(self._flat, op=dist.ReduceOp.SUM)
        self._flat /= world_size
        off = 0
        for g in grads:
            n = g.numel()
            g.copy_(self._flat[off:off + n].view_as(g))
            off += n


def avg_grad_across_workers(model, world_size: int):
    """Kept for API compatibility; prefer GradReducer inside the loop."""
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= world_size


# ---------------------------------------------------------------------------
# DRPA metadata
# ---------------------------------------------------------------------------
# Loading, dtype handling and the invariant checks moved to drpa_meta.py,
# which the vertex-cut converter imports as well. The converter now
# finalizes the dtypes, validates every partition, checks cross-partition
# leader agreement and stamps validated=True into drpa_meta.pt (format
# v4). At start-up this wrapper therefore only reads and attaches:
#
#   * v4 files (validated=True): no casts, no asserts, no all_gather --
#     unless --recheck-drpa is passed.
#   * older files (v1-v3, written by the previous converter): cast and
#     validated at start-up exactly as before. Old artefacts keep
#     running without re-conversion; they only miss the start-up saving.
#   * --check-drpa-consistency keeps its meaning (cross-rank all_gather
#     leader check) via drpa_meta.validate_leaders_consistent_dist.


def attach_drpa_metadata(graph, part_metadata, part_config_parent,
                         rank, num_parts, world_size, args):
    if args.nr == -1:
        _info("DRPA disabled (--nr -1); skipping metadata load", rank=rank)
        return None

    node_map = attach_drpa_meta_to_graph(
        graph, part_metadata, part_config_parent, rank, num_parts,
        require_orig_nid=bool(getattr(args, "validate_drpa", False)),
        trust_meta=not getattr(args, "recheck_drpa", False),
        log=lambda m: _info(m, rank=rank),
    )

    if args.check_drpa_consistency:
        gids_key = "_ID" if "_ID" in graph.ndata else dgl.NID
        if gids_key in graph.ndata:
            validate_leaders_consistent_dist(
                graph.dstdata["lf"], graph.ndata[gids_key],
                graph.ndata["inner_node"], rank, world_size, dist)
            if rank == 0 and world_size > 1:
                _info("DRPA global-consistency check passed", rank=rank)
        else:
            _warn("no reshuffled NID; skip global-consistency check",
                  rank=rank)
    return node_map


# ---------------------------------------------------------------------------
# Partition loading (CSC-only)
# ---------------------------------------------------------------------------
def load_partition(part_metadata, part_config_parent, rank):
    pf = part_metadata[f"part-{rank}"]
    for k in ("node_feats", "edge_feats", "part_graph"):
        assert k in pf, f"part-{rank} missing '{k}'"

    t0 = time.time()
    # Edge features: never load -- GraphSAGE does not use them, and on
    # large graphs they can dwarf node features.
    node_feats = load_tensors((part_config_parent / pf["node_feats"]).as_posix())
    graph = load_graphs((part_config_parent / pf["part_graph"]).as_posix())[0][0]

    if rank == 0:
        _info(f"partition graph formats (loaded): {graph.formats()}", rank=rank)

    # CRITICAL: build CSC up-front. SAGEConv with the 'gcn' / 'mean'
    # aggregator calls g.update_all(copy_u, mean), which uses a fused
    # SpMM kernel ONLY when CSC is present. If CSC is missing, DGL falls
    # back to materialising per-edge messages: ~70 GB per rank on a
    # 28M-edge partition with 602-d features.
    t_fmt = time.time()
    try:
        try:
            graph = graph.formats(['csc'])   # CSC only, drop CSR/COO
            if rank == 0:
                _info("graph format set to CSC only", rank=rank)
        except Exception:
            graph.create_formats_()   # fallback if .formats() unavailable
        if rank == 0:
            _info(f"partition graph formats (after create_formats_): "
                  f"{graph.formats()} (took {time.time() - t_fmt:.3f}s)",
                  rank=rank)
    except AttributeError:
        _warn("graph.create_formats_() unavailable in this DGL; first "
              "forward will materialise per-edge messages and likely OOM. "
              "Upgrade DGL or call _CAPI_DGLAsNumBits manually.", rank=rank)

    _info(f"partition loaded in {time.time() - t0:.3f}s "
          f"(N={graph.num_nodes()}, E={graph.num_edges()})", rank=rank)

    if "inner_node" not in graph.ndata:
        raise RuntimeError("Partition has no 'inner_node' -- not a vertex-cut "
                           "partition from edge_partition_dgl_format_saver.py.")
    return graph, node_feats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _make_optimizer(model, args):
    """
    Adam with foreach=True where available (PyTorch >= 1.13): fuses the
    per-parameter update loop into multi-tensor ops. Falls back silently
    on older builds.
    """
    try:
        return th.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay, foreach=True)
    except TypeError:
        return th.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)


def _clone_state_dict(model) -> Dict[str, th.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def run_training(gobj, n_classes: int, num_parts: int, drpa_node_map,
                 rank: int, world_size: int, args, gr_full=None):
    g = gobj
    in_feats = g.ndata["feat"].shape[1]

    if args.gpu >= 0:
        raise ValueError("CPU-only; pass --gpu -1.")

    inner_mask = g.ndata["inner_node"].bool()
    n_master_local = int(inner_mask.sum().item())
    n_total_local = int(inner_mask.numel())

    # Train mask restricted to masters: mirrors are zero-padded inputs;
    # their labels are meaningless and must never enter the loss.
    train_mask_master = g.ndata["train_mask"].bool() & inner_mask
    train_nid = th.where(train_mask_master)[0]

    # Hoist loop-invariant lookups/copies out of the epoch loop:
    #   * features reference (dict lookup per epoch is cheap but free to hoist)
    #   * train labels: g.ndata["label"][train_nid] previously materialised a
    #     fresh index_select copy EVERY epoch. Do it once.
    features = g.ndata["feat"]
    train_labels = g.ndata["label"][train_nid] if train_nid.numel() > 0 else None
    has_train = train_nid.numel() > 0

    _info(f"partition: total={n_total_local} master={n_master_local} "
          f"mirror={n_total_local - n_master_local} "
          f"train(master)={int(train_nid.numel())}", rank=rank)
    _mem("before model build", rank=rank)

    model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers,
                      F.relu, args.dropout, args.aggregator_type)
    sync_model_state(model, args.distributed)
    optimizer = _make_optimizer(model, args)
    reducer = GradReducer(model) if args.distributed else None

    # ----- baseline eval -----
    if not args.skip_initial_eval:
        if rank == 0:
            _info("initial eval", rank=rank)
        _mem("before initial eval", rank=rank)
        _peak_rss_report("reset", rank=rank)
        train_acc0, val_acc0, test_acc0, _ = evaluate(
            model, gobj, gr_full, args.eval_mode, args.distributed, rank)
        _peak_rss_report("during initial eval", rank=rank)
        _mem("after initial eval", rank=rank)
    else:
        train_acc0 = val_acc0 = test_acc0 = 0.0
        if rank == 0:
            _info("initial eval skipped (--skip-initial-eval)", rank=rank)

    best_val = float(val_acc0)
    # Best model is kept IN MEMORY instead of round-tripping through
    # /tmp: the model is tiny (hidden=16 default) and this removes two
    # disk serialisations + deserialisation from the critical path.
    best_state: Optional[Dict[str, th.Tensor]] = None
    if rank == 0:
        best_state = _clone_state_dict(model)
        if args.save_to_csv:
            save_epoch_info_in_csv(rank, 0, 0.0, 0.0, 0.0, 0.0, -1.0,
                                   train_acc0 * 100, val_acc0 * 100,
                                   test_acc0 * 100, args.csv_path)
        if args.send_to_database:
            send_epoch_info_to_database(
                rank, 0, -1.0, 0.0, 0.0, 0.0, 0.0,
                train_acc0 * 100, val_acc0 * 100, test_acc0 * 100)

    if args.distributed:
        dist.barrier()
    _info("training loop start", rank=rank)
    _mem("training loop start", rank=rank)

    total_time_sum = 0.0
    interval = {"forward": 0.0, "backward": 0.0, "update": 0.0,
                "total": 0.0, "loss": 0.0}
    epoch_count = 0

    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()
        logits = model(g, features)

        if not has_train:
            # Keep loss in the graph so backward signatures match.
            loss = logits.sum() * 0.0
        else:
            loss = F.cross_entropy(logits[train_nid], train_labels)

        del logits
        t1 = time.time()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        t2 = time.time()
        loss_val = float(loss.item())
        del loss
        # NOTE: gc.collect() was previously called here EVERY epoch.
        # A full collection walks every live object (the graph structures
        # alone are large) and can cost 100+ ms per epoch. PyTorch frees
        # tensors via refcounting; cyclic garbage is rare in this loop.
        # We now collect only at report intervals (below).

        if args.distributed:
            reducer.all_reduce_mean(args.world_size)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        t_sync = time.time()
        optimizer.step()
        t3 = time.time()

        if epoch == 0:
            _info(f"epoch-0 breakdown: fwd={t1-t0:.4f}s bwd={t2-t1:.4f}s "
                  f"sync={t_sync-t2:.4f}s step={t3-t_sync:.4f}s "
                  f"total={t3-t0:.4f}s", rank=rank)
            _peak_rss_report("epoch 0", rank=rank)

        ftime = t1 - t0; btime = t2 - t1; utime = t3 - t2; ttime = t3 - t0
        total_time_sum += ttime
        epoch_count += 1
        interval["forward"]  += ftime
        interval["backward"] += btime
        interval["update"]   += utime
        interval["total"]    += ttime
        interval["loss"]      = loss_val

        if args.send_to_database:
            send_system_metrics_to_database(rank, epoch)

        # Eval is itself an expensive forward; do it only at the report
        # intervals, and only when we actually need the numbers.
        is_report = epoch % args.epoch_info_interval == 0 or \
                    epoch == args.n_epochs - 1
        if is_report and (args.val or args.send_to_database or args.save_to_csv):
            train_acc, val_acc, test_acc, _ = evaluate(
                model, gobj, gr_full, args.eval_mode, args.distributed, rank)
            gc.collect()
            if rank == 0 and val_acc > best_val:
                best_val = val_acc
                best_state = _clone_state_dict(model)

            if rank == 0:
                avg = lambda x: x / max(epoch_count, 1)
                if args.save_to_csv:
                    save_epoch_info_in_csv(
                        rank, epoch,
                        avg(interval["total"]), avg(interval["forward"]),
                        avg(interval["backward"]), avg(interval["update"]),
                        interval["loss"],
                        train_acc * 100, val_acc * 100, test_acc * 100,
                        args.csv_path)
                if args.send_to_database:
                    send_epoch_info_to_database(
                        rank, epoch, interval["loss"],
                        avg(interval["total"]), avg(interval["forward"]),
                        avg(interval["backward"]), avg(interval["update"]),
                        train_acc * 100, val_acc * 100, test_acc * 100)
                if args.val:
                    print(f"\nEpoch {epoch:05d} | Total {avg(interval['total']):.4f}s | "
                          f"Loss {interval['loss']:.4f} | Train {train_acc*100:.4f} | "
                          f"Val {val_acc*100:.4f} | Test {test_acc*100:.4f}\n",
                          flush=True)
            interval = {"forward": 0.0, "backward": 0.0, "update": 0.0,
                        "total": 0.0, "loss": 0.0}
            epoch_count = 0

        if rank == 0:
            print(f"Epoch {epoch} time: {ttime:.4f}s", flush=True)

    # ----- final eval with best model -----
    if rank == 0 and best_state is not None:
        model.load_state_dict(best_state)
        _info(f"loaded best (val {best_val*100:.4f}%)", rank=rank)
    if args.distributed:
        # broadcast the best weights so every rank evaluates the same model
        sync_model_state(model, args.distributed)

    train_acc, val_acc, test_acc, _ = evaluate(
        model, gobj, gr_full, args.eval_mode, args.distributed, rank)
    if rank == 0:
        # Persist the best-model eval under epoch=-1 so downstream
        # tooling (and the runner's "Epoch -0001" watchdog marker)
        # can detect the post-training final evaluation.
        if args.save_to_csv:
            save_epoch_info_in_csv(
                rank, -1, 0.0, 0.0, 0.0, 0.0, -1.0,
                train_acc * 100, val_acc * 100, test_acc * 100,
                args.csv_path)
        if args.send_to_database:
            send_epoch_info_to_database(
                rank, -1, -1.0, 0.0, 0.0, 0.0, 0.0,
                train_acc * 100, val_acc * 100, test_acc * 100)
        # Explicit marker line picked up by the benchmark watchdog
        # (run_benchmarks_distgnn.sh looks for "Epoch -0001").
        print(f"\nEpoch -0001 | Total 0.0000s | Loss 0.0000 | "
              f"Train {train_acc*100:.4f} | Val {val_acc*100:.4f} | "
              f"Test {test_acc*100:.4f}\n", flush=True)
        print("#############################################################", flush=True)
        print(f"Best model: Train {train_acc*100:.4f} | "
              f"Val {val_acc*100:.4f} | Test {test_acc*100:.4f} | "
              f"Train time: {total_time_sum:.4f}s", flush=True)
        print("#############################################################", flush=True)


# ---------------------------------------------------------------------------
# DRPA forward-equivalence check (--validate-drpa)
# ---------------------------------------------------------------------------
def _gather_variable_rows(local, rank, world_size):
    n = int(local.shape[0])
    w = int(local.shape[1]) if local.dim() > 1 else 1
    counts = [th.zeros(1, dtype=th.int64) for _ in range(world_size)]
    dist.all_gather(counts, th.tensor([n], dtype=th.int64))
    counts = [int(c.item()) for c in counts]
    max_n = max(counts) if counts else 0
    if max_n == 0:
        return [local.new_zeros((0, w)) for _ in range(world_size)]
    padded = local.new_zeros((max_n, w))
    if n > 0:
        padded[:n] = local
    gathered = [local.new_zeros((max_n, w)) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return [gathered[r][:counts[r]] for r in range(world_size)]


def run_drpa_validation(gobj, gr_full, n_classes, num_parts,
                        rank, world_size, args):
    g = gobj
    features = g.ndata["feat"]
    in_feats = features.shape[1]

    if "orig_nid" not in g.ndata:
        raise RuntimeError("--validate-drpa needs orig_nid on the partition. "
                           "Re-run the converter for v2+ metadata.")
    orig_nid = g.ndata["orig_nid"]
    inner_mask = g.ndata["inner_node"].bool()

    if rank == 0:
        print("=" * 64, flush=True)
        print("DRPA CORRECTNESS VALIDATION", flush=True)
        print("=" * 64, flush=True)

    model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers,
                      F.relu, args.dropout, args.aggregator_type)
    model.eval()
    sync_model_state(model, args.distributed)

    t0 = time.time()
    with _inference_mode():
        drpa_out = model(g, features, True)
    _info(f"DRPA forward: {time.time() - t0:.3f}s", rank=rank)

    master_idx = th.where(inner_mask)[0]
    master_emb = drpa_out[master_idx].contiguous().to(th.float32)
    master_onid = orig_nid[master_idx].contiguous().to(th.float32).unsqueeze(1)

    if args.distributed:
        emb_parts = _gather_variable_rows(master_emb, rank, world_size)
        onid_parts = _gather_variable_rows(master_onid, rank, world_size)
    else:
        emb_parts = [master_emb]
        onid_parts = [master_onid]

    passed = True
    if rank == 0:
        assert gr_full is not None, "rank 0 needs the full graph for validation"
        gr_features = gr_full.ndata["feat"]
        with _inference_mode():
            full_out = model(gr_full, gr_features, True).to(th.float32)

        all_emb = th.cat(emb_parts, dim=0)
        all_onid = th.cat(onid_parts, dim=0).squeeze(1).to(th.int64)

        n_full = int(full_out.shape[0])
        if int(all_onid.min()) < 0 or int(all_onid.max()) >= n_full:
            raise RuntimeError(f"orig_nid out of range "
                               f"[{int(all_onid.min())}, {int(all_onid.max())}] "
                               f"vs n_full={n_full}")

        full_aligned = full_out[all_onid]
        diff = (all_emb - full_aligned).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        denom = full_aligned.abs().clamp_min(1e-8)
        max_rel = float((diff / denom).max())
        tol = args.validate_drpa_tol
        passed = max_abs <= tol

        print(f"[validate-drpa] compared {int(all_onid.numel())} masters; "
              f"max_abs={max_abs:.3e} (tol {tol:.1e}) mean_abs={mean_abs:.3e} "
              f"max_rel={max_rel:.3e}", flush=True)
        print(f"[validate-drpa] {'PASS' if passed else 'FAIL'}", flush=True)
        print("=" * 64, flush=True)

    if args.distributed:
        verdict = th.tensor([1 if passed else 0], dtype=th.int32)
        dist.broadcast(verdict, src=0)
        passed = bool(verdict.item() == 1)
    return passed


# ---------------------------------------------------------------------------
# Optional full-graph load (only for --validate-drpa or --eval-mode=full)
# ---------------------------------------------------------------------------
def load_full_graph_for_eval(graph_path, rank, downcast_f64=False):
    if rank != 0:
        return None
    gp = Path(graph_path)
    if not gp.exists():
        raise FileNotFoundError(f"--graph-path missing: {gp}")
    t0 = time.time()
    gr = load_graphs(gp.as_posix())[0][0]
    # Same critical fix as in load_partition: build CSC before any
    # forward, or the eval will materialise per-edge messages on the
    # full graph.
    t_fmt = time.time()
    try:
        try:
            gr = gr.formats(['csc'])
        except Exception:
            gr.create_formats_()
    except AttributeError:
        _warn("full graph: create_formats_ unavailable, eval will be "
              "huge", rank=rank)
    _info(f"full graph loaded in {time.time() - t0:.3f}s "
          f"(N={gr.num_nodes()}, E={gr.num_edges()}, formats={gr.formats()}, "
          f"create_formats={time.time() - t_fmt:.3f}s)", rank=rank)

    # canonicalise feature names
    for role in ("feat", "label", "train_mask", "val_mask", "test_mask"):
        key = resolve_feature_key(gr.ndata, role)
        if key and key != role:
            gr.ndata[role] = gr.ndata[key]
            del gr.ndata[key]
    for m in ("train_mask", "val_mask", "test_mask"):
        if m in gr.ndata and gr.ndata[m].dtype != th.bool:
            gr.ndata[m] = gr.ndata[m].bool()
    # Keep dtype in sync with the (possibly downcast) partition features,
    # so weights trained in fp32 can run on the full graph.
    if downcast_f64 and gr.ndata["feat"].dtype == th.float64:
        gr.ndata["feat"] = gr.ndata["feat"].to(th.float32)
        _info("full graph: feat downcast float64 -> float32", rank=rank)
    return gr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="GraphSAGE DistGNN training "
                                            "(memory- and comm-tuned)")
    register_data_args(p)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=-1, help="must be -1 (CPU)")
    p.add_argument("--nr", type=int, default=1,
                   help="DRPA delay (-1 = disable DRPA)")
    p.add_argument("--val", action="store_true")
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--n-hidden", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--aggregator-type", default="gcn",
                   choices=["gcn", "mean", "pool", "lstm"])
    p.add_argument("--world-size", default=-1, type=int)
    p.add_argument("--rank", default=-1, type=int)
    p.add_argument("--dist-url", default="env://")
    p.add_argument("--dist-backend", default="ccl")
    p.add_argument("--graph-path", type=str, default=None,
                   help="Full (non-partitioned) DGL graph. Optional; only "
                        "needed for --validate-drpa or --eval-mode=full.")
    p.add_argument("--partition-config-path", type=str, required=True)
    p.add_argument("--save-to-csv", action="store_true")
    p.add_argument("--csv-path", default="training_log.csv")
    p.add_argument("--send-to-database", action="store_true")
    p.add_argument("--epoch-info-interval", default=10, type=int)
    p.add_argument("--eval-mode", default="full",
                   choices=["distributed", "full"],
                   help="distributed (default): score own masters per rank, "
                        "all_reduce(correct,total). Memory-frugal; needs no "
                        "full graph. full: legacy path, requires --graph-path "
                        "on rank 0.")
    p.add_argument("--check-drpa-consistency", action="store_true",
                   help="cross-rank check that all copies of a node agree on "
                        "the same leader. O(N log N) tensor ops; off by default.")
    p.add_argument("--recheck-drpa", action="store_true",
                   help="Re-run the per-partition DRPA invariant checks at "
                        "start-up even when drpa_meta.pt was already "
                        "validated by the converter (validated=True in the "
                        "file). Metadata without the flag (format v1-v3) is "
                        "always checked, as before.")
    p.add_argument("--skip-initial-eval", action="store_true",
                   help="Skip the untrained-baseline eval at epoch 0. Saves "
                        "one full forward -- useful when RAM is tight or you "
                        "just want to confirm the training loop survives.")
    p.add_argument("--no-f32-features", action="store_true",
                   help="Keep float64 features as-is instead of downcasting "
                        "to float32 (downcast is the default: halves feature "
                        "RAM and speeds up SpMM).")
    p.add_argument("--validate-drpa", action="store_true")
    p.add_argument("--validate-drpa-tol", type=float, default=1e-4)
    p.add_argument("--debug", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _setup_threading(rank: int) -> None:
    """
    Defensive thread setup to avoid CPU oversubscription when multiple
    ranks share a node. Each rank should get cores/ranks_per_node BLAS
    threads, not cores threads. If OMP_NUM_THREADS is already set (the
    launcher script knows best), we trust it.
    """
    omp_set = os.environ.get("OMP_NUM_THREADS")
    if omp_set:
        try:
            th.set_num_threads(int(omp_set))
        except ValueError:
            pass
        _info(f"OMP_NUM_THREADS={omp_set} (from env); "
              f"torch.get_num_threads={th.get_num_threads()}", rank=rank)
        return

    cores = os.cpu_count() or 1
    local_size_env = (
        os.environ.get("MPI_LOCALNRANKS")
        or os.environ.get("PMI_LOCAL_SIZE")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE")
        or os.environ.get("SLURM_NTASKS_PER_NODE")
    )
    if local_size_env:
        try:
            ranks_per_node = max(1, int(local_size_env))
        except ValueError:
            ranks_per_node = 1
    else:
        ranks_per_node = 1
        _warn("OMP_NUM_THREADS unset and ranks-per-node not detectable "
              "from env. Defaulting to assume 1 rank/node. If multiple "
              "ranks share a node, set OMP_NUM_THREADS in the launcher "
              "or you will see severe CPU oversubscription.", rank=rank)

    threads = max(1, cores // ranks_per_node)
    th.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    _info(f"threading: cpu_count={cores} ranks_per_node={ranks_per_node} "
          f"-> torch.set_num_threads({threads})", rank=rank)


def main():
    global DEBUG, _RANK_FOR_LOG
    th.set_printoptions(threshold=10)
    args = build_parser().parse_args()
    DEBUG = args.debug
    print(args, flush=True)

    # resolve rank/world-size
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("PMI_SIZE",
                                             os.environ.get("WORLD_SIZE", 1)))
    args.distributed = args.world_size > 1
    if args.world_size > 1:
        args.rank = int(os.environ.get("PMI_RANK",
                                       os.environ.get("RANK", 0)))
    else:
        args.rank = 0

    # DRPA's drpa_init_buckets does dist.all_to_all even for world_size==1.
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    _RANK_FOR_LOG = args.rank
    print(f"Rank: {args.rank}  World_size: {args.world_size}", flush=True)
    _setup_threading(args.rank)
    start_peak_rss_sampler()
    _mem("after init_process_group", rank=args.rank)

    if args.send_to_database:
        start_background_sampler(proc_pid=os.getpid())

    # Semantic warning: distributed eval needs DRPA to make mirror
    # features meaningful. With --nr -1 mirrors are zero-padded, so
    # the per-master logits are computed without remote neighbours and
    # the eval is biased low.
    if args.nr == -1 and args.eval_mode == "distributed":
        _warn("--nr -1 with --eval-mode=distributed: mirror features are "
              "zero-padded, remote neighbours do NOT contribute to local "
              "aggregations, eval accuracy will be biased. Consider "
              "--eval-mode=full (needs --graph-path) or enable DRPA "
              "(--nr 1).", rank=args.rank)

    # partition metadata
    pcp = Path(args.partition_config_path)
    if not pcp.exists():
        raise FileNotFoundError(f"missing: {pcp}")
    with open(pcp.as_posix()) as f:
        part_metadata = json.load(f)
    part_config_parent = pcp.parent

    if part_metadata.get("_partition_type", "") != "vertex-cut":
        _warn(f"_partition_type='{part_metadata.get('_partition_type')}' "
              f"!= 'vertex-cut'; assuming vertex-cut layout anyway",
              rank=args.rank)

    num_parts = part_metadata["num_parts"]
    if num_parts != args.world_size:
        raise ValueError(f"num_parts {num_parts} != world_size {args.world_size}")

    # load partition (CSC-only, no edge features)
    graph, node_feats = load_partition(part_metadata, part_config_parent, args.rank)
    _mem("after partition load", rank=args.rank)

    # masters/total
    inner = graph.ndata["inner_node"]
    n_master = int((inner == 1).sum().item())
    n_total = int(graph.num_nodes())

    # Sanity: converter guarantees masters in [0, n_master). Check the
    # boundary only -- cheaper than min/max over the whole tensor.
    if n_master > 0:
        assert int(inner[n_master - 1].item()) == 1, \
            "expected master at position n_master-1"
    if n_master < n_total:
        assert int(inner[n_master].item()) == 0, \
            "expected mirror at position n_master"

    # load features one role at a time, popping the source tensor so it
    # is not held twice.
    downcast = not args.no_f32_features
    load_partition_feature(graph, node_feats, "feat",       n_master, args.rank,
                           downcast_f64=downcast)
    load_partition_feature(graph, node_feats, "label",      n_master, args.rank)
    load_partition_feature(graph, node_feats, "train_mask", n_master, args.rank)
    load_partition_feature(graph, node_feats, "val_mask",   n_master, args.rank, required=False)
    load_partition_feature(graph, node_feats, "test_mask",  n_master, args.rank, required=False)

    for m in ("train_mask", "val_mask", "test_mask"):
        if m in graph.ndata and graph.ndata[m].dtype != th.bool:
            graph.ndata[m] = graph.ndata[m].bool()

    del node_feats
    gc.collect()
    _mem("after feature load + gc", rank=args.rank)

    # DRPA
    drpa_node_map = attach_drpa_metadata(graph, part_metadata, part_config_parent,
                                         args.rank, num_parts, args.world_size, args)

    # n_classes -- from the partition's own labels; no full-graph read.
    labels = graph.ndata["label"]
    if labels.dim() == 2:
        n_local = int(labels.shape[1])
    else:
        n_local = int(labels[:n_master].max().item()) + 1 if n_master > 0 else 0
    if args.distributed:
        t = th.tensor([n_local], dtype=th.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        n_classes = int(t.item())
    else:
        n_classes = n_local
    del labels
    _info(f"n_classes={n_classes}", rank=args.rank)

    # Fix 1: enforce int64 label dtype (cross_entropy requires it)
    if graph.ndata["label"].dtype != th.int64:
        graph.ndata["label"] = graph.ndata["label"].long()

    # Fix 2: label-range check
    _lbl = graph.ndata["label"][:n_master]
    _lbl_max = int(_lbl.max().item()) if _lbl.numel() > 0 else 0
    if _lbl_max >= n_classes:
        raise RuntimeError(
            f"Label max={_lbl_max} >= n_classes={n_classes}. "
            f"n_classes was inferred incorrectly.")

    # full graph: only for --validate-drpa or --eval-mode=full
    gr_full = None
    if args.validate_drpa or args.eval_mode == "full":
        if args.graph_path is None:
            raise ValueError("--graph-path is required for --validate-drpa "
                             "or --eval-mode=full")
        gr_full = load_full_graph_for_eval(args.graph_path, args.rank,
                                           downcast_f64=downcast)
        if args.rank == 0:
            gc.collect()
        _mem("after full-graph load", rank=args.rank)

    # DRPA wrap (skipped when --nr -1)
    drpa_enabled = args.nr != -1
    if drpa_enabled:
        if drpa_node_map is None:
            drpa_node_map = th.zeros(num_parts, dtype=th.int32)
        t0 = time.time()
        gobj = drpa(graph, args.rank, num_parts, drpa_node_map, args.nr,
                    dist, args.n_layers)
        _info(f"DRPA wrap: {time.time() - t0:.3f}s", rank=args.rank)
    else:
        gobj = graph
        _info("DRPA disabled; training on the bare partition", rank=args.rank)

    # validate-drpa: run, then exit
    if args.validate_drpa:
        if args.nr == -1:
            raise ValueError("--validate-drpa requires --nr >= 0 (--nr 1 ideal)")
        if args.nr != 1:
            _warn(f"--validate-drpa with --nr {args.nr}: DRPA delays remote "
                  f"aggregation; only --nr 1 is directly comparable",
                  rank=args.rank)
        ok = run_drpa_validation(gobj, gr_full, n_classes, num_parts,
                                 args.rank, args.world_size, args)
        try:
            gobj.drpa_finalize()
        except Exception as e:
            _warn(f"drpa_finalize: {e}", rank=args.rank)
        if args.distributed:
            dist.barrier()
        sys.exit(0 if ok else 1)

    # train
    print("Starting a run:", flush=True)
    run_training(gobj, n_classes, num_parts, drpa_node_map,
                 args.rank, args.world_size, args, gr_full=gr_full)
    print("Run completed!", flush=True)

    if drpa_enabled:
        try:
            gobj.drpa_finalize()
        except Exception as e:
            _warn(f"drpa_finalize: {e}", rank=args.rank)

    if args.rank == 0:
        print(f"Dataset: {args.dataset}  --nr: {args.nr}  "
              f"lr: {args.lr}  agg: {args.aggregator_type}", flush=True)

    if args.distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
