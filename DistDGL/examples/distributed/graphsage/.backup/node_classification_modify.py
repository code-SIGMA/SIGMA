import argparse
import copy
import socket
import time
import os
import json
import csv
import uuid
import threading
from typing import Optional

import torch
import torch._dynamo
import torch.nested._internal.nested_tensor
import torch.multiprocessing.reductions

import dgl
import dgl.distributed
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import psycopg
    import psycopg_pool
    _PSYCOPG_AVAILABLE = True
except Exception:
    _PSYCOPG_AVAILABLE = False


#================================================================================
# Setup: run_id + db_config from /root/env.json
#================================================================================

run_id = None
db_config = None

# Global debug flag — toggled from CLI via --debug. Use _dbg() everywhere.
DEBUG = False


def _dbg(msg: str, rank: Optional[int] = None):
    """Print a timestamped debug line if --debug is set. Optionally tag rank."""
    if not DEBUG:
        return
    ts = time.strftime("%H:%M:%S")
    prefix = f"[DEBUG {ts}]"
    if rank is not None:
        prefix += f"[rank {rank}]"
    print(f"{prefix} {msg}", flush=True)

if os.path.exists("/root/env.json"):
    with open("/root/env.json") as f:
        env_json = json.load(f)
        try:
            run_id = uuid.UUID(env_json["run_id"])
            db_config = {
                "db_host":     env_json["db_host"],
                "db_port":     env_json["db_port"],
                "db_name":     env_json["db_name"],
                "db_user":     env_json["db_user"],
                "db_password": env_json["db_password"],
            }
            print(f"run_id: {run_id}\ndb_config: {db_config}")
        except Exception as e:
            print(e, flush=True)


#================================================================================
# Connection Pool (one per process)
#================================================================================

_pool: Optional["psycopg_pool.ConnectionPool"] = None


def _get_pool():
    global _pool
    if db_config is None or not _PSYCOPG_AVAILABLE:
        return None
    if _pool is None:
        conninfo = (
            f"host={db_config['db_host']} "
            f"port={db_config['db_port']} "
            f"dbname={db_config['db_name']} "
            f"user={db_config['db_user']} "
            f"password={db_config['db_password']}"
        )
        t0 = time.time()
        _pool = psycopg_pool.ConnectionPool(conninfo, min_size=1, max_size=4, open=True)
        _dbg(f"psycopg ConnectionPool opened in {time.time() - t0:.3f}s")
    return _pool


#================================================================================
# Background System Metrics Sampler
#================================================================================

SAMPLE_INTERVAL_S = 0.5

_metrics_cache: dict = {}
_metrics_lock = threading.Lock()
_sampler_started = False


def _sample_gpu():
    if not _NVML_AVAILABLE:
        return [], [], []
    gpu_util, gpu_mem_gb, gpu_mem_pct = [], [], []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util.append(float(util.gpu))
            gpu_mem_gb.append(mem.used / (1024**3))
            gpu_mem_pct.append(100.0 * mem.used / mem.total)
    except Exception:
        pass
    return gpu_util, gpu_mem_gb, gpu_mem_pct


def _sampler_loop(proc_pid: Optional[int] = None):
    proc = psutil.Process(proc_pid) if proc_pid else psutil.Process()

    # Prime rolling counters
    psutil.cpu_percent(interval=0.5)
    psutil.cpu_percent(interval=None, percpu=True)
    psutil.cpu_times_percent(interval=0.5)
    proc.cpu_percent(interval=0.5)

    while True:
        try:
            snapshot = {
                "proc_rss_gb":      proc.memory_info().rss / (1024**3),
                "proc_threads":     proc.num_threads(),
                "proc_cpu_percent": proc.cpu_percent(interval=None),

                "cpu_total_percent": psutil.cpu_percent(interval=None),
                "cpu_per_core":      psutil.cpu_percent(interval=None, percpu=True),
                "cpu_iowait":        psutil.cpu_times_percent(interval=None).iowait,

                **{k: v for k, v in zip(
                    ["ram_percent", "ram_available_gb", "swap_percent"],
                    [
                        psutil.virtual_memory().percent,
                        psutil.virtual_memory().available / (1024**3),
                        psutil.swap_memory().percent,
                    ]
                )},

                "disk_read_mb_s":  psutil.disk_io_counters().read_bytes  / (1024**2),
                "disk_write_mb_s": psutil.disk_io_counters().write_bytes / (1024**2),

                "net_recv_mb_s":   psutil.net_io_counters().bytes_recv / (1024**2),
                "net_sent_mb_s":   psutil.net_io_counters().bytes_sent / (1024**2),

                **dict(zip(
                    ["gpu_util_percent", "gpu_mem_used_gb", "gpu_mem_percent"],
                    _sample_gpu()
                )),
            }
            with _metrics_lock:
                _metrics_cache.update(snapshot)
        except Exception as e:
            print(f"[sampler] error: {e}", flush=True)

        time.sleep(SAMPLE_INTERVAL_S)


def start_background_sampler(proc_pid: Optional[int] = None):
    global _sampler_started
    if _sampler_started:
        return
    t0 = time.time()
    t = threading.Thread(target=_sampler_loop, args=(proc_pid,), daemon=True)
    t.start()
    _sampler_started = True
    time.sleep(SAMPLE_INTERVAL_S * 2)
    _dbg(f"Background sampler started (warm-up took {time.time() - t0:.3f}s)")


def _get_cached_metrics() -> dict:
    with _metrics_lock:
        return dict(_metrics_cache)


#================================================================================
# DB Inserts
#================================================================================

def _insert_epoch(db_config, run_id, rank, epoch,
                  train_loss, total_time, forward_time,
                  backward_time, update_time,
                  train_accuracy, val_accuracy, test_accuracy):
    try:
        t0 = time.time()
        pool = _get_pool()
        if pool is None:
            return
        with pool.connection() as conn:
            conn.execute('''
                INSERT INTO epoch_metric (
                    run_id, rank, epoch, train_loss,
                    total_time, forward_time, backward_time, update_time,
                    train_accuracy, val_accuracy, test_accuracy
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ''', (run_id, rank, epoch, train_loss,
                  total_time, forward_time, backward_time, update_time,
                  train_accuracy, val_accuracy, test_accuracy))
        _dbg(f"epoch_metric insert (epoch={epoch}) took {time.time() - t0:.3f}s",
             rank=rank)
    except Exception as e:
        print(f"Failed to send epoch info: {e}", flush=True)


def _insert_system_metric(db_config, run_id, rank, epoch, m: dict):
    try:
        t0 = time.time()
        pool = _get_pool()
        if pool is None:
            return
        with pool.connection() as conn:
            conn.execute('''
                INSERT INTO system_metric (
                    run_id, rank, epoch,
                    cpu_total_percent, cpu_iowait, cpu_per_core,
                    ram_percent, ram_available_gb, swap_percent,
                    disk_read_mb_s, disk_write_mb_s,
                    net_recv_mb_s, net_sent_mb_s,
                    proc_rss_gb, proc_threads, proc_cpu_percent,
                    gpu_util_percent, gpu_mem_used_gb, gpu_mem_percent
                ) VALUES (
                    %s,%s,%s,
                    %s,%s,%s,
                    %s,%s,%s,
                    %s,%s,
                    %s,%s,
                    %s,%s,%s,
                    %s,%s,%s
                )
            ''', (
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
        _dbg(f"system_metric insert (epoch={epoch}) took {time.time() - t0:.3f}s",
             rank=rank)
    except Exception as e:
        print(f"Failed to send system metrics: {e}", flush=True)


def send_epoch_info_to_database(rank, epoch, train_loss,
                                total_time, forward_time, backward_time, update_time,
                                train_accuracy, val_accuracy, test_accuracy):
    global db_config, run_id
    if run_id is None:
        print("Can't send data because run_id is not set.", flush=True)
        return
    if db_config is None:
        print("Database configuration not provided.", flush=True)
        return

    thread = threading.Thread(
        target=_insert_epoch,
        args=(db_config, run_id, rank, epoch, train_loss,
              total_time, forward_time, backward_time, update_time,
              train_accuracy, val_accuracy, test_accuracy),
        daemon=True,
    )
    thread.start()


def send_system_metrics_to_database(rank, epoch, proc_pid=None):
    global db_config, run_id
    if run_id is None:
        print("Can't send data because run_id is not set.", flush=True)
        return
    if db_config is None:
        print("Database configuration not provided.", flush=True)
        return

    m = _get_cached_metrics()
    if not m:
        print("Sampler not ready yet, Skip send data to database.", flush=True)
        return

    thread = threading.Thread(
        target=_insert_system_metric,
        args=(db_config, run_id, rank, epoch, m),
        daemon=True,
    )
    thread.start()


#================================================================================
# CSV logging
#================================================================================

def save_epoch_info_in_csv(rank, epoch, total_time, forward_time, backward_time,
                           update_time, train_loss, train_acc, val_acc, test_acc,
                           filename):
    columns = ["rank", "epoch", "total_time", "forward_time", "backward_time",
               "update_time", "train_loss", "train_acc", "val_acc", "test_acc"]

    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(filename) == 0:
            writer.writerow(columns)

        writer.writerow([
            rank, epoch, total_time, forward_time, backward_time, update_time,
            train_loss, train_acc, val_acc, test_acc,
        ])


#================================================================================
# Model
#================================================================================

class DistSAGE(nn.Module):
    """
    SAGE model for distributed train and evaluation.

    Parameters
    ----------
    in_feats : int
        Feature dimension.
    n_hidden : int
        Hidden layer dimension.
    n_classes : int
        Number of classes.
    n_layers : int
        Number of layers.
    activation : callable
        Activation function.
    dropout : float
        Dropout value.
    """

    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for _ in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Distributed layer-wise inference with the GraphSAGE model on full
        neighbors.
        """
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                out_dim = self.n_classes
                name = "h_last"
            else:
                out_dim = self.n_hidden
                name = "h"
            y = dgl.distributed.DistTensor(
                (g.num_nodes(), out_dim),
                th.float32,
                name,
                persistent=True,
            )
            print(f"|V|={g.num_nodes()}, inference batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.distributed.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return x


def compute_acc(pred, labels):
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, batch_size, device,
             train_nid=None, val_nid=None, test_nid=None):
    """
    Run distributed layer-wise inference once and return accuracies for
    whichever splits are requested.

    Note: model.inference() is *distributed* — every rank must participate
    (it uses DistTensor and g.barrier() internally). So this function must
    be called by all ranks, even if only rank 0 consumes the result.

    Any of train_nid / val_nid / test_nid may be None; the corresponding
    returned accuracy is then None. Indexing into `pred` after the
    distributed inference is essentially free, so including extra splits
    adds no real cost.

    Returns
    -------
    (train_acc, val_acc, test_acc) with None for any split that was not
    requested.
    """
    rank = g.rank()
    model.eval()

    t0 = time.time()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    t_inference = time.time() - t0
    _dbg(f"evaluate: distributed inference took {t_inference:.4f}s "
         f"(|V|={g.num_nodes()}, batch={batch_size})", rank=rank)

    model.train()

    train_acc = val_acc = test_acc = None
    timings = []

    if train_nid is not None:
        t0 = time.time()
        train_acc = compute_acc(pred[train_nid], labels[train_nid])
        timings.append(f"train={time.time() - t0:.4f}s")

    if val_nid is not None:
        t0 = time.time()
        val_acc = compute_acc(pred[val_nid], labels[val_nid])
        timings.append(f"val={time.time() - t0:.4f}s")

    if test_nid is not None:
        t0 = time.time()
        test_acc = compute_acc(pred[test_nid], labels[test_nid])
        timings.append(f"test={time.time() - t0:.4f}s")

    if timings:
        _dbg("evaluate: acc compute   " + " | ".join(timings), rank=rank)

    return train_acc, val_acc, test_acc


def run(args, device, data):
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data

    t0 = time.time()
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.distributed.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    _dbg(f"DistNodeDataLoader built in {time.time() - t0:.3f}s "
         f"(fan_out={args.fan_out}, batch_size={args.batch_size})",
         rank=g.rank())

    t0 = time.time()
    model = DistSAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    if args.num_gpus == 0:
        model = th.nn.parallel.DistributedDataParallel(model)
    else:
        model = th.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    _dbg(f"Model+DDP+optimizer built in {time.time() - t0:.3f}s "
         f"(in_feats={in_feats}, hidden={args.num_hidden}, "
         f"classes={n_classes}, layers={args.num_layers})",
         rank=g.rank())

    # ------------------------------------------------------------------ #
    # Logging helpers                                                    #
    # ------------------------------------------------------------------ #
    # By default only rank 0 writes to CSV/DB   val/test acc are identical
    # across ranks (they come from a distributed inference), and per-rank
    # timings are usually redundant. Pass --log-every-rank to change this.
    rank = g.rank()
    is_logger = (rank == 0) or args.log_every_rank

    def _write_row(epoch_id, total_t, fwd_t, bwd_t, upd_t,
                   loss_val, tr_acc, v_acc, te_acc):
        if not is_logger:
            return
        tr  = float(tr_acc) * 100 if tr_acc is not None else -1
        va  = float(v_acc)  * 100 if v_acc  is not None else -1
        te  = float(te_acc) * 100 if te_acc is not None else -1

        if args.save_to_csv:
            save_epoch_info_in_csv(
                rank, epoch_id, total_t, fwd_t, bwd_t, upd_t,
                loss_val, tr, va, te, args.csv_path,
            )
        if args.send_to_database:
            send_epoch_info_to_database(
                rank, epoch_id, loss_val,
                total_t, fwd_t, bwd_t, upd_t,
                tr, va, te,
            )

    # ------------------------------------------------------------------ #
    # Training loop                                                      #
    # ------------------------------------------------------------------ #
    iter_tput = []
    epoch = 0
    epoch_time = []
    test_acc = 0.0
    val_acc = 0.0

    # Best-model tracking (kept on rank 0; all ranks run evaluate() together
    # because model.inference() is distributed).
    best_val_acc = -1.0
    best_model_state = None

    #=======================================================================
    # Initial evaluation before training (epoch 0 = untrained baseline)
    _dbg("Starting initial eval at epoch 0 (train + val)", rank=rank)
    eval_start = time.time()
    train_acc0, val_acc0, _ = evaluate(
        model.module, g,
        g.ndata["feat"], g.ndata["label"],
        args.batch_size_eval, device,
        train_nid=train_nid, val_nid=val_nid, test_nid=None,
    )
    eval_elapsed = time.time() - eval_start
    if rank == 0:
        print(f"Part {rank}, Epoch 0 (init) | Train Acc {float(train_acc0):.4f}, "
            f"Val Acc {float(val_acc0):.4f}, time: {eval_elapsed:.4f}")

    # Track as best so far so the final-eval path has something to load
    if rank == 0:
        best_val_acc = float(val_acc0)
        best_model_state = copy.deepcopy(model.module.state_dict())

    _write_row(0, eval_elapsed, 0.0, 0.0, 0.0, -1.0,
            float(train_acc0), float(val_acc0), None)
    #=======================================================================

    for _ in range(args.num_epochs):
        epoch += 1
        tic = time.time()
        sample_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        step_time = []

        # Per-epoch loss accumulator (mean over mini-batches)
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                tic_step = time.time()
                sample_time += tic_step - start
                batch_inputs = g.ndata["feat"][input_nodes]
                batch_labels = g.ndata["label"][seeds].long()
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                blocks = [block.to(device) for block in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                start = time.time()
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                forward_end = time.time()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_time += forward_end - start
                backward_time += compute_end - forward_end

                optimizer.step()
                update_time += time.time() - compute_end

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)

                epoch_loss_sum += loss.item()
                epoch_loss_count += 1

                if (step + 1) % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )
                    sample_speed = np.mean(iter_tput[-args.log_every :])
                    mean_step_time = np.mean(step_time[-args.log_every :])
                    print(
                        f"Part {rank} | Epoch {epoch:05d} | Step {step:05d}"
                        f" | Loss {loss.item():.4f} | Train Acc {acc.item():.4f}"
                        f" | Speed (samples/sec) {sample_speed:.4f}"
                        f" | GPU {gpu_mem_alloc:.1f} MB | "
                        f"Mean step time {mean_step_time:.3f} s"
                    )
                start = time.time()

        toc = time.time()
        total_epoch_time = toc - tic
        print(
            f"Part {rank}, Epoch Time(s): {total_epoch_time:.4f}, "
            f"sample+data_copy: {sample_time:.4f}, forward: {forward_time:.4f},"
            f" backward: {backward_time:.4f}, update: {update_time:.4f}, "
            f"#seeds: {num_seeds}, #inputs: {num_inputs}"
        )
        epoch_time.append(total_epoch_time)

        mean_train_loss = (epoch_loss_sum / epoch_loss_count) if epoch_loss_count > 0 else -1.0

        # Per-epoch system metrics (non-blocking cached snapshot)
        if args.send_to_database and is_logger:
            send_system_metrics_to_database(rank, epoch, proc_pid=None)

        # ------------- Evaluation (train + val every eval_every) ------- #
        # model.inference() is DISTRIBUTED   all ranks must participate.
        # We evaluate train + val during training (test only at the end
        # with the best model   as epoch = -1).
        do_eval = (epoch % args.eval_every == 0 or epoch == args.num_epochs)
        if do_eval:
            _dbg(f"Starting periodic eval at epoch {epoch} (train + val)",
                 rank=rank)
            eval_start = time.time()
            train_acc, val_acc, _ = evaluate(
                model.module,
                g,
                g.ndata["feat"],
                g.ndata["label"],
                args.batch_size_eval,
                device,
                train_nid=train_nid,
                val_nid=val_nid,
                test_nid=None,          # no test during training
            )
            eval_elapsed = time.time() - eval_start
            if rank == 0:
                print(
                    f"Part {rank}, Train Acc {float(train_acc):.4f}, "
                    f"Val Acc {float(val_acc):.4f}, "
                    f"time: {eval_elapsed:.4f}"
                )
            _dbg(f"Periodic eval at epoch {epoch} finished in "
                 f"{eval_elapsed:.4f}s "
                 f"(train={float(train_acc):.4f}, val={float(val_acc):.4f})",
                 rank=rank)

            # Track the best model (on rank 0; others just follow the same
            # schedule so the distributed eval stays in sync).
            if rank == 0 and float(val_acc) > best_val_acc:
                best_val_acc = float(val_acc)
                t_copy = time.time()
                best_model_state = copy.deepcopy(model.module.state_dict())
                print(f"New best val acc {best_val_acc*100:.4f}%  "
                      f"(copy took {time.time() - t_copy:.4f}s)", flush=True)

            train_for_log = float(train_acc)
            val_for_log   = float(val_acc)
        else:
            train_for_log = None
            val_for_log   = None

        # test is never computed during training   only in the final eval
        test_for_log = None

        _write_row(
            epoch, total_epoch_time, forward_time, backward_time, update_time,
            mean_train_loss, train_for_log, val_for_log, test_for_log,
        )

    # ------------------------------------------------------------------ #
    # Final evaluation with the best model   logged as epoch = -1        #
    # ------------------------------------------------------------------ #
    # Always run final eval. If rank 0 tracked a best model, load it and
    # broadcast it to everyone; otherwise just use the current state on
    # all ranks (they're already in sync from DDP).
    if rank == 0:
        print(f"\n[Final Eval] Entering final evaluation phase. "
              f"best_model_state is {'set' if best_model_state is not None else 'None'}. "
              f"best_val_acc so far: {best_val_acc*100:.4f}%\n", flush=True)

    # Sync all ranks before we start the final phase so no one is stuck
    # in a previous collective.
    th.distributed.barrier()

    if rank == 0:
        if best_model_state is not None:
            t_load = time.time()
            model.module.load_state_dict(best_model_state)
            print(f"[Final Eval] Loaded best model (val acc: "
                  f"{best_val_acc*100:.4f}%) in {time.time() - t_load:.4f}s",
                  flush=True)
        else:
            print("[Final Eval] No best_model_state recorded   "
                  "using current model state.", flush=True)

    # Broadcast parameters + buffers from rank 0 to all ranks. This is a
    # collective op — every rank MUST participate.
    t_bcast = time.time()
    n_params = n_buffers = 0
    with th.no_grad():
        for param in model.module.parameters():
            th.distributed.broadcast(param.data, src=0)
            n_params += 1
        for buf in model.module.buffers():
            th.distributed.broadcast(buf.data, src=0)
            n_buffers += 1
    if rank == 0:
        print(f"[Final Eval] Broadcast of {n_params} params + "
              f"{n_buffers} buffers took {time.time() - t_bcast:.4f}s",
              flush=True)

    # Run the final eval. All ranks must participate.
    if rank == 0:
        print("[Final Eval] Running distributed inference "
              "(train + val + test)...", flush=True)
    t_final = time.time()
    final_train_acc, final_val_acc, final_test_acc = evaluate(
        model.module,
        g,
        g.ndata["feat"],
        g.ndata["label"],
        args.batch_size_eval,
        device,
        train_nid=train_nid,
        val_nid=val_nid,
        test_nid=test_nid,
    )
    final_time = time.time() - t_final

    if rank == 0:
        print("#############################################################",
              flush=True)
        print(
            f"Best model final eval: "
            f"Train {float(final_train_acc)*100:.4f} | "
            f"Val {float(final_val_acc)*100:.4f} | "
            f"Test {float(final_test_acc)*100:.4f} | "
            f"Time: {final_time:.4f}",
            flush=True,
        )
        print("#############################################################",
              flush=True)

    # Write final row with epoch = -1. Timings carry the eval time in
    # `total_time`; the others are 0 (no forward/backward/update here).
    _write_row(
        -1,                 # epoch
        final_time,         # total_time
        0.0, 0.0, 0.0,      # forward/backward/update
        -1.0,               # loss (not meaningful for eval)
        final_train_acc, final_val_acc, final_test_acc,
    )

    # Report the test accuracy of the best model as the final result.
    test_acc = final_test_acc

    return np.mean(epoch_time[-int(args.num_epochs * 0.8) :]), test_acc


def main(args):
    host_name = socket.gethostname()
    t_overall = time.time()

    print(f"{host_name}: Initializing DistDGL.")
    t0 = time.time()
    dgl.distributed.initialize(args.ip_config, use_graphbolt=args.use_graphbolt)
    _dbg(f"dgl.distributed.initialize took {time.time() - t0:.3f}s")

    print(f"{host_name}: Initializing PyTorch process group.")
    t0 = time.time()
    th.distributed.init_process_group(backend=args.backend)
    _dbg(f"th.distributed.init_process_group ({args.backend}) took "
         f"{time.time() - t0:.3f}s")

    print(f"{host_name}: Initializing DistGraph.")
    t0 = time.time()
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    _dbg(f"DistGraph construction took {time.time() - t0:.3f}s "
         f"(|V|={g.num_nodes()}, |E|={g.num_edges()})")
    print(f"Rank of {host_name}: {g.rank()}")

    # Start background sampler once the process is up
    if args.send_to_database:
        start_background_sampler(proc_pid=os.getpid())

    t0 = time.time()
    pb = g.get_partition_book()
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        train_nid = dgl.distributed.node_split(g.ndata["train_mask"], pb, force_even=True)
        val_nid   = dgl.distributed.node_split(g.ndata["val_mask"],   pb, force_even=True)
        test_nid  = dgl.distributed.node_split(g.ndata["test_mask"],  pb, force_even=True)
    _dbg(f"node_split (train/val/test) took {time.time() - t0:.3f}s",
         rank=g.rank())

    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    num_train_local = len(np.intersect1d(train_nid.numpy(), local_nid))
    num_val_local   = len(np.intersect1d(val_nid.numpy(),   local_nid))
    num_test_local  = len(np.intersect1d(test_nid.numpy(),  local_nid))
    print(
        f"part {g.rank()}, train: {len(train_nid)} (local: {num_train_local}), "
        f"val: {len(val_nid)} (local: {num_val_local}), "
        f"test: {len(test_nid)} (local: {num_test_local})"
    )
    del local_nid

    if args.num_gpus == 0:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))

    n_classes = args.n_classes
    if n_classes == 0:
        t0 = time.time()
        labels = g.ndata["label"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
        _dbg(f"n_classes auto-detection took {time.time() - t0:.3f}s",
             rank=g.rank())
    print(f"Number of classes: {n_classes}")

    in_feats = g.ndata["feat"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g

    _dbg(f"Startup (init + split + setup) total: "
         f"{time.time() - t_overall:.3f}s", rank=g.rank())

    t_run = time.time()
    epoch_time, test_acc = run(args, device, data)
    _dbg(f"run() total wall time: {time.time() - t_run:.3f}s",
         rank=g.rank())

    print(
        f"Summary of node classification(GraphSAGE): GraphName "
        f"{args.graph_name} | TrainEpochTime(mean) {epoch_time:.4f} "
        f"| TestAccuracy {test_acc:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed GraphSAGE.")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--ip_config", type=str, help="The file for IP configuration")
    parser.add_argument("--part_config", type=str, help="The path to the partition config file")
    parser.add_argument("--n_classes", type=int, default=0, help="the number of classes")
    parser.add_argument("--backend", type=str, default="gloo", help="pytorch distributed backend")
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="the number of GPU device. Use 0 for CPU training")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--local_rank", type=int, help="get rank of the process")
    parser.add_argument("--pad-data", default=False, action="store_true",
                        help="Pad train nid to the same length across machine, to ensure num "
                             "of batches to be the same.")
    parser.add_argument("--use_graphbolt", action="store_true",
                        help="Use GraphBolt for distributed train.")

    # ---- New: CSV / Database logging options (analogous to distgnn) ----
    parser.add_argument("--save-to-csv", default=False, action="store_true",
                        help="Append per-epoch metrics to a CSV file.")
    parser.add_argument("--csv-path", type=str, default="training_log.csv",
                        help="Path of the CSV log file (used with --save-to-csv).")
    parser.add_argument("--send-to-database", default=False, action="store_true",
                        help="Send per-epoch and system metrics to the configured database "
                             "(needs /root/env.json with run_id + db_* fields).")
    parser.add_argument("--log-every-rank", default=False, action="store_true",
                        help="By default only rank 0 writes to CSV/DB (val/test acc are "
                             "identical across ranks after distributed inference). "
                             "Enable this flag to write a row per rank instead.")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Print extra [DEBUG] timing/diagnostic info: init steps, "
                             "eval durations, best-state broadcast, DB insert latency, etc.")

    args = parser.parse_args()
    DEBUG = args.debug  # shadow local — also set the module global below
    globals()["DEBUG"] = args.debug
    print(f"Arguments: {args}")
    main(args)
