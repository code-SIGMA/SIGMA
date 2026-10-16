#!/usr/bin/env python3
#================================================================================
# Libraries
#================================================================================
import logging

logging.basicConfig(level=logging.INFO)

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR.parent.parent))
sys.path.append(str(BASE_DIR / "libs"))

import logging

import json
import time

import re
from datetime import datetime, timezone

#import faulthandler
#faulthandler.enable()

from dataclasses import dataclass
from typing import Optional

import argparse

import math
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from dgl.data import register_data_args
from dgl.data.utils import load_graphs, load_tensors
from dgl.distgnn.drpa_sym import drpa
from dgl.nn.pytorch.conv import SAGEConv
from torch import nn


try:
    import torch_ccl
except ImportError as e:
    print(f"[startup] torch_ccl not available: {e}", flush=True)

from drpa_meta import (
    attach_drpa_meta_to_graph,
    validate_leaders_consistent_dist,
)

from database import Database
from env_loader import EnvLoader
from csv_writer import CsvWriter
from system_monitor import SystemMonitor

#================================================================================
# Setup
#================================================================================
loader = EnvLoader("/root/env.json")

# Gruppe: General
general = loader.group("general", required=["run_id"])
run_id = general.get("run_id")

if run_id is None:
    raise KeyError("Run id is not given via /root/env.json")

print(f"Run ID: {run_id}", flush = True)

# Gruppe: Database (required)
db_env = loader.group(
    "database",
    required=["db_host", "db_port", "db_name", "db_user", "db_password"],
)

database = db_env.build(Database)
DATABASE_LOADED = bool(database)

#================================================================================
# Main Script
#================================================================================

class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        dropout,
        aggregator_type,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        # remote_ = False

        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))

    def forward(self, graph, inputs, inf=False):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

#================================================================================
# Helper Functions
#================================================================================

def evaluate_all(model, graph, features, labels, nid_dict):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        _, pred = th.max(logits, dim=1)
        return {k: (int((pred[nid] == labels[nid]).sum()), len(nid))
                for k, nid in nid_dict.items()}

def send_system_metrics_to_db(samples, run_id, rank, epoch=-1, batch_size=2000):
    gpu_re = re.compile(r"^gpu(\d+)_(util_percent|mem_util_percent|mem_used_gb)$")
    columns = [
        "run_id", "rank", "recorded_at",
        "cpu_total_percent", "cpu_iowait", "cpu_per_core",
        "ram_percent", "ram_available_gb", "swap_percent",
        "disk_read_mb_s", "disk_write_mb_s",
        "net_recv_mb_s", "net_sent_mb_s",
        "proc_rss_gb", "proc_threads", "proc_cpu_percent",
        "gpu_util_percent", "gpu_mem_used_gb", "gpu_mem_percent",
    ]

    def build(s):
        gpu_ids = set()
        for k in s:
            m = gpu_re.match(k)
            if m:
                gpu_ids.add(int(m.group(1)))
        gpus = sorted(gpu_ids)

        def arr(metric):
            return [s.get(f"gpu{i}_{metric}") for i in gpus] if gpus else None

        return {
            "run_id": run_id, "rank": rank,
            "recorded_at": datetime.fromtimestamp(s["timestamp"], timezone.utc).replace(tzinfo=None),
            "cpu_total_percent": s.get("cpu_percent_system"),
            "cpu_iowait": s.get("cpu_iowait_percent"),
            "cpu_per_core": s.get("cpu_per_core"),
            "ram_percent": s.get("ram_percent"),
            "ram_available_gb": s.get("ram_available_gb"),
            "swap_percent": s.get("swap_percent"),
            "disk_read_mb_s": s.get("disk_read_mb_s"),
            "disk_write_mb_s": s.get("disk_write_mb_s"),
            "net_recv_mb_s": s.get("net_recv_mb_s"),
            "net_sent_mb_s": s.get("net_sent_mb_s"),
            "proc_rss_gb": s.get("proc_rss_gb"),
            "proc_threads": s.get("proc_threads"),
            "proc_cpu_percent": s.get("cpu_percent_process_normalized"),
            "gpu_util_percent": arr("util_percent"),
            "gpu_mem_used_gb": arr("mem_used_gb"),
            "gpu_mem_percent": arr("mem_util_percent"),
        }

    rows = [build(s) for s in samples]
    sent = 0
    for i in range(0, len(rows), batch_size):
        sent += database.copy_rows("system_metric", rows[i:i + batch_size], columns=columns)
    return sent


def feature_stats(feat, chunk_rows=1_000_000):
    """Per-dimension feature statistics.

    Accumulated in float64 and chunked over rows, so the check stays
    numerically stable and memory-bounded even for large graphs.
    """
    n, d = feat.shape
    dev = feat.device
    col_sum = th.zeros(d, dtype=th.float64, device=dev)
    col_sqsum = th.zeros(d, dtype=th.float64, device=dev)
    n_nonfinite = 0
    fmin, fmax = math.inf, -math.inf

    for i in range(0, n, chunk_rows):
        chunk = feat[i:i + chunk_rows].to(th.float64)
        finite = th.isfinite(chunk)
        bad = int((~finite).sum().item())
        if bad:
            n_nonfinite += bad
            chunk = th.where(finite, chunk, th.zeros_like(chunk))
        col_sum += chunk.sum(dim=0)
        col_sqsum += (chunk * chunk).sum(dim=0)
        fmin = min(fmin, float(chunk.min()))
        fmax = max(fmax, float(chunk.max()))

    col_mean = col_sum / n
    col_std = (col_sqsum / n - col_mean * col_mean).clamp_min(0.0).sqrt()

    total = n * d
    mean = float(col_sum.sum()) / total
    std = math.sqrt(max(float(col_sqsum.sum()) / total - mean * mean, 0.0))

    return {
        "n_nodes": n,
        "n_dims": d,
        "min": fmin,
        "max": fmax,
        "mean": mean,
        "std": std,
        "n_nonfinite": n_nonfinite,
        "n_dead_dims": int((col_std == 0).sum().item()),
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

#================================================================================
# Run Training
#================================================================================

def run(g, data):
    n_classes, _, _, rank, _, data_dirpath = data

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    in_feats = features.shape[1]

    train_nid = train_mask.nonzero(as_tuple=False).squeeze(-1)

    # create GraphSAGE model
    model = GraphSAGE(
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers,
        F.relu,
        args.dropout,
        args.aggregator_type,
    )

    if args.distributed:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    # Weight each rank's contribution by its share of training nodes. The local
    # loss is a mean over local training nodes, so summing unweighted gradients
    # and dividing by world_size overweights small partitions.
    grad_scale = 1.0
    if args.distributed:
        n_local = th.tensor([float(train_nid.numel())], dtype=th.float64)
        n_total = n_local.clone()
        dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
        grad_scale = float(n_local.item()) / float(n_total.item())
        print(f"[rank {rank}] train_nid={int(n_local.item())} "
              f"grad_scale={grad_scale:.6f}", flush=True)

    # use optimizer
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if rank == 0:
        graph_path = Path(args.graph_path)
        gr = load_graphs(graph_path.as_posix())[0][0]

        gr_features = gr.ndata["feat"]
        gr_labels = gr.ndata["label"]

        gr_train_mask = gr.ndata["train_mask"]
        gr_train_nid = gr_train_mask.nonzero(as_tuple=False).squeeze(-1)
        gr_val_mask = gr.ndata["val_mask"]
        gr_val_nid = gr_val_mask.nonzero(as_tuple=False).squeeze(-1)
        gr_test_mask = gr.ndata["test_mask"]
        gr_test_nid = gr_test_mask.nonzero(as_tuple=False).squeeze(-1)

    best_val_acc = float("-inf")
    best_model_state = None
    best_epoch = -1

    # initialize graph
    total_time_sum = 0
    max_time_sum = 0
    max_time = 0

    # ------------------------------------------------------------------
    # What is this Rank writing?
    # ------------------------------------------------------------------
    write_here = args.save_not_only_rank0 or rank == 0

    save_metric = args.save_system_metrics_to_csv and write_here
    save_csv_data = args.save_data_to_csv and write_here
    send_data = args.send_data_to_database and write_here

    # ------------------------------------------------------------------
    # Initialize monitoring/CSV defensively:
    # a non-writable directory must not kill the training process
    # ------------------------------------------------------------------
    mon = None
    if save_metric:
        try:
            mon = SystemMonitor(
                interval=0.5,
                csv_path=(data_dirpath / f"system_metrics_rank{rank}-{run_id}.csv").as_posix(),
                gpu_indices=[],
            )
            mon.start()
        except OSError as exc:
            print(f"[rank {rank}] System-Monitoring deaktiviert: {exc}", flush=True)
            mon = None
            save_metric = False

    writer = None
    if save_csv_data:
        try:
            writer = CsvWriter(
                path=data_dirpath / f"epoch_metrics_rank{rank}-{run_id}.csv",
                fieldnames=[
                    "epoch", "train_loss", "total_time", "forward_time",
                    "backward_time", "wait_time", "update_time",
                    "train_accuracy", "val_accuracy", "test_accuracy",
                ],
            )
        except OSError as exc:
            print(f"[rank {rank}] CSV-Logging deaktiviert: {exc}", flush=True)
            writer = None
            save_csv_data = False

    # ==================================================================
    # Trainings-Schleife
    # ==================================================================
    try:
        for epoch in range(args.n_epochs):
            is_last_epoch = epoch == args.n_epochs - 1

            # Align all ranks before the timer starts. Without this, waiting
            # caused by the previous epoch (e.g. rank 0's evaluation) is billed
            # to this epoch's compute time on every other rank.
            if args.distributed:
                dist.barrier()

            start_time = time.time()
            model.train()

            logits = model(g, features)
            loss = F.cross_entropy(logits[train_nid], labels[train_nid])

            forward_end = time.time()
            optimizer.zero_grad()
            loss.backward()
            compute_end = time.time()

            # Separate straggler wait from the actual collective: the allreduce
            # would otherwise absorb the load imbalance and appear as comm cost.
            if args.distributed:
                dist.barrier()
            sync_end = time.time()

            if args.distributed:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad.data *= grad_scale
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

            optimizer.step()

            wait_time = sync_end - compute_end
            update_time = time.time() - sync_end
            forward_time = forward_end - start_time
            backward_time = compute_end - forward_end
            total_time = forward_time + backward_time + wait_time + update_time

            total_time_sum += total_time

            max_time = total_time
            if args.distributed:
                t = th.tensor(total_time, dtype=th.float64)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                max_time = float(t.item())

            max_time_sum += max_time

            # ----------------------------------------------------------
            # Everything below runs outside the timed section.
            # ----------------------------------------------------------

            # Global training loss, weighted like the gradients. loss.item()
            # alone is the loss on this rank's partition only.
            loss_val = loss.item() * grad_scale
            if args.distributed:
                lt = th.tensor([loss_val], dtype=th.float64)
                dist.all_reduce(lt, op=dist.ReduceOp.SUM)
                loss_val = float(lt.item())

            # ----------------------------------------------------------
            # Evaluation (rank 0 only, evaluation epochs only)
            # ----------------------------------------------------------
            train_acc = -1.0
            val_acc = -1.0
            test_acc = -1.0

            do_eval = rank == 0 and (
                epoch % args.eval_each_n_epoch == 0 or is_last_epoch
            )

            if do_eval:
                print("--Start Evaluation--", flush=True)
                nid_dict = {"train": gr_train_nid, "val": gr_val_nid}
                if is_last_epoch:
                    nid_dict["test"] = gr_test_nid

                res = evaluate_all(model, gr, gr_features, gr_labels, nid_dict)

                train_acc = res["train"][0] / res["train"][1] * 100
                val_acc = res["val"][0] / res["val"][1] * 100
                if is_last_epoch:
                    test_acc = res["test"][0] / res["test"][1] * 100

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                    best_epoch = epoch

            # ----------------------------------------------------------
            # Logging: jeder schreibende Rank, jede Epoche
            # Accuracy is -1 if no evaluation was performed in this epoch
            # ----------------------------------------------------------
            row = {
                "epoch": epoch,
                "train_loss": loss_val,
                "total_time": total_time,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "wait_time": wait_time,
                "update_time": update_time,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
            }

            if save_csv_data:
                writer.append_row(row)

            del row['wait_time']

            if send_data:
                db_row = dict(row)
                db_row["run_id"] = run_id
                db_row["rank"] = rank
                database.send_data(table_name="epoch_metric", data=db_row)

            # ----------------------------------------------------------
            # Konsolenausgabe
            # ----------------------------------------------------------
            if do_eval:
                msg = (
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Epoch {epoch:05d} | "
                    f"Max Time(s) {max_time:.4f} | "
                    f"Total Time(s) {total_time:.4f} | "
                    f"Forward Time(s) {forward_time:.4f} | "
                    f"Backward Time(s) {backward_time:.4f} | "
                    f"Wait Time(s) {wait_time:.4f} | "
                    f"Update Time(s) {update_time:.4f} | "
                    f"Train Loss {loss_val:.4f} | "
                    f"Train Accuracy {train_acc:.4f} | "
                    f"Val Accuracy {val_acc:.4f} | "
                )
                if is_last_epoch:
                    msg += f"Test Accuracy {test_acc:.4f} | "
                msg += f"Best Epoch {best_epoch:05d}"
                print(msg, flush=True)
                print(flush=True)
            elif rank == 0:
                print((
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Epoch {epoch:05d} | "
                    f"Max Time(s) {max_time:.4f} | "
                    f"Total Time(s) {total_time:.4f} | "
                    f"Forward Time(s) {forward_time:.4f} | "
                    f"Backward Time(s) {backward_time:.4f} | "
                    f"Wait Time(s) {wait_time:.4f} | "
                    f"Update Time(s) {update_time:.4f}"
                    ),
                    flush=True
                )
                print(flush=True)

    finally:
        # ==============================================================
        # Stop the monitor AFTER the loop - even after a crash
        # ==============================================================
        if mon is not None:
            mon.stop()
            print("\n===================================================\n", flush = True)
            print(mon.summary_text(title=f"System metrics | rank {rank}"), flush=True)
            print("\n===================================================\n", flush = True)

            if send_data:
                try:
                    sent = send_system_metrics_to_db(mon.samples(), run_id, rank)
                    print(
                        f"[rank {rank}] {sent} system metrics sent to the database.",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[rank {rank}] Could not send system metrics: {exc}",
                        flush=True,
                    )

    # ==================================================================
    # Abschluss-Evaluation des besten Modells
    # ==================================================================
    if rank == 0 and best_model_state is not None:
        model.load_state_dict(best_model_state)
        res = evaluate_all(
            model, gr, gr_features, gr_labels,
            {"train": gr_train_nid, "val": gr_val_nid, "test": gr_test_nid},
        )

        best_train_acc = res["train"][0] / res["train"][1] * 100
        best_val_acc = res["val"][0] / res["val"][1] * 100
        best_test_acc = res["test"][0] / res["test"][1] * 100

        eps_rank0 = args.n_epochs / total_time_sum if total_time_sum > 0 else float("nan")
        eps_max = args.n_epochs / max_time_sum if max_time_sum > 0 else float("nan")

        if send_data:
            database.send_data(
                table_name="epoch_metric",
                data={
                    "run_id": run_id,
                    "rank": rank,
                    "epoch": -1,
                    "train_loss": -1,
                    "total_time": -1,
                    "forward_time": -1,
                    "backward_time": -1,
#                    "wait_time": -1,
                    "update_time": -1,
                    "train_accuracy": best_train_acc,
                    "val_accuracy": best_val_acc,
                    "test_accuracy": best_test_acc,
                },
            )

        print("#############################################################", flush=True)
        print(
            ("Best Model Single Node Evaluation:\n"
            f"\tTrain Accuracy: {best_train_acc:.4f}\n"
            f"\tVal Accuracy: {best_val_acc:.4f}\n"
            f"\tTest Accuracy: {best_test_acc:.4f}\n"
            f"\tFrom which Epoch: {best_epoch:05d}\n"
            f"General Epochs/s for rank == 0: {eps_rank0:.4f}\n"
            f"General Epochs/s: {eps_max:.4f}"),
            flush=True,
        )
        print("#############################################################", flush=True)

    elif rank == 0:
        print(
            f"No Checkpoint. n_epochs({args.n_epochs}) to small "
            f"for eval_each_n_epoch({args.eval_each_n_epoch})",
            flush=True,
        )

#----------------------------------------------------------------------------------
# DRPA metadata
#----------------------------------------------------------------------------------

def attach_drpa_metadata(graph, part_metadata, part_config_parent,
                         rank, num_parts, world_size, args):
    if args.nr == -1:
        return None

    node_map = attach_drpa_meta_to_graph(
        graph, part_metadata, part_config_parent, rank, num_parts,
        require_orig_nid=True,
        trust_meta=False,
        log=lambda m: print(f"Rank: {rank}, Message: {m}", flush=True),
    )

    assert "_ID" in graph.ndata, "_ID doesnt exist in graph.ndata"
    gids_key = "_ID"
    if gids_key in graph.ndata:
        validate_leaders_consistent_dist(
            graph.dstdata["lf"], graph.ndata[gids_key],
            graph.ndata["inner_node"], rank, world_size, dist)
        if rank == 0 and world_size > 1:
            print(flush=True)
            print(f"Rank: {rank}, Message: DRPA global-consistency check passed", flush=True),
    else:
        print(f"Rank: {rank}, Message: no reshuffled NID; skip global-consistency check", flush=True),
    return node_map

#----------------------------------------------------------------------------------
def main(graph, data):
    print("Starting a run: ", flush = True)
    run(graph, data)
    print("Run completed !!!", flush = True)


def abort_if_any_failed(ok, message, rank, distributed):
    """Turn a per-rank condition into a collective one.

    Every rank reports whether its local check passed; the failure count is summed
    across the group. If any rank failed, all ranks raise together. Without this,
    a single failing rank leaves the others blocked in the next collective until
    the timeout fires, which surfaces as "Connection reset by peer".
    """
    if not ok:
        print(f"[rank {rank}] {message}", flush=True)

    n_failed = 0 if ok else 1
    if distributed:
        counter = th.tensor([n_failed], dtype=th.int32)
        dist.all_reduce(counter, op=dist.ReduceOp.SUM)
        n_failed = int(counter.item())

    if n_failed:
        raise RuntimeError(
            f"Aborting: check failed on {n_failed} rank(s) -- see the [rank N] lines above"
        )

if __name__ == "__main__":
    #----------------------------------------------------------------------------------
    # Parser Setup
    #----------------------------------------------------------------------------------
    th.set_printoptions(threshold=10)

    parser = argparse.ArgumentParser(description="GraphSAGE")
    register_data_args(parser)
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument("--nr", type=int, default=1, help="#delay in delayed updates")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--aggregator-type",
        type=str,
        default="gcn",
        choices=["gcn", "mean", "pool", "lstm"],
        help="Aggregator type: mean/gcn/pool/lstm",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="ccl", type=str, help="distributed backend"
    )

    #----------------------

    parser.add_argument(
        "--graph-path", type=str, help="Path to dgl graph.", required=True
    )

    parser.add_argument(
        "--partition-config-path", type=str, help="Path to partition config.", required=True
    )

    parser.add_argument(
        "--data-dirpath", type=str, help="Path to Directory to save data(like epoch metrics or system metrics)."
    )

    parser.add_argument(
        "--eval-each-n-epoch", default=1, type=int, help="Epoch interval for evaluation"
    )

    parser.add_argument(
        "--send-data-to-database", action="store_true"
    )

    parser.add_argument(
        "--save-data-to-csv", action="store_true"
    )

    parser.add_argument(
        "--save-system-metrics-to-csv", action="store_true"
    )

    parser.add_argument(
        "--save-not-only-rank0", action="store_true"
    )


    args = parser.parse_args()
    print(args, flush=True)

    #----------------------------------------------------------------------------------
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("PMI_SIZE", "-1"))
        if args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1
    if args.distributed:
        args.rank = int(os.environ.get("PMI_RANK", "-1"))
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    else:
        args.rank = 0

    print("Rank: ", args.rank, " World_size: ", args.world_size, flush=True)
    nc = args.world_size
    #----------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------
    # Check Condition
    #----------------------------------------------------------------------------------
    abort_if_any_failed(
        not (args.send_data_to_database and not DATABASE_LOADED),
        "--send-data-to-database used but database is not loaded",
        args.rank, args.distributed,
    )

    data_dirpath = Path("./metrics")
    if args.data_dirpath:
        data_dirpath = Path(args.data_dirpath)

    if args.rank == 0 or args.save_not_only_rank0:
        data_dirpath.mkdir(parents=True, exist_ok=True)
        print(f"Save metrics to {data_dirpath.as_posix()}", flush=True)
    #----------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------
    # Load Partition config
    #----------------------------------------------------------------------------------

    part_config_path = Path(args.partition_config_path)

    if not part_config_path.exists():
        raise FileNotFoundError(f"The Partition Config Path '{part_config_path.as_posix()}' doesn't exists.")

    with open(part_config_path.as_posix()) as conf_f:
        part_metadata = json.load(conf_f)
    #----------------------------------------------------------------------------------


    #----------------------------------------------------------------------------------
    # Validate Partitions
    #----------------------------------------------------------------------------------
    part_files = part_metadata.get(f"part-{args.rank}")

    missing = []
    if part_files is None:
        missing.append(f"part-{args.rank} (entry absent from partition config)")
    else:
        for key in ("node_feats", "edge_feats", "part_graph"):
            if key not in part_files:
                missing.append(key)

    abort_if_any_failed(
        not missing,
        f"partition is incomplete, missing: {', '.join(missing)}",
        args.rank, args.distributed,
    )
    #----------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------
    # Load from Partition
    #----------------------------------------------------------------------------------
    node_feats = load_tensors((part_config_path.parent / part_files["node_feats"]).as_posix())

    graph = load_graphs((part_config_path.parent / part_files["part_graph"]).as_posix())[0][0]

    #----------------------------------------------------------------------------------

    num_parts = part_metadata["num_parts"]
    node_map = part_metadata["node_map"]

    #--------------------------------
    inner = graph.ndata["inner_node"].bool()
    n_inner = int(inner.sum())
    n_masks = node_feats["_N/train_mask"].shape[0]

    abort_if_any_failed(
        n_inner == n_masks,
        f"inner_node={n_inner} != masks={n_masks}",
        args.rank, args.distributed,
    )
    contiguous = bool(th.all(inner[:n_inner]))

    def _place(src):
        full = th.zeros((graph.num_nodes(), *src.shape[1:]), dtype=src.dtype)
        if contiguous:
            full[:n_inner] = src
        else:
            full[inner] = src
        return full

    for name in ("train_mask", "val_mask", "test_mask"):
        graph.ndata[name] = _place(node_feats[f"_N/{name}"])

    orig_nid = graph.ndata["orig_id"].long()
    abort_if_any_failed(
        orig_nid.shape[0] == graph.num_nodes(),
        f"orig_id={orig_nid.shape[0]} != num_nodes={graph.num_nodes()}",
        args.rank, args.distributed,
    )

    graph_path = Path(args.graph_path)
    g_orig = load_graphs(graph_path.as_posix())[0][0]
    for name in ("feat", "label"):
        graph.ndata[name] = g_orig.ndata[name][orig_nid]

    sel = graph.ndata["feat"][:n_inner] if contiguous else graph.ndata["feat"][inner]
    abort_if_any_failed(
        th.allclose(sel, node_feats["_N/feat"]),
        "_ID mapping does not match the partition features",
        args.rank, args.distributed,
    )
    #--------------------------------

    # --------------------------------
    # Feature sanity check on the full graph; only rank 0 can judge it, but the
    # verdict is shared so that every rank exits together.
    ok = True
    reason = "feature sanity check failed on rank 0 -- see the [rank 0] lines above"
    if args.rank == 0:
        st = feature_stats(g_orig.ndata["feat"])
        print(
            f"feat n/dim: {st['n_nodes']}/{st['n_dims']}  "
            f"min/max/mean/std: {st['min']:.3f} {st['max']:.3f} "
            f"{st['mean']:.3f} {st['std']:.3f}  "
            f"dead_dims: {st['n_dead_dims']}  nonfinite: {st['n_nonfinite']}",
            flush=True,
        )
        ok, errors, warnings = check_features(st)
        for w in warnings:
            print(f"[rank 0] warning: {w}", flush=True)
        for e in errors:
            print(f"[rank 0] ERROR: {e}", flush=True)
        if not ok:
            reason = "bad node features: " + "; ".join(errors)
    abort_if_any_failed(ok, reason, args.rank, args.distributed)

    labels_all = g_orig.ndata["label"]
    if not th.is_floating_point(labels_all):
        n_classes = len(th.unique(labels_all))
    else:
        n_classes = len(th.unique(labels_all[th.logical_not(th.isnan(labels_all))]))
    del g_orig, labels_all
    print("n_classes: ", n_classes, flush=True)
    #--------------------------------

    # --------------------------------
    # CPU parallelism check. Every rank measures itself, then all ranks must
    # agree: an outlier makes epoch times incomparable across configurations.
    from startup_checks import (
        cpu_parallelism_report, check_cpu_parallelism, format_report,
        agree_across_ranks,
    )

    rep = cpu_parallelism_report()
    print(format_report(rep, args.rank), flush=True)

    ok, errors, warnings = check_cpu_parallelism(rep)
    for w in warnings:
        print(f"[rank {args.rank}] warning: {w}", flush=True)
    for e in errors:
        print(f"[rank {args.rank}] ERROR: {e}", flush=True)

    reason = "cpu parallelism check failed"
    if not ok:
        reason = "cpu parallelism: " + "; ".join(errors)
    abort_if_any_failed(ok, reason, args.rank, args.distributed)

    # Ranks must reach comparable parallelism; small differences are noise,
    # a factor-level gap is what makes epoch times incomparable.
    lo, hi = agree_across_ranks(rep["effective_cores"], args.distributed)
    abort_if_any_failed(
        lo >= 0.8 * hi,
        f"ranks disagree on effective cores: min={lo} max={hi}",
        args.rank, args.distributed,
    )


    #----------------------------------------------------------------------------------
    # Load DRPA Node Map
    #----------------------------------------------------------------------------------
    drpa_node_map = attach_drpa_metadata(graph, part_metadata, part_config_path.parent,
                                         args.rank, num_parts, args.world_size, args)
    #----------------------------------------------------------------------------------

    drpa_enabled = args.nr != -1
    if drpa_enabled:
        abort_if_any_failed(
            drpa_node_map is not None,
            "attach_drpa_metadata did not return a node_map",
            args.rank, args.distributed,
        )
        gobj = drpa(graph, args.rank, num_parts, drpa_node_map, args.nr, dist, args.n_layers)
    else:
        gobj = graph

    data = n_classes, node_map, num_parts, args.rank, args.world_size, data_dirpath
    main(gobj, data)

    if drpa_enabled:
        gobj.drpa_finalize()

    if args.rank == 0:
        print("Run details:", flush=True)
        print("Dataset:", args.dataset, flush=True)
        print("Delay: ", args.nr, flush=True)
        print("lr: ", args.lr, flush=True)
        print("Aggregator: ", args.aggregator_type, flush=True)
        print(flush=True)
        print(flush=True)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
        sys.stdout.flush(); sys.stderr.flush()

#================================================================================
