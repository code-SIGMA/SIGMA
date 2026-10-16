#!/usr/bin/env python3
# ================================================================================
# Libraries
# ================================================================================
import logging

logging.basicConfig(level=logging.INFO)

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR.parent.parent))
sys.path.append(str(BASE_DIR / "libs"))

import json
import time
import re
import math
import argparse
from datetime import datetime, timezone

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

import dgl
from dgl.data import register_data_args
from dgl.data.utils import load_graphs, load_tensors
from dgl.nn.pytorch.conv import GATConv

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

# ================================================================================
# Setup
# ================================================================================
loader = EnvLoader("/root/env.json")

general = loader.group("general", required=["run_id"])
run_id = general.get("run_id")

if run_id is None:
    raise KeyError("Run id is not given via /root/env.json")

print(f"Run ID: {run_id}", flush=True)

db_env = loader.group(
    "database",
    required=["db_host", "db_port", "db_name", "db_user", "db_password"],
)

database = db_env.build(Database)
DATABASE_LOADED = bool(database)

# ================================================================================
# Model
# ================================================================================


class GAT(nn.Module):
    """GAT as in the DGL example, but the graph is passed to forward()
    (like GraphSAGE here), so the same model runs on the local partition
    during training and on the full graph during evaluation."""

    def __init__(
        self,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # input projection (no residual)
        self.layers.append(
            GATConv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            self.layers.append(
                GATConv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    activation,
                )
            )
        # output projection
        self.layers.append(
            GATConv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, graph, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.layers[l](graph, h).flatten(1)  # concat heads
        return self.layers[-1](graph, h).mean(1)  # average output heads


# ================================================================================
# Helper Functions
# ================================================================================


def evaluate_all(model, graph, features, labels, nid_dict):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        _, pred = th.max(logits, dim=1)
        return {
            k: (int((pred[nid] == labels[nid]).sum()), len(nid))
            for k, nid in nid_dict.items()
        }


def with_self_loops(g):
    """GAT needs a self loop on every node: GATConv refuses zero in-degree
    nodes, and halo nodes in a partition usually have none."""
    return dgl.add_self_loop(dgl.remove_self_loop(g))


def send_system_metrics_to_db(samples, run_id, rank, epoch=-1, batch_size=2000):
    gpu_re = re.compile(r"^gpu(\d+)_(util_percent|mem_util_percent|mem_used_gb)$")
    columns = [
        "run_id",
        "rank",
        "recorded_at",
        "cpu_total_percent",
        "cpu_iowait",
        "cpu_per_core",
        "ram_percent",
        "ram_available_gb",
        "swap_percent",
        "disk_read_mb_s",
        "disk_write_mb_s",
        "net_recv_mb_s",
        "net_sent_mb_s",
        "proc_rss_gb",
        "proc_threads",
        "proc_cpu_percent",
        "gpu_util_percent",
        "gpu_mem_used_gb",
        "gpu_mem_percent",
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
            "run_id": run_id,
            "rank": rank,
            "recorded_at": datetime.fromtimestamp(s["timestamp"], timezone.utc).replace(
                tzinfo=None
            ),
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
        sent += database.copy_rows(
            "system_metric", rows[i : i + batch_size], columns=columns
        )
    return sent


def feature_stats(feat, chunk_rows=1_000_000):
    """Per-dimension feature statistics (float64, chunked)."""
    n, d = feat.shape
    dev = feat.device
    col_sum = th.zeros(d, dtype=th.float64, device=dev)
    col_sqsum = th.zeros(d, dtype=th.float64, device=dev)
    n_nonfinite = 0
    fmin, fmax = math.inf, -math.inf

    for i in range(0, n, chunk_rows):
        chunk = feat[i : i + chunk_rows].to(th.float64)
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
    """Returns (ok, errors, warnings)."""
    errors, warnings = [], []
    max_abs = max(abs(st["min"]), abs(st["max"]))

    if st["n_nonfinite"]:
        errors.append(f"{st['n_nonfinite']} non-finite value(s) (NaN/Inf)")
    if st["max_col_std"] == 0.0:
        errors.append(
            "every feature dimension is constant (std=0), no signal to learn from"
        )
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
        warnings.append(f"heavy outliers: max|x|/std = {max_abs / st['std']:.1f}")
    if st["std"] > 0 and st["std"] < 1e-3:
        warnings.append(
            f"very small dynamic range (std={st['std']:.3g}), gradients may vanish"
        )

    return not errors, errors, warnings


def abort_if_any_failed(ok, message, rank, distributed):
    """Turn a per-rank condition into a collective one (all ranks exit together)."""
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


# ================================================================================
# Run Training
# ================================================================================


def run(g, data):
    n_classes, _, _, rank, _, data_dirpath = data

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    in_feats = features.shape[1]

    train_nid = train_mask.nonzero(as_tuple=False).squeeze(-1)

    # create GAT model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(
        args.num_layers,
        in_feats,
        args.num_hidden,
        n_classes,
        heads,
        F.elu,
        args.in_drop,
        args.attn_drop,
        args.negative_slope,
        args.residual,
    )
    if rank == 0:
        print(model, flush=True)

    if args.distributed:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    # Weight each rank's contribution by its share of training nodes.
    grad_scale = 1.0
    if args.distributed:
        n_local = th.tensor([float(train_nid.numel())], dtype=th.float64)
        n_total = n_local.clone()
        dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
        grad_scale = float(n_local.item()) / float(n_total.item())
        print(
            f"[rank {rank}] train_nid={int(n_local.item())} "
            f"grad_scale={grad_scale:.6f}",
            flush=True,
        )

    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if rank == 0:
        graph_path = Path(args.graph_path)
        gr = load_graphs(graph_path.as_posix())[0][0]
        gr = with_self_loops(gr)

        gr_features = gr.ndata["feat"]
        gr_labels = gr.ndata["label"]

        gr_train_nid = gr.ndata["train_mask"].nonzero(as_tuple=False).squeeze(-1)
        gr_val_nid = gr.ndata["val_mask"].nonzero(as_tuple=False).squeeze(-1)
        gr_test_nid = gr.ndata["test_mask"].nonzero(as_tuple=False).squeeze(-1)

    best_val_acc = float("-inf")
    best_model_state = None
    best_epoch = -1
    epochs_done = 0

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

    mon = None
    if save_metric:
        try:
            mon = SystemMonitor(
                interval=0.5,
                csv_path=(
                    data_dirpath / f"system_metrics_rank{rank}-{run_id}.csv"
                ).as_posix(),
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
                    "epoch",
                    "train_loss",
                    "total_time",
                    "forward_time",
                    "backward_time",
                    "wait_time",
                    "update_time",
                    "train_accuracy",
                    "val_accuracy",
                    "test_accuracy",
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
            epochs_done += 1

            # ----------------------------------------------------------
            # Everything below runs outside the timed section.
            # ----------------------------------------------------------
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
            stop_now = False

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

                if args.early_stop and epoch - best_epoch >= args.patience:
                    stop_now = True

            # ----------------------------------------------------------
            # Logging
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

            del row["wait_time"]

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
                print(
                    (
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        f"Epoch {epoch:05d} | "
                        f"Max Time(s) {max_time:.4f} | "
                        f"Total Time(s) {total_time:.4f} | "
                        f"Forward Time(s) {forward_time:.4f} | "
                        f"Backward Time(s) {backward_time:.4f} | "
                        f"Wait Time(s) {wait_time:.4f} | "
                        f"Update Time(s) {update_time:.4f}"
                    ),
                    flush=True,
                )
                print(flush=True)

            # ----------------------------------------------------------
            # Early stopping: rank 0 decides, every rank follows.
            # Must be collective, otherwise the other ranks hang in the
            # next barrier.
            # ----------------------------------------------------------
            if args.early_stop:
                if args.distributed:
                    flag = th.tensor([1 if stop_now else 0], dtype=th.int32)
                    dist.broadcast(flag, src=0)
                    stop_now = bool(flag.item())
                if stop_now:
                    if rank == 0:
                        print(
                            f"Early stopping at epoch {epoch:05d} "
                            f"(no val improvement for {args.patience} epochs, "
                            f"best epoch {best_epoch:05d})",
                            flush=True,
                        )
                    break

    finally:
        if mon is not None:
            mon.stop()
            print("\n===================================================\n", flush=True)
            print(mon.summary_text(title=f"System metrics | rank {rank}"), flush=True)
            print("\n===================================================\n", flush=True)

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
            model,
            gr,
            gr_features,
            gr_labels,
            {"train": gr_train_nid, "val": gr_val_nid, "test": gr_test_nid},
        )

        best_train_acc = res["train"][0] / res["train"][1] * 100
        best_val_acc = res["val"][0] / res["val"][1] * 100
        best_test_acc = res["test"][0] / res["test"][1] * 100

        eps_rank0 = epochs_done / total_time_sum if total_time_sum > 0 else float("nan")
        eps_max = epochs_done / max_time_sum if max_time_sum > 0 else float("nan")

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
                    "update_time": -1,
                    "train_accuracy": best_train_acc,
                    "val_accuracy": best_val_acc,
                    "test_accuracy": best_test_acc,
                },
            )

        print(
            "#############################################################", flush=True
        )
        print(
            (
                "Best Model Single Node Evaluation:\n"
                f"\tTrain Accuracy: {best_train_acc:.4f}\n"
                f"\tVal Accuracy: {best_val_acc:.4f}\n"
                f"\tTest Accuracy: {best_test_acc:.4f}\n"
                f"\tFrom which Epoch: {best_epoch:05d}\n"
                f"\tEpochs trained: {epochs_done}\n"
                f"General Epochs/s for rank == 0: {eps_rank0:.4f}\n"
                f"General Epochs/s: {eps_max:.4f}"
            ),
            flush=True,
        )
        print(
            "#############################################################", flush=True
        )

    elif rank == 0:
        print(
            f"No Checkpoint. n_epochs({args.n_epochs}) to small "
            f"for eval_each_n_epoch({args.eval_each_n_epoch})",
            flush=True,
        )


# ----------------------------------------------------------------------------------
# DRPA metadata
# ----------------------------------------------------------------------------------


def attach_drpa_metadata(
    graph, part_metadata, part_config_parent, rank, num_parts, world_size, args
):
    if args.nr == -1:
        return None

    node_map = attach_drpa_meta_to_graph(
        graph,
        part_metadata,
        part_config_parent,
        rank,
        num_parts,
        require_orig_nid=True,
        trust_meta=False,
        log=lambda m: print(f"Rank: {rank}, Message: {m}", flush=True),
    )

    assert "_ID" in graph.ndata, "_ID doesnt exist in graph.ndata"
    validate_leaders_consistent_dist(
        graph.dstdata["lf"],
        graph.ndata["_ID"],
        graph.ndata["inner_node"],
        rank,
        world_size,
        dist,
    )
    if rank == 0 and world_size > 1:
        print(flush=True)
        print(
            f"Rank: {rank}, Message: DRPA global-consistency check passed", flush=True
        )
    return node_map


# ----------------------------------------------------------------------------------
def main(graph, data):
    print("Starting a run: ", flush=True)
    run(graph, data)
    print("Run completed !!!", flush=True)


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # Parser Setup
    # ----------------------------------------------------------------------------------
    th.set_printoptions(threshold=10)

    parser = argparse.ArgumentParser(description="Distributed GAT")
    register_data_args(parser)

    # --- GAT hyperparameters (defaults from the DGL GAT example) -------------
    parser.add_argument(
        "--num-heads", type=int, default=8, help="number of hidden attention heads"
    )
    parser.add_argument(
        "--num-out-heads", type=int, default=1, help="number of output attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=1, help="number of hidden layers"
    )
    parser.add_argument(
        "--num-hidden", type=int, default=8, help="number of hidden units per head"
    )
    parser.add_argument(
        "--residual", action="store_true", default=False, help="use residual connection"
    )
    parser.add_argument(
        "--in-drop", type=float, default=0.6, help="input feature dropout"
    )
    parser.add_argument(
        "--attn-drop", type=float, default=0.6, help="attention dropout"
    )
    parser.add_argument(
        "--negative-slope",
        type=float,
        default=0.2,
        help="the negative slope of leaky relu",
    )
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=False,
        help="stop when val accuracy does not improve for --patience epochs",
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="early-stopping patience in epochs"
    )

    # --- DistGNN / DRPA -------------------------------------------------------
    parser.add_argument(
        "--nr",
        type=int,
        default=-1,
        help="#delay in delayed updates; -1 = no DRPA (recommended for GAT)",
    )
    parser.add_argument(
        "--unsafe-drpa",
        action="store_true",
        help="allow DRPA (nr != -1) with GAT although edge softmax is not split-aware",
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

    # --- Data / logging -------------------------------------------------------
    parser.add_argument(
        "--graph-path", type=str, help="Path to dgl graph.", required=True
    )
    parser.add_argument(
        "--partition-config-path",
        type=str,
        help="Path to partition config.",
        required=True,
    )
    parser.add_argument(
        "--data-dirpath",
        type=str,
        help="Path to Directory to save data(like epoch metrics or system metrics).",
    )
    parser.add_argument(
        "--eval-each-n-epoch", default=1, type=int, help="Epoch interval for evaluation"
    )
    parser.add_argument("--send-data-to-database", action="store_true")
    parser.add_argument("--save-data-to-csv", action="store_true")
    parser.add_argument("--save-system-metrics-to-csv", action="store_true")
    parser.add_argument("--save-not-only-rank0", action="store_true")

    args = parser.parse_args()
    print(args, flush=True)

    if args.num_layers < 1:
        raise ValueError("--num-layers must be >= 1")

    # ----------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------
    # Check Condition
    # ----------------------------------------------------------------------------------
    abort_if_any_failed(
        not (args.send_data_to_database and not DATABASE_LOADED),
        "--send-data-to-database used but database is not loaded",
        args.rank,
        args.distributed,
    )

    abort_if_any_failed(
        args.nr == -1 or args.unsafe_drpa,
        "DRPA (--nr != -1) aggregates split vertices by summing partial results; "
        "GAT's edge softmax is normalized per partition, so this is not exact. "
        "Use --nr -1, or pass --unsafe-drpa to run it anyway.",
        args.rank,
        args.distributed,
    )

    data_dirpath = Path("./metrics")
    if args.data_dirpath:
        data_dirpath = Path(args.data_dirpath)

    if args.rank == 0 or args.save_not_only_rank0:
        data_dirpath.mkdir(parents=True, exist_ok=True)
        print(f"Save metrics to {data_dirpath.as_posix()}", flush=True)

    # ----------------------------------------------------------------------------------
    # Load Partition config
    # ----------------------------------------------------------------------------------
    part_config_path = Path(args.partition_config_path)

    if not part_config_path.exists():
        raise FileNotFoundError(
            f"The Partition Config Path '{part_config_path.as_posix()}' doesn't exists."
        )

    with open(part_config_path.as_posix()) as conf_f:
        part_metadata = json.load(conf_f)

    # ----------------------------------------------------------------------------------
    # Validate Partitions
    # ----------------------------------------------------------------------------------
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
        args.rank,
        args.distributed,
    )

    # ----------------------------------------------------------------------------------
    # Load from Partition
    # ----------------------------------------------------------------------------------
    node_feats = load_tensors(
        (part_config_path.parent / part_files["node_feats"]).as_posix()
    )
    graph = load_graphs(
        (part_config_path.parent / part_files["part_graph"]).as_posix()
    )[0][0]

    num_parts = part_metadata["num_parts"]
    node_map = part_metadata["node_map"]

    inner = graph.ndata["inner_node"].bool()
    n_inner = int(inner.sum())
    n_masks = node_feats["_N/train_mask"].shape[0]

    abort_if_any_failed(
        n_inner == n_masks,
        f"inner_node={n_inner} != masks={n_masks}",
        args.rank,
        args.distributed,
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
        args.rank,
        args.distributed,
    )

    graph_path = Path(args.graph_path)
    g_orig = load_graphs(graph_path.as_posix())[0][0]
    for name in ("feat", "label"):
        graph.ndata[name] = g_orig.ndata[name][orig_nid]

    sel = graph.ndata["feat"][:n_inner] if contiguous else graph.ndata["feat"][inner]
    abort_if_any_failed(
        th.allclose(sel, node_feats["_N/feat"]),
        "_ID mapping does not match the partition features",
        args.rank,
        args.distributed,
    )

    # GAT: self loops on the local partition (node data is preserved,
    # node IDs do not change, so all mappings above stay valid).
    n_edges_before = graph.num_edges()
    graph = with_self_loops(graph)
    print(
        f"[rank {args.rank}] self loops: edges {n_edges_before} -> {graph.num_edges()}",
        flush=True,
    )

    # --------------------------------
    # Feature sanity check (rank 0, verdict shared)
    # --------------------------------
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

    # --------------------------------
    # CPU parallelism check
    # --------------------------------
    from startup_checks import (
        cpu_parallelism_report,
        check_cpu_parallelism,
        format_report,
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

    lo, hi = agree_across_ranks(rep["effective_cores"], args.distributed)
    abort_if_any_failed(
        lo >= 0.8 * hi,
        f"ranks disagree on effective cores: min={lo} max={hi}",
        args.rank,
        args.distributed,
    )

    # ----------------------------------------------------------------------------------
    # DRPA (only with --unsafe-drpa)
    # ----------------------------------------------------------------------------------
    drpa_node_map = attach_drpa_metadata(
        graph,
        part_metadata,
        part_config_path.parent,
        args.rank,
        num_parts,
        args.world_size,
        args,
    )

    drpa_enabled = args.nr != -1
    if drpa_enabled:
        from dgl.distgnn.drpa_sym import drpa

        abort_if_any_failed(
            drpa_node_map is not None,
            "attach_drpa_metadata did not return a node_map",
            args.rank,
            args.distributed,
        )
        # Same layer semantics as GraphSAGE: total conv layers = num_layers + 1
        gobj = drpa(
            graph, args.rank, num_parts, drpa_node_map, args.nr, dist, args.num_layers
        )
    else:
        gobj = graph

    data = n_classes, node_map, num_parts, args.rank, args.world_size, data_dirpath
    main(gobj, data)

    if drpa_enabled:
        gobj.drpa_finalize()

    if args.rank == 0:
        print("Run details:", flush=True)
        print("Model: GAT", flush=True)
        print("Dataset:", args.dataset, flush=True)
        print("Delay: ", args.nr, flush=True)
        print("lr: ", args.lr, flush=True)
        print(
            f"Heads: {args.num_heads} x {args.num_layers} hidden, "
            f"{args.num_out_heads} out",
            flush=True,
        )
        print("Hidden per head: ", args.num_hidden, flush=True)
        print(flush=True)
        print(flush=True)

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
        sys.stdout.flush()
        sys.stderr.flush()

# ================================================================================
