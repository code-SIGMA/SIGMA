#!/usr/bin/env python3
"""
Random Graph Partitioner using DGL
====================================
Supports node and edge partitioning of DGL graphs with debug output.

Usage:
  # Node Partitioning
  python partition.py \
      --graph graph.dgl \
      --parts 4 \
      --mode node \
      --output node_part.txt

  # Edge Partitioning
  python partition.py \
      --graph graph.dgl \
      --parts 4 \
      --mode edge \
      --output edge_part.txt
"""

import dgl
import torch
import numpy as np
import argparse
import random
import os
import sys
import time
import math
import json
import logging
from collections import Counter
from typing import Dict

# ---------------------------------------------------------------------------
#  Logging setup
# ---------------------------------------------------------------------------

def setup_logger(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )
    return logging.getLogger("partition")


# ---------------------------------------------------------------------------
#  Graph loading
# ---------------------------------------------------------------------------

def load_graph(path: str) -> dgl.DGLGraph:
    """Load a DGL graph from disk with validation and debug info."""
    log = logging.getLogger("partition")

    if not os.path.exists(path):
        log.error("Graph file not found: %s", path)
        sys.exit(1)

    file_size = os.path.getsize(path)
    log.info("Loading graph from: %s  (%.2f MB)", path, file_size / 1_048_576)

    t0 = time.perf_counter()
    try:
        graphs, label_dict = dgl.load_graphs(path)
    except Exception as exc:
        log.error("Failed to load graph: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    log.info("Loaded %d graph(s) in %.3f s", len(graphs), elapsed)

    if len(graphs) == 0:
        log.error("No graphs found in file.")
        sys.exit(1)

    if len(graphs) > 1:
        log.warning("File contains %d graphs — using the first one.", len(graphs))

    g = graphs[0]

    log.info("─── Graph summary ───────────────────────────────────────")
    log.info("  Nodes          : %d", g.num_nodes())
    log.info("  Edges          : %d", g.num_edges())
    log.info("  Node feat keys : %s", list(g.ndata.keys()) or "none")
    log.info("  Edge feat keys : %s", list(g.edata.keys()) or "none")

    if g.num_nodes() == 0:
        log.error("Graph has no nodes.")
        sys.exit(1)

    if log.isEnabledFor(logging.DEBUG):
        in_deg  = g.in_degrees().float()
        out_deg = g.out_degrees().float()
        log.debug("  In-degree    min: %d  max: %d  mean: %.2f",
                  in_deg.min().item(), in_deg.max().item(), in_deg.mean().item())
        log.debug("  Out-degree   min: %d  max: %d  mean: %.2f",
                  out_deg.min().item(), out_deg.max().item(), out_deg.mean().item())
    log.info("─────────────────────────────────────────────────────────")

    return g


# ---------------------------------------------------------------------------
#  Partition helpers
# ---------------------------------------------------------------------------

def _partition_balance_stats(partitions: torch.Tensor, num_parts: int, label: str) -> None:
    """Log per-partition counts and imbalance ratio (vectorised)."""
    log = logging.getLogger("partition")

    # Use bincount instead of Counter — O(n) single C-pass
    counts = torch.bincount(partitions, minlength=num_parts)
    total  = partitions.numel()

    log.info("─── %s partition balance ────────────────────────────────", label)
    for pid in range(num_parts):
        n   = counts[pid].item()
        pct = 100.0 * n / total if total else 0.0
        bar = "█" * int(pct / 2)
        log.info("  Part %3d : %6d  (%5.1f %%)  %s", pid, n, pct, bar)

    max_count = counts.max().item()
    min_count = counts.min().item()
    ideal     = total / num_parts
    imbalance = (max_count - min_count) / ideal * 100 if ideal else 0.0
    log.info("  Imbalance (max-min)/ideal : %.1f %%", imbalance)
    log.info("─────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
#  Vectorised file writers
# ---------------------------------------------------------------------------

def _write_node_partition(output_file: str, partitions: torch.Tensor) -> None:
    """Write one partition-id per line using numpy (10-20× faster than Python loop)."""
    np.savetxt(output_file, partitions.numpy(), fmt="%d")


def _write_edge_partition(
    output_file: str,
    unique_edges: torch.Tensor,
    partitions: torch.Tensor,
) -> None:
    """Write 'u v part_id' rows using numpy (10-20× faster than Python loop)."""
    data = np.stack([
        unique_edges[:, 0].numpy(),
        unique_edges[:, 1].numpy(),
        partitions.numpy(),
    ], axis=1)
    np.savetxt(output_file, data, fmt="%d")


# ---------------------------------------------------------------------------
#  Node partitioning
# ---------------------------------------------------------------------------

def random_node_partition(
    graph: dgl.DGLGraph,
    num_parts: int,
    output_file: str,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Randomly partition nodes. Returns (partitions tensor [CPU], core_time)."""
    log = logging.getLogger("partition")
    num_nodes = graph.num_nodes()

    log.info("Partitioning %d nodes into %d parts (random)", num_nodes, num_parts)
    t0 = time.perf_counter()

    # Generate on target device, then move to CPU once for I/O
    partitions_dev = torch.randint(0, num_parts, (num_nodes,), device=device)
    partitions     = partitions_dev.cpu()

    _partition_balance_stats(partitions, num_parts, "Node")

    # Cross-partition edge count — fully vectorised
    src, dst = graph.edges()
    src = src.to(device)
    dst = dst.to(device)
    cross      = (partitions_dev[src] != partitions_dev[dst]).sum().item()
    total_edges = graph.num_edges()
    log.info("Cross-partition edges : %d / %d  (%.1f %%)",
             cross, total_edges, 100.0 * cross / total_edges if total_edges else 0.0)

    log.info("Writing node partition to: %s", output_file)
    _write_node_partition(output_file, partitions)

    core_time = time.perf_counter() - t0
    log.info("Node partitioning done in %.3f s → %s", core_time, output_file)
    return partitions, core_time


# ---------------------------------------------------------------------------
#  Edge partitioning
# ---------------------------------------------------------------------------

def random_edge_partition(
    graph: dgl.DGLGraph,
    num_parts: int,
    output_file: str,
    device: torch.device,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Randomly partition unique undirected edges.

    Returns:
        partitions:   1D tensor of length n_unique_edges, one part ID per edge.
        core_time:    wall-clock time spent partitioning.
        unique_edges: 2D tensor of shape (n_unique_edges, 2) canonical (u<v).
    """
    log = logging.getLogger("partition")
    num_edges = graph.num_edges()

    if num_edges == 0:
        log.error("Graph has no edges — cannot partition edges.")
        sys.exit(1)

    log.info("Partitioning %d edges into %d parts (random)", num_edges, num_parts)
    t0 = time.perf_counter()

    src, dst = graph.edges()
    src = src.to(device)
    dst = dst.to(device)

    # --- Canonicalise to undirected unique edges (no self-loops) -------------
    # Use Cantor-pairing on a 1D int64 key — torch.unique on 1D is much faster
    # than on a 2D tensor (better cache behaviour, simpler sort).
    mask = src != dst
    u = torch.minimum(src[mask], dst[mask])
    v = torch.maximum(src[mask], dst[mask])

    num_nodes  = graph.num_nodes()
    stride     = num_nodes + 1                          # safe stride (u < v < num_nodes)
    keys       = u * stride + v                         # unique int64 per pair
    unique_keys = torch.unique(keys, sorted=False)      # 1D unique — fast

    u_unique = (unique_keys // stride).cpu()
    v_unique = (unique_keys %  stride).cpu()
    unique_edges = torch.stack([u_unique, v_unique], dim=1)  # (n_unique, 2) on CPU

    n_unique = unique_edges.size(0)
    if n_unique != num_edges:
        log.info(
            "Reduced %d directed edges → %d unique undirected edges "
            "(self-loops and reciprocal pairs collapsed).",
            num_edges, n_unique,
        )

    partitions = torch.randint(0, num_parts, (n_unique,))   # CPU; small alloc

    _partition_balance_stats(partitions, num_parts, "Edge")

    # Per-partition node footprint (only computed in verbose/debug mode)
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Computing per-partition node footprint")
        for pid in range(num_parts):
            pid_mask = partitions == pid
            nodes = torch.cat(
                [unique_edges[pid_mask, 0], unique_edges[pid_mask, 1]]
            ).unique()
            log.debug("  Part %3d : %d unique endpoint nodes", pid, nodes.numel())

    log.info("Writing edge partition to: %s", output_file)
    _write_edge_partition(output_file, unique_edges, partitions)

    core_time = time.perf_counter() - t0
    log.info("Edge partitioning done in %.3f s → %s", core_time, output_file)
    return partitions, core_time, unique_edges


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def compute_node_metrics(
    graph: dgl.DGLGraph,
    partitions: torch.Tensor,
    num_parts: int,
    dataset: str,
    core_time: float,
) -> Dict:
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    src, dst  = graph.edges()

    # bincount instead of per-partition loops
    node_counts = torch.bincount(partitions, minlength=num_parts).tolist()
    edge_counts = torch.bincount(partitions[src], minlength=num_parts).tolist()

    node_balance   = max(node_counts) / (num_nodes / num_parts) if num_nodes else math.nan
    edge_balance   = max(edge_counts) / (num_edges / num_parts) if num_edges else math.nan
    edge_cut       = int((partitions[src] != partitions[dst]).sum().item())
    edge_cut_ratio = edge_cut / num_edges if num_edges else math.nan

    return {
        "algorithm":         "random",
        "dataset":           dataset,
        "k":                 num_parts,
        "partitioning_type": "node",
        "num_parts":         num_parts,
        "num_nodes":         num_nodes,
        "num_edges":         num_edges,
        "node_counts":       node_counts,
        "edge_counts":       edge_counts,
        "edge_cut":          edge_cut,
        "edge_cut_ratio":    edge_cut_ratio,
        "node_balance":      node_balance,
        "edge_balance":      edge_balance,
        "core_time":         core_time,
    }


def compute_edge_metrics(
    graph: dgl.DGLGraph,
    partitions: torch.Tensor,
    unique_edges: torch.Tensor,
    num_parts: int,
    dataset: str,
    core_time: float,
) -> Dict:
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    n_unique  = unique_edges.size(0)

    u = unique_edges[:, 0]
    v = unique_edges[:, 1]

    # bincount instead of Counter + loop
    edge_counts = torch.bincount(partitions, minlength=num_parts).tolist()

    # Node replication: nodes can appear in multiple partitions
    node_sets = [
        set(u[partitions == pid].tolist()) | set(v[partitions == pid].tolist())
        for pid in range(num_parts)
    ]
    node_counts = [len(s) for s in node_sets]

    edge_balance       = max(edge_counts) / (n_unique / num_parts) if n_unique else math.nan
    replication_factor = sum(node_counts) / num_nodes if num_nodes else math.nan
    node_balance       = max(node_counts) / (sum(node_counts) / num_parts) if num_parts else math.nan

    return {
        "algorithm":           "random",
        "dataset":             dataset,
        "k":                   num_parts,
        "partitioning_type":   "edge",
        "num_parts":           num_parts,
        "num_nodes":           num_nodes,
        "num_edges":           num_edges,
        "num_unique_edges":    n_unique,
        "node_counts":         node_counts,
        "edge_counts":         edge_counts,
        "edge_balance":        edge_balance,
        "node_balance":        node_balance,
        "replication_factor":  replication_factor,
        "core_time":           core_time,
    }


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    main_start = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Random graph partitioner (DGL) — optimised",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--graph",   type=str, required=True,
                        help="Path to the DGL graph file (.dgl)")
    parser.add_argument("--parts",   type=int, required=True,
                        help="Number of partitions (>= 1)")
    parser.add_argument("--mode",    type=str, choices=["node", "edge"], required=True,
                        help="Partition mode: 'node' or 'edge'")
    parser.add_argument("--output",  type=str, required=True,
                        help="Output file path")
    parser.add_argument("--name",    type=str, default=None,
                        help="Dataset/graph name used in metrics (default: graph filename stem)")
    parser.add_argument("--seed",    type=int, default=None,
                        help="Random seed for reproducibility (optional)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level output")
    parser.add_argument("--gpu",     action="store_true",
                        help="Use CUDA GPU for tensor operations if available")
    args = parser.parse_args()

    log = setup_logger(args.verbose)

    if args.parts < 1:
        log.error("--parts must be >= 1, got %d", args.parts)
        sys.exit(1)

    # Device selection
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        if args.gpu:
            log.warning("--gpu requested but CUDA is not available — falling back to CPU.")
        log.info("Using device: CPU")

    dataset      = args.name or os.path.splitext(os.path.basename(args.graph))[0]
    output_dir   = os.path.dirname(os.path.abspath(args.output))
    metrics_path = os.path.join(output_dir, "metrics.json")

    # Seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        log.info("Random seed set to %d", args.seed)
    else:
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        torch.manual_seed(seed)
        log.info("No seed provided — using random seed %d", seed)

    log.info("Mode: %s  |  Parts: %d  |  Output: %s", args.mode, args.parts, args.output)
    log.info("Metrics : %s", metrics_path)

    graph = load_graph(args.graph)

    if args.mode == "node":
        partitions, _ = random_node_partition(graph, args.parts, args.output, device)
        core_time = time.perf_counter() - main_start
        metrics   = compute_node_metrics(graph, partitions, args.parts, dataset, core_time)
        metric_suffix = (
            f"edge_cut={metrics['edge_cut']} "
            f"edge_cut_ratio={metrics['edge_cut_ratio']:.9f} "
            f"node_balance={metrics['node_balance']:.9f} "
            f"edge_balance={metrics['edge_balance']:.9f}"
        )
    else:
        partitions, _, unique_edges = random_edge_partition(
            graph, args.parts, args.output, device
        )
        core_time = time.perf_counter() - main_start
        metrics   = compute_edge_metrics(
            graph, partitions, unique_edges, args.parts,
            dataset, core_time,
        )
        metric_suffix = (
            f"node_balance={metrics['node_balance']:.9f} "
            f"edge_balance={metrics['edge_balance']:.9f} "
            f"replication_factor={metrics['replication_factor']:.9f}"
        )

    print("\n[done]  Partitioning completed successfully.")
    print(
        "CODEx_METRICS "
        f"algorithm=random "
        f"dataset={dataset} "
        f"partitioning_type={args.mode} "
        f"k={args.parts} "
        f"num_nodes={metrics['num_nodes']} "
        f"num_edges={metrics['num_edges']} "
        f"core_time={core_time:.9f} "
        f"{metric_suffix}"
    )

    metrics["core_time"] = time.perf_counter() - main_start
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
