#!/usr/bin/env python3
"""
METIS Node Partitioner for DGL Graphs
=======================================
Partitions graph nodes using the METIS multilevel k-way algorithm,
which minimises edge-cut while keeping partition sizes balanced.

Dependencies:
  pip install dgl torch metis
  # metis also requires the native libmetis.so / metis.dll:
  #   Ubuntu/Debian : sudo apt install libmetis-dev
  #   Fedora/RHEL   : sudo dnf install metis-devel
  #   macOS         : brew install metis
  #   Windows       : build from source or use a conda package

Usage:
  python partition_metis.py \
      --graph graph.dgl \
      --parts 4 \
      --output node_part.txt

  # With extra options:
  python partition_metis.py \
      --graph  graph.dgl \
      --parts  8 \
      --output node_part.txt \
      --objtype cut \
      --verbose
"""

import os
import sys
import time
import math
import json
import logging
import argparse
from collections import Counter
from typing import Dict, List, Tuple

import torch
import dgl

# metis (Python bindings for the METIS library)
try:
    import metis
except ImportError:
    print(
        "[ERROR] The 'metis' Python package is not installed.\n"
        "        Install it with:  pip install metis\n"
        "        You also need the native METIS library; see the docstring for details.",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )
    return logging.getLogger("partition_metis")


# ── Graph loading ─────────────────────────────────────────────────────────────

def load_graph(path: str) -> dgl.DGLGraph:
    log = logging.getLogger("partition_metis")

    if not os.path.exists(path):
        log.error("Graph file not found: %s", path)
        sys.exit(1)

    file_size = os.path.getsize(path)
    log.info("Loading graph from : %s  (%.2f MB)", path, file_size / 1_048_576)

    t0 = time.perf_counter()
    try:
        graphs, _ = dgl.load_graphs(path)
    except Exception as exc:
        log.error("Failed to load graph: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    log.info("Loaded %d graph(s) in %.3f s", len(graphs), elapsed)

    if not graphs:
        log.error("No graphs found in file.")
        sys.exit(1)
    if len(graphs) > 1:
        log.warning("File contains %d graphs — using the first one.", len(graphs))

    g = graphs[0]

    log.info("── Graph summary ───────────────────────────────────────")
    log.info("  Nodes          : %d", g.num_nodes())
    log.info("  Edges          : %d", g.num_edges())
    log.info("  Node feat keys : %s", list(g.ndata.keys()) or "none")
    log.info("  Edge feat keys : %s", list(g.edata.keys()) or "none")

    if g.num_nodes() == 0:
        log.error("Graph has no nodes.")
        sys.exit(1)
    if g.num_edges() == 0:
        log.warning("Graph has no edges — METIS will still run but partitioning is trivial.")

    in_deg  = g.in_degrees().float()
    out_deg = g.out_degrees().float()
    log.debug("  In-degree  — min: %d  max: %d  mean: %.2f",
              int(in_deg.min()), int(in_deg.max()), in_deg.mean().item())
    log.debug("  Out-degree — min: %d  max: %d  mean: %.2f",
              int(out_deg.min()), int(out_deg.max()), out_deg.mean().item())
    log.info("────────────────────────────────────────────────────────")

    return g


# ── Graph → METIS adjacency list ──────────────────────────────────────────────

def build_metis_adjacency(g: dgl.DGLGraph) -> List[List[int]]:
    """
    Convert a DGL graph to the undirected CSR adjacency list that the
    Python `metis` package expects: a list of neighbour-lists, one per node,
    with self-loops removed and duplicate edges deduplicated.
    """
    log = logging.getLogger("partition_metis")
    log.info("Building undirected adjacency list for METIS…")
    t0 = time.perf_counter()

    num_nodes = g.num_nodes()
    src, dst  = g.edges()
    src = src.tolist()
    dst = dst.tolist()

    # Build undirected adjacency (add both directions, remove self-loops)
    adj: List[set] = [set() for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    adj_list = [sorted(neighbours) for neighbours in adj]

    elapsed = time.perf_counter() - t0
    total_undirected = sum(len(nb) for nb in adj_list) // 2
    log.info("Adjacency list built in %.3f s  (%d undirected edges)", elapsed, total_undirected)
    log.debug("  Isolated nodes (degree 0): %d", sum(1 for nb in adj_list if not nb))

    return adj_list


# ── Partition quality stats ───────────────────────────────────────────────────

def _partition_balance_stats(partitions: torch.Tensor, num_parts: int) -> None:
    log = logging.getLogger("partition_metis")
    counts = Counter(partitions.tolist())
    total  = partitions.numel()
    ideal  = total / num_parts

    log.info("── Node partition balance ───────────────────────────────")
    for pid in range(num_parts):
        n   = counts.get(pid, 0)
        pct = 100.0 * n / total if total else 0.0
        bar = "█" * int(pct / 2)
        log.info("  Part %3d : %6d nodes  (%5.1f %%)  %s", pid, n, pct, bar)

    if counts:
        max_c     = max(counts.values())
        min_c     = min(counts.values())
        imbalance = (max_c - min_c) / ideal * 100 if ideal else 0.0
        log.info("  Max / Min / Ideal        : %d / %d / %.1f", max_c, min_c, ideal)
        log.info("  Imbalance (max−min)/ideal: %.1f %%", imbalance)
    log.info("────────────────────────────────────────────────────────")


def _edge_cut_stats(g: dgl.DGLGraph, partitions: torch.Tensor) -> None:
    log = logging.getLogger("partition_metis")
    src, dst    = g.edges()
    total_edges = g.num_edges()

    if total_edges == 0:
        log.info("Edge-cut stats: no edges in graph.")
        return

    cross = (partitions[src] != partitions[dst]).sum().item()
    local = total_edges - cross
    log.info("── Edge-cut stats ──────────────────────────────────────")
    log.info("  Total edges       : %d", total_edges)
    log.info("  Cross-part edges  : %d  (%.2f %%)", cross, 100.0 * cross / total_edges)
    log.info("  Local edges       : %d  (%.2f %%)", local, 100.0 * local / total_edges)
    log.info("────────────────────────────────────────────────────────")


# ── METIS partitioning ────────────────────────────────────────────────────────

def metis_node_partition(
    graph: dgl.DGLGraph,
    num_parts: int,
    output_file: str,
    objtype: str = "cut",
) -> Tuple[torch.Tensor, float]:
    """
    Run METIS node partitioning.
    Returns (partitions tensor, core_time in seconds).
    """
    log = logging.getLogger("partition_metis")
    num_nodes = graph.num_nodes()

    log.info("Partitioning %d nodes into %d parts via METIS (objtype=%s)…",
             num_nodes, num_parts, objtype)

    # ── Edge case: single partition ────────────────────────────────────────
    if num_parts == 1:
        log.info("Only 1 partition requested — assigning all nodes to part 0.")
        t0 = time.perf_counter()
        partitions = torch.zeros(num_nodes, dtype=torch.long)
        _write_output(partitions, output_file)
        _partition_balance_stats(partitions, num_parts)
        _edge_cut_stats(graph, partitions)
        return partitions, time.perf_counter() - t0

    adj_list = build_metis_adjacency(graph)

    # ── Run METIS ──────────────────────────────────────────────────────────
    log.info("Running METIS partitioner…")
    core_start = time.perf_counter()

    try:
        # metis.part_graph returns (edgecut, partition_list)
        edge_cut, membership = metis.part_graph(
            adj_list,
            nparts=num_parts,
            objtype=objtype,   # 'cut' (minimise edge-cut) or 'vol' (minimise comm. volume)
        )
    except Exception as exc:
        log.error("METIS partitioning failed: %s", exc)
        sys.exit(1)

    core_time = time.perf_counter() - core_start
    log.info("METIS finished in %.3f s", core_time)
    log.info("METIS reported edge-cut : %d", edge_cut)

    if membership is None or len(membership) != num_nodes:
        log.error("METIS returned unexpected membership list (length %s, expected %d).",
                  len(membership) if membership else "None", num_nodes)
        sys.exit(1)

    partitions = torch.tensor(membership, dtype=torch.long)

    # ── Stats ──────────────────────────────────────────────────────────────
    _partition_balance_stats(partitions, num_parts)
    _edge_cut_stats(graph, partitions)

    # ── Write output ───────────────────────────────────────────────────────
    _write_output(partitions, output_file)
    log.info("Total time (adjacency→METIS→write): %.3f s",
             time.perf_counter() - core_start)

    return partitions, core_time


def _write_output(partitions: torch.Tensor, output_file: str) -> None:
    log = logging.getLogger("partition_metis")
    log.info("Writing node partition to: %s", output_file)
    t0 = time.perf_counter()

    with open(output_file, "w") as f:
#        f.write("# node_id partition_id\n")
        for node_id, part_id in enumerate(partitions.tolist()):
            f.write(f"{part_id}\n")
            #f.write(f"{node_id} {part_id}\n")

    elapsed = time.perf_counter() - t0
    out_size = os.path.getsize(output_file)
    log.info("Output written in %.3f s  (%.2f MB)", elapsed, out_size / 1_048_576)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    graph: dgl.DGLGraph,
    partitions: torch.Tensor,
    num_parts: int,
    dataset: str,
    objtype: str,
    core_time: float,
) -> Dict:
    """Compute and return the full metrics dictionary."""
    num_nodes  = graph.num_nodes()
    num_edges  = graph.num_edges()
    src, dst   = graph.edges()
    counts     = Counter(partitions.tolist())

    node_counts  = [counts.get(pid, 0) for pid in range(num_parts)]
    ideal_nodes  = num_nodes / num_parts if num_parts else 1

    # Edge counts per partition (edges where src belongs to that partition)
    edge_counts = [
        int((partitions[src] == pid).sum().item()) for pid in range(num_parts)
    ]
    ideal_edges = num_edges / num_parts if num_parts else 1

    node_balance = max(node_counts) / ideal_nodes if ideal_nodes else math.nan
    edge_balance = max(edge_counts) / ideal_edges if ideal_edges else math.nan

    edge_cut       = int((partitions[src] != partitions[dst]).sum().item())
    edge_cut_ratio = edge_cut / num_edges if num_edges else math.nan

    return {
        "algorithm":         "metis",
        "dataset":           dataset,
        "k":                 num_parts,
        "objtype":           objtype,
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    main_start = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="METIS node partitioner for DGL graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--graph",   type=str, required=True,
                        help="Path to the DGL graph file (.dgl)")
    parser.add_argument("--parts",   type=int, required=True,
                        help="Number of partitions (>= 1)")
    parser.add_argument("--output",  type=str, required=True,
                        help="Output file path  (format: node_id partition_id)")
    parser.add_argument("--name",    type=str, default=None,
                        help="Dataset/graph name used in metrics (default: graph filename stem)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path to write metrics JSON (default: <output>.metrics.json)")
    parser.add_argument("--objtype", type=str, default="cut",
                        choices=["cut", "vol"],
                        help="METIS objective: 'cut' minimises edge-cut (default), "
                             "'vol' minimises total communication volume")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level output")
    args = parser.parse_args()

    log = setup_logger(args.verbose)

    if args.parts < 1:
        log.error("--parts must be >= 1, got %d", args.parts)
        sys.exit(1)

    # Derive dataset name and metrics path from defaults if not provided
    dataset      = args.name or os.path.splitext(os.path.basename(args.graph))[0]
    output_dir   = os.path.dirname(os.path.abspath(args.output))
    metrics_path = args.metrics or os.path.join(output_dir, "metrics.json")

    log.info("METIS node partitioner")
    log.info("  Graph   : %s", args.graph)
    log.info("  Parts   : %d", args.parts)
    log.info("  Objtype : %s", args.objtype)
    log.info("  Output  : %s", args.output)
    log.info("  Metrics : %s", metrics_path)

    graph = load_graph(args.graph)
    partitions, _ = metis_node_partition(
        graph, args.parts, args.output, objtype=args.objtype
    )
    core_time = time.perf_counter() - main_start

    # ── Compute & write metrics ────────────────────────────────────────────
    metrics = compute_metrics(
        graph      = graph,
        partitions = partitions,
        num_parts  = args.parts,
        dataset    = dataset,
        objtype    = args.objtype,
        core_time  = core_time,
    )

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n[done]  Partitioning completed successfully.")
    print(
        "CODEx_METRICS "
        f"algorithm=metis "
        f"dataset={dataset} "
        f"partitioning_type=node "
        f"k={args.parts} "
        f"num_nodes={metrics['num_nodes']} "
        f"num_edges={metrics['num_edges']} "
        f"core_time={core_time:.9f} "
        f"edge_cut={metrics['edge_cut']} "
        f"edge_cut_ratio={metrics['edge_cut_ratio']:.9f} "
        f"node_balance={metrics['node_balance']:.9f} "
        f"edge_balance={metrics['edge_balance']:.9f}"
    )

    # Repository-wide convention: measure until the final metrics write.
    metrics["core_time"] = time.perf_counter() - main_start
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
