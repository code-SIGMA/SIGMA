#!/usr/bin/env python3
"""
LDG Node Partitioner for DGL Graphs
======================================
Partitions graph nodes using the Linear Deterministic Greedy (LDG) streaming
algorithm, ported from the Java EdgeCutSGP implementation.

Each node is assigned to the partition that maximises:
    score(i) = (1 - partitionSize[i] / capacity) * neighbors_in_partition(i)

where capacity = (num_nodes / k) * (1 + slack).

Ties are broken randomly among all tied partitions.  A partition is only
eligible if it has not yet reached capacity.

Dependencies:
  pip install dgl torch numpy

Usage:
  python ldg_partitioner.py \\
      --graph  graph.dgl \\
      --parts  4 \\
      --output node_part.txt

  # With slack and verbose logging:
  python ldg_partitioner.py \\
      --graph   graph.dgl \\
      --parts   8 \\
      --output  node_part.txt \\
      --slack   0.1 \\
      --verbose
"""

import os
import sys
import time
import math
import json
import random
import logging
import argparse
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
import dgl


#    Logging

def setup_logger(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )
    return logging.getLogger("ldg_partitioner")


#    Graph loading

def load_graph(path: str) -> dgl.DGLGraph:
    log = logging.getLogger("ldg_partitioner")

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

    log.info("   Graph summary                                        ")
    log.info("  Nodes          : %d", g.num_nodes())
    log.info("  Edges          : %d", g.num_edges())
    log.info("  Node feat keys : %s", list(g.ndata.keys()) or "none")
    log.info("  Edge feat keys : %s", list(g.edata.keys()) or "none")

    if g.num_nodes() == 0:
        log.error("Graph has no nodes.")
        sys.exit(1)
    if g.num_edges() == 0:
        log.warning("Graph has no edges — all nodes will score equally; "
                    "partitioning degrades to round-robin.")

    in_deg  = g.in_degrees().float()
    out_deg = g.out_degrees().float()
    log.debug("  In-degree    min: %d  max: %d  mean: %.2f",
              int(in_deg.min()), int(in_deg.max()), in_deg.mean().item())
    log.debug("  Out-degree   min: %d  max: %d  mean: %.2f",
              int(out_deg.min()), int(out_deg.max()), out_deg.mean().item())
    log.info("                                                        ")

    return g


#    Adjacency list

def build_adjacency(g: dgl.DGLGraph) -> List[List[int]]:
    """
    Build an undirected neighbour list (list-of-lists) from the DGL graph.
    Self-loops are removed; duplicate edges are deduplicated.
    Index i holds the sorted list of neighbours of node i.
    """
    log = logging.getLogger("ldg_partitioner")
    log.info("Building adjacency list ")
    t0 = time.perf_counter()

    num_nodes = g.num_nodes()
    src, dst  = g.edges()
    src = src.tolist()
    dst = dst.tolist()

    adj: List[set] = [set() for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    adj_list = [sorted(nb) for nb in adj]

    elapsed          = time.perf_counter() - t0
    undirected_edges = sum(len(nb) for nb in adj_list) // 2
    isolated         = sum(1 for nb in adj_list if not nb)
    log.info("Adjacency list built in %.3f s  (%d undirected edges)", elapsed, undirected_edges)
    log.debug("  Isolated nodes (degree 0): %d", isolated)
    log.info("                                                        ")

    return adj_list


#    LDG node partitioner

class LDGPartitioner:
    """
    Streaming node partitioner — direct port of Java EdgeCutSGP.ldg_partition.

    Processes nodes one at a time in order 0..N-1.  For each node the
    already-assigned neighbours are counted per partition; the partition with
    the highest LDG score that has not reached capacity is chosen.
    Ties are broken uniformly at random, matching Java's Random.nextInt logic.
    """

    def __init__(self, num_nodes: int, num_partitions: int, slack: float = 0.0):
        self.n        = num_nodes
        self.k        = num_partitions
        self.slack    = slack

        # capacity = (n / k) * (1 + slack)  — mirrors Java exactly
        self.capacity = max(int((num_nodes / num_partitions) * (1.0 + slack)), 1)

        # Per-partition node counts  (Java: partitionSizes[])
        self.partition_sizes: List[int] = [0] * num_partitions

        # Final assignment: node_id -> partition_id
        self.vertex_to_partition: Dict[int, int] = {}

        # Running edge-cut counters  (Java: numberOfEdges / numberOfEdgecut)
        self._total_edge_refs: int = 0
        self._edge_cut_refs:   int = 0

        self.log = logging.getLogger("ldg_partitioner")

    # ------------------------------------------------------------------
    def _neighbors_in_partition(self, pid: int, neighbors: List[int]) -> int:
        """
        Count how many of this node's neighbours are already assigned to
        partition pid.  Mirrors Java EdgeCutSGP.neighbors_in_partition.
        """
        count = 0
        for nb in neighbors:
            if self.vertex_to_partition.get(nb, -1) == pid:
                count += 1
        return count

    # ------------------------------------------------------------------
    def _assign_node(self, node_id: int, neighbors: List[int]) -> int:
        """
        Choose and return the best partition for node_id using LDG scoring:
            score(i) = (1 - partitionSize[i] / capacity) * neighbors_in_partition(i)
        Mirrors Java EdgeCutSGP.ldg_partition.
        """
        k = self.k

        nb_counts = [self._neighbors_in_partition(i, neighbors) for i in range(k)]

        best_score:  float     = float("-inf")
        best_pid:    int       = -1
        tie_breaker: List[int] = []

        for i in range(k):
            if self.partition_sizes[i] >= self.capacity:
                continue   # partition full — ineligible

            score = (1.0 - self.partition_sizes[i] / self.capacity) * nb_counts[i]

            if score > best_score:
                best_score  = score
                best_pid    = i
                tie_breaker = [i]
            elif score == best_score:
                tie_breaker.append(i)

        # Random tie-breaking — matches Java's Random.nextInt(TieBreaker.size())
        if len(tie_breaker) > 1:
            best_pid = random.choice(tie_breaker)
        elif len(tie_breaker) == 1:
            best_pid = tie_breaker[0]
        else:
            # All partitions at capacity (slack too tight) — fall back to least loaded
            self.log.warning(
                "Node %d: all partitions at capacity — assigning to least loaded.", node_id
            )
            best_pid = int(np.argmin(self.partition_sizes))

        # Update running edge-cut counters (mirrors Java)
        for i in range(k):
            self._total_edge_refs += nb_counts[i]
            if i != best_pid:
                self._edge_cut_refs += nb_counts[i]

        return best_pid

    # ------------------------------------------------------------------
    def run(self, adj_list: List[List[int]]) -> Tuple[Dict[int, int], float]:
        """
        Stream all nodes through the LDG algorithm in order 0..N-1.
        Returns (vertex_to_partition dict, core_time in seconds).
        """
        log       = self.log
        num_nodes = len(adj_list)

        log.info("Running LDG node partitioner ")
        log.info("  Nodes      : %d", num_nodes)
        log.info("  Partitions : %d", self.k)
        log.info("  Capacity   : %d  (slack=%.2f)", self.capacity, self.slack)

        core_start   = time.perf_counter()
        log_interval = max(1, num_nodes // 20)

        for node_id in range(num_nodes):
            neighbors = adj_list[node_id]
            best_pid  = self._assign_node(node_id, neighbors)

            self.vertex_to_partition[node_id]  = best_pid
            self.partition_sizes[best_pid]     += 1

            self.log.debug(
                "Node %d  neighbors=%d  -> pid=%d  (partition_size=%d)",
                node_id, len(neighbors), best_pid, self.partition_sizes[best_pid],
            )

            if (node_id + 1) % log_interval == 0 or node_id == num_nodes - 1:
                elapsed = time.perf_counter() - core_start
                log.info(
                    "  Progress: %d/%d nodes (%.1f%%) | elapsed=%.2fs",
                    node_id + 1, num_nodes,
                    100.0 * (node_id + 1) / num_nodes,
                    elapsed,
                )

        core_time = time.perf_counter() - core_start
        log.info("LDG finished in %.3f s", core_time)

        self._log_partition_balance()
        self._log_edge_cut_stats()

        return self.vertex_to_partition, core_time

    # ------------------------------------------------------------------
    def _log_partition_balance(self) -> None:
        log   = self.log
        total = sum(self.partition_sizes)
        ideal = total / self.k if self.k else 1

        log.info("   Node partition balance                                ")
        for pid in range(self.k):
            n   = self.partition_sizes[pid]
            pct = 100.0 * n / total if total else 0.0
            bar = "|" * int(pct / 2)
            log.info("  Part %3d : %6d nodes  (%5.1f %%)  %s", pid, n, pct, bar)

        max_c     = max(self.partition_sizes)
        min_c     = min(self.partition_sizes)
        imbalance = (max_c - min_c) / ideal * 100 if ideal else 0.0
        log.info("  Max / Min / Ideal        : %d / %d / %.1f", max_c, min_c, ideal)
        log.info("  Imbalance (max-min)/ideal: %.1f %%", imbalance)
        log.info("  Capacity per partition   : %d", self.capacity)
        log.info("                                                        ")

    def _log_edge_cut_stats(self) -> None:
        log = self.log
        log.info("   Edge-cut stats (streaming estimate)                  ")
        log.info("  Total neighbour refs : %d", self._total_edge_refs)
        log.info("  Cut neighbour refs   : %d", self._edge_cut_refs)
        if self._total_edge_refs > 0:
            log.info(
                "  Estimated cut ratio  : %.4f %%",
                100.0 * self._edge_cut_refs / self._total_edge_refs,
            )
        log.info("                                                        ")


#    Partition quality stats

def _partition_balance_stats(partitions: torch.Tensor, num_parts: int) -> None:
    log    = logging.getLogger("ldg_partitioner")
    counts = Counter(partitions.tolist())
    total  = partitions.numel()
    ideal  = total / num_parts

    log.info("   Node partition balance (post-hoc)                     ")
    for pid in range(num_parts):
        n   = counts.get(pid, 0)
        pct = 100.0 * n / total if total else 0.0
        bar = "|" * int(pct / 2)
        log.info("  Part %3d : %6d nodes  (%5.1f %%)  %s", pid, n, pct, bar)

    if counts:
        max_c     = max(counts.values())
        min_c     = min(counts.values())
        imbalance = (max_c - min_c) / ideal * 100 if ideal else 0.0
        log.info("  Max / Min / Ideal        : %d / %d / %.1f", max_c, min_c, ideal)
        log.info("  Imbalance (max-min)/ideal: %.1f %%", imbalance)
    log.info("                                                        ")


def _edge_cut_stats(g: dgl.DGLGraph, partitions: torch.Tensor) -> None:
    log         = logging.getLogger("ldg_partitioner")
    src, dst    = g.edges()
    total_edges = g.num_edges()

    if total_edges == 0:
        log.info("Edge-cut stats: no edges in graph.")
        return

    cross = (partitions[src] != partitions[dst]).sum().item()
    local = total_edges - cross
    log.info("   Edge-cut stats (post-hoc on full graph)               ")
    log.info("  Total edges       : %d", total_edges)
    log.info("  Cross-part edges  : %d  (%.2f %%)", cross, 100.0 * cross / total_edges)
    log.info("  Local edges       : %d  (%.2f %%)", local, 100.0 * local / total_edges)
    log.info("                                                        ")


#    Output writing

def _write_output(partitions: torch.Tensor, output_file: str) -> None:
    log = logging.getLogger("ldg_partitioner")
    log.info("Writing node partition to: %s", output_file)
    t0 = time.perf_counter()

    with open(output_file, "w") as f:
        for part_id in partitions.tolist():
            f.write(f"{part_id}\n")

    elapsed  = time.perf_counter() - t0
    out_size = os.path.getsize(output_file)
    log.info("Output written in %.3f s  (%.2f MB)", elapsed, out_size / 1_048_576)


#    Metrics

def compute_metrics(
    graph      : dgl.DGLGraph,
    partitions : torch.Tensor,
    num_parts  : int,
    dataset    : str,
    core_time  : float,
) -> Dict:
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    src, dst  = graph.edges()

    counts      = Counter(partitions.tolist())
    node_counts = [counts.get(pid, 0) for pid in range(num_parts)]
    ideal_nodes = num_nodes / num_parts if num_parts else 1

    edge_counts = [
        int((partitions[src] == pid).sum().item()) for pid in range(num_parts)
    ]
    ideal_edges = num_edges / num_parts if num_parts else 1

    node_balance = max(node_counts) / ideal_nodes if ideal_nodes else math.nan
    edge_balance = max(edge_counts) / ideal_edges if ideal_edges else math.nan

    edge_cut = (
        int((partitions[src] != partitions[dst]).sum().item())
        if num_edges > 0 else 0
    )
    edge_cut_ratio = edge_cut / num_edges if num_edges else math.nan

    sizes            = np.array(node_counts, dtype=float)
    node_balance_cov = (
        float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else math.nan
    )

    return {
        "algorithm"         : "ldg",
        "dataset"           : dataset,
        "k"                 : num_parts,
        "partitioning_type" : "node",
        "num_parts"         : num_parts,
        "num_nodes"         : num_nodes,
        "num_edges"         : num_edges,
        "node_counts"       : node_counts,
        "edge_counts"       : edge_counts,
        "edge_cut"          : edge_cut,
        "edge_cut_ratio"    : edge_cut_ratio,
        "node_balance"      : node_balance,
        "edge_balance"      : edge_balance,
        "node_balance_cov"  : node_balance_cov,
        "core_time"         : core_time,
    }


#    CLI

def main() -> None:
    main_start = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="LDG node partitioner for DGL graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--graph",   type=str,   required=True,
                        help="Path to the DGL graph file (.dgl / .bin)")
    parser.add_argument("--parts",   type=int,   required=True,
                        help="Number of partitions (>= 1)")
    parser.add_argument("--output",  type=str,   required=True,
                        help="Output file path (one partition-id per line)")
    parser.add_argument("--slack",   type=float, default=0.0,
                        help="Balance slack s: capacity = (n/k)*(1+s)  (default: 0.0)")
    parser.add_argument("--name",    type=str,   default=None,
                        help="Dataset/graph name for metrics "
                             "(default: graph filename stem)")
    parser.add_argument("--metrics", type=str,   default=None,
                        help="Path to write metrics JSON "
                             "(default: <o>.metrics.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level output (per-node assignments)")
    args = parser.parse_args()

    log = setup_logger(args.verbose)

    if args.parts < 1:
        log.error("--parts must be >= 1, got %d", args.parts)
        sys.exit(1)

    dataset      = args.name or os.path.splitext(os.path.basename(args.graph))[0]
    output_dir   = os.path.dirname(os.path.abspath(args.output))
    metrics_path = args.metrics or os.path.join(output_dir, "metrics.json")

    log.info("LDG node partitioner")
    log.info("  Graph   : %s", args.graph)
    log.info("  Parts   : %d", args.parts)
    log.info("  Slack   : %.4f", args.slack)
    log.info("  Output  : %s", args.output)
    log.info("  Metrics : %s", metrics_path)

    graph    = load_graph(args.graph)
    adj_list = build_adjacency(graph)

    num_nodes = graph.num_nodes()

    # Single-partition shortcut
    if args.parts == 1:
        log.info("Only 1 partition requested — assigning all nodes to part 0.")
        partitions = torch.zeros(num_nodes, dtype=torch.long)
    else:
        partitioner = LDGPartitioner(
            num_nodes      = num_nodes,
            num_partitions = args.parts,
            slack          = args.slack,
        )
        vertex_to_partition, _ = partitioner.run(adj_list)

        partitions = torch.tensor(
            [vertex_to_partition[n] for n in range(num_nodes)],
            dtype=torch.long,
        )

    # Post-hoc quality stats on the full graph
    _partition_balance_stats(partitions, args.parts)
    _edge_cut_stats(graph, partitions)

    _write_output(partitions, args.output)
    core_time = time.perf_counter() - main_start

    metrics = compute_metrics(
        graph      = graph,
        partitions = partitions,
        num_parts  = args.parts,
        dataset    = dataset,
        core_time  = core_time,
    )

    print("\n[done]  Partitioning completed successfully.")
    print(
        "CODEx_METRICS "
        f"algorithm=ldg "
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
