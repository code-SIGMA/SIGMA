# SIGMA: A Versatile Streaming Graph Partitioner for Distributed GNN Training

**SIGMA** (Streaming Integrated Graph Partitioning with Multi-objective Awareness) is a unified streaming graph partitioner for distributed Graph Neural Network (GNN) training. It supports both **vertex partitioning** (minimizing edge cut) and **edge partitioning** (minimizing vertex replication) within a single framework, while simultaneously enforcing balance constraints on both vertices and edges.

---

## Problem

Distributed GNN training distributes a graph across multiple compute workers. The partitioning strategy directly determines three interconnected system-level properties:

- **Communication overhead**: edge cut drives embedding synchronization in vertex partitioning; replication factor drives it in edge partitioning.
- **Memory balance**: uneven vertex assignments cause out-of-memory failures on individual GPU/CPU workers.
- **Compute balance**: edge imbalance creates straggler workers that delay synchronization across the cluster.

These three objectives are coupled: enforcing strict vertex balance to satisfy memory constraints may increase edge cut, raising communication overhead. Existing partitioners address only a subset of this space: Fennel minimizes edge cut under a vertex balance constraint; HDRF minimizes replication under an edge balance constraint. Neither enforces the simultaneous vertex-and-edge balance that GNN workloads require, and the literature treats vertex and edge partitioning as entirely separate problems.

SIGMA addresses this with a single framework that:
- Enforces **both** vertex balance and edge-volume balance simultaneously.
- Supports **both** vertex and edge partitioning modes.
- Optimizes **both** edge cut and replication factor simultaneously in a single scoring function.
- Operates in a **streaming** fashion: the graph is read once and each vertex or edge is assigned online.

---

## Algorithm

The algorithm runs in up to three phases.

### Phase 1: Streaming Clustering (CluStRE)

A modularity-based streaming clustering pass assigns each vertex to a cluster as it arrives. Optional refinement stages (restreaming, local search, or in-memory VieClus) improve cluster quality. Per-cluster capacity bounds equal the partition capacity, preventing any cluster from growing too large to fit in a single block.

### Phase 2: Cluster-to-Block Mapping

Clusters are sorted by decreasing volume and greedily assigned to the least-loaded block (Graham's sorted list scheduling, a 4/3-approximation of the optimal makespan). A second graph pass derives conservative pre-assignments: a vertex is committed to its cluster's block only if all pre-assigned neighbors agree on the same block and capacity is not exceeded. Unresolved vertices fall through to Phase 3.

### Phase 3: One-Pass Streaming Partitioning

Each vertex or edge is scored against all **admissible** blocks, i.e., those where accepting the assignment would not violate any capacity bound. The assignment rule follows `S = affinity − penalty`.

**Vertex partitioning - Multi-Objective (`fennel_mult_obj`):**
```
S_MO(v, p) = e(v,p) / deg(v)  −  ρ_p^(γ−1.1)  −  τ · R(v,p) / (deg(v) + k)
```
where `R(v,p)` estimates the additional vertex-to-block incidences (replications) that assigning `v` to `p` would introduce. Defaults: `γ = 2.5`, `τ = 0.5`.

**Edge partitioning - Modified HDRF:**
```
S_edge(u,v,p) = g_u(p) + g_v(p)  +  λ · (0.5·b_edge_p + 0.5·b_rep_p)
```
where `g_u(p)` is a degree-weighted affinity for endpoint `u` already present in `p`, `b_edge_p` is the edge-load slack of `p`, and `b_rep_p` is the replica-load slack. This extends HDRF by discouraging uneven accumulation of vertex replicas.

**Dynamic capacity scaling** tightens block bounds early in the stream and relaxes them to full capacity as the stream progresses, preventing greedy over-filling of blocks at the start:
```
σ(t) = σ_min + (1 − σ_min) · t^0.5
```

---

## Results

Evaluated on six benchmark graphs (Amazon Computers, Flickr, Twitch, ogbn-arxiv, Reddit, ogbn-products) using two distributed GNN training systems: **DistGNN** (edge partitioning, full-batch) and **DistDGL** (vertex partitioning, mini-batch).

### Edge Partitioning (DistGNN)

- **Replication factor:** SIGMA achieves the lowest replication factor among all streaming partitioners. At `k=32` on Reddit, SIGMA reduces replication factor by ~87% vs. Random and ~82% vs. HDRF. Relative to HeiStreamE (the strongest streaming baseline), SIGMA reduces replication factor by ~25%.
- **Balance:** Vertex balance 1.00–1.53; edge balance consistently within the configured 1.10 tolerance. Substantially outperforms HeiStreamE and 2PS on larger graphs.
- **Training time:** Fastest streaming partitioner on every evaluated dataset. On Twitch, SIGMA reduces training time by 25–62% vs. competing streaming approaches.
- **Memory:** Lowest peak RAM on Flickr and Twitch; competitive with FSM and HEP on larger graphs.

### Vertex Partitioning (DistDGL)

- **Edge-cut ratio:** Competitive with streaming baselines; reliably outperforms Random and LDG. On Flickr at `k=32`, SIGMA achieves 0.642 vs. 0.663 (Cuttana) and 0.680 (Fennel).
- **Vertex balance:** 1.00–1.09 across all datasets and partition counts. Reduces imbalance by ~55% vs. Fennel on Flickr at `k=32`.
- **Edge balance:** 1.01–1.18 across all settings; ranks first or near-first on every dataset.
- **Training time:** Matches or outperforms streaming baselines (Fennel, LDG); remains competitive with in-memory methods (KaHIP, METIS). Avoids out-of-memory failures encountered by Fennel on ogbn-products.

SIGMA consistently delivers robust performance across all metrics without major weaknesses in any single category, outperforming existing streaming baselines while remaining competitive with significantly more expensive in-memory partitioners such as METIS, KaHIP, and HEP.

---

## Build

```bash
bash compile.sh
```

Requires MPI and OpenMP. External dependencies (KaGen, VieClus, Abseil, FlatBuffers, STXXL, robin-hood-hashing, argtable3) are fetched automatically by CMake. The binary is placed at `deploy/sigma`.

To rebuild without cleaning:
```bash
cd build && make
```

---

## Usage

```bash
./deploy/sigma <graph_file> -k <num_partitions> [options]
```

Key options:

| Option | Description |
|---|---|
| `-k <int>` | Number of partitions |
| `--partitioning_type node\|edge` | Vertex or edge partitioning mode |
| `--one_pass_algorithm \|fennel_mult_obj\|hdrf` | Scoring function |
| `--prepartitioner clustre` | Enable CluStRE preprocessing |
| `--mode light_plus\|light\|evo\|standard\|strong` | CluStRE refinement intensity |
| `--epsilon_node <float>` | Vertex balance tolerance ε |
| `--epsilon_edge <float>` | Edge balance tolerance ε_E |
| `--fennel_gamma <float>` | Penalty exponent γ (default 2.5) |
| `--tau_mult_obj <float>` | Replication penalty weight τ (default 0.5) |
| `--lambda_hdrf <int>` | Balance weight λ for HDRF |

**Vertex partitioning example:**
```bash
./deploy/sigma /path/to/delaunay_n15.graph -k 8 \
  --one_pass_algorithm fennel_mult_obj --partitioning_type node \
  --prepartitioner clustre --mode light_plus \
  --write_results --output_path results/
```

**Edge partitioning example:**
```bash
./deploy/sigma /path/to/delaunay_n15.graph -k 8 \
  --one_pass_algorithm hdrf --partitioning_type edge \
  --prepartitioner clustre --mode light_plus \
  --write_results --output_path results/
```

Example graphs are in `examples/` (e.g., `delaunay_n15.graph`, `rgg_n_2_15_s0.graph`).

---