#!/usr/bin/env python3
"""
graph_info.py – Fast statistics for a DGL graph.

Usage:
    python graph_info.py graph.dgl
    python graph_info.py graph.dgl --threads 16
    python graph_info.py graph.dgl --approx              # Approximation for huge graphs
    python graph_info.py graph.dgl --backend scipy       # Without NetworKit
    python graph_info.py edges.npz                       # Edge list (src, dst, num_nodes), no DGL needed

Backends:
  networkit (default, if installed): C++/OpenMP, uses all cores.
      pip install networkit
  scipy (fallback): vectorized with NumPy/SciPy, parallelized across processes.

Speedups over the NetworkX version:
  * No NetworkX: build the graph directly as a SciPy CSR matrix.
  * Fully vectorized degrees and components.
  * Clustering via triangle counting with sparse matrix multiplication,
    computed in parallel using load-balanced blocks.
  * Exact diameter via iFUB instead of BFS from every node: usually only
    a small percentage of nodes is needed, and these BFS runs are parallel.

Semantics: Treat the graph as undirected and simple (no parallel edges,
no self-loops). Compute the diameter on the largest connected component.
Nodes with degree < 2 have clustering coefficient 0 (as in NetworkX).
"""
import argparse
import math
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

try:
    import networkit as nk
    HAS_NETWORKIT = True
except ImportError:
    HAS_NETWORKIT = False


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        print(f"  [{self.label}: {time.perf_counter() - self.t0:.2f}s]", flush=True)


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class SyncExecutor:
    """Execute tasks immediately in the main process (for 1 thread / small graphs)."""

    def submit(self, fn, *args):
        f = Future()
        f.set_result(fn(*args))
        return f

    def shutdown(self, **kwargs):
        pass


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_edges(path, index):
    if path.endswith(".npz"):
        data = np.load(path)
        return data["src"], data["dst"], int(data["num_nodes"]), len(data["src"])

    import dgl  # import only here: worker processes do not need DGL/Torch

    graphs, _ = dgl.load_graphs(path)
    if not graphs:
        sys.exit(f"Error: '{path}' contains no graphs.")
    if not 0 <= index < len(graphs):
        sys.exit(f"Error: invalid index {index} (file contains {len(graphs)} graph(s)).")
    print(f"File contains {len(graphs)} graph(s), using index {index}.")
    g = graphs[index]
    if len(g.ntypes) > 1 or len(g.canonical_etypes) > 1:
        print("Heterograph detected – converting to a homogeneous graph.")
        g = dgl.to_homogeneous(g)
    src, dst = g.edges()
    return src.cpu().numpy(), dst.cpu().numpy(), g.num_nodes(), g.num_edges()


def build_adjacency(src, dst, n):
    """Symmetric, binary CSR adjacency matrix without self-loops or parallel edges."""
    mask = src != dst
    n_self_loops = int((~mask).sum())
    src, dst = src[mask], dst[mask]
    idx_dtype = np.int32 if n < 2**31 else np.int64
    row = np.concatenate([src, dst]).astype(idx_dtype, copy=False)
    col = np.concatenate([dst, src]).astype(idx_dtype, copy=False)
    A = sp.coo_matrix((np.ones(len(row), dtype=np.int32), (row, col)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1
    return A, n_self_loops


# --------------------------------------------------------------------------- #
# SciPy backend: worker functions (run in separate processes)
# --------------------------------------------------------------------------- #
_WORKER = {}


def _init_worker(indptr, indices, n):
    data = np.ones(len(indices), dtype=np.int32)
    _WORKER["A"] = sp.csr_matrix((data, indices, indptr), shape=(n, n))


def _triangles(nodes):
    A = _WORKER["A"]
    S = A[nodes]
    tri = np.asarray((S @ A).multiply(S).sum(axis=1)).ravel() // 2
    return nodes, tri


def _eccentricities(sources):
    D = shortest_path(_WORKER["A"], method="D", directed=True, unweighted=True, indices=sources)
    D[np.isinf(D)] = -1
    return D.max(axis=1)


def _bfs(A, s):
    d = shortest_path(A, method="D", directed=True, unweighted=True, indices=int(s))
    d[np.isinf(d)] = -1
    return d.astype(np.int64)


def make_executor(A, workers):
    if workers <= 1:
        _init_worker(A.indptr, A.indices, A.shape[0])
        return SyncExecutor()
    ctx = mp.get_context("spawn")  # safe even when Torch/OpenMP is loaded
    return ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                               initializer=_init_worker,
                               initargs=(A.indptr, A.indices, A.shape[0]))


def scipy_clustering(A, deg, executor, workers, approx, trials, rng):
    n = A.shape[0]
    if approx and trials < n:
        nodes = np.sort(rng.choice(n, size=trials, replace=False))
    else:
        nodes = np.arange(n)

    # Balance blocks by computational cost (sum of neighbor degrees)
    work = (A @ deg.astype(np.float64))[nodes] + 1.0
    n_chunks = max(1, min(len(nodes), workers * 16))
    bounds = np.searchsorted(np.cumsum(work), np.linspace(0, work.sum(), n_chunks + 1)[1:-1])
    chunks = [c for c in np.split(nodes, bounds) if len(c)]

    tri = np.zeros(n, dtype=np.int64)
    for f in as_completed([executor.submit(_triangles, c) for c in chunks]):
        idx, t = f.result()
        tri[idx] = t

    d = deg[nodes].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(d > 1, 2.0 * tri[nodes] / (d * (d - 1)), 0.0)
    return float(c.mean())


def scipy_diameter(A, deg, labels, lcc_label, executor, workers, approx):
    """iFUB (Crescenzi et al.) with the 2-sweep midpoint as the starting node."""
    n = A.shape[0]
    lcc_nodes = np.flatnonzero(labels == lcc_label)
    if len(lcc_nodes) <= 1:
        return 0, 0

    u0 = lcc_nodes[np.argmax(deg[lcc_nodes])]
    a = int(np.argmax(_bfs(A, u0)))
    dist_a = _bfs(A, a)
    b = int(np.argmax(dist_a))
    dist_b = _bfs(A, b)
    lb = int(max(dist_a.max(), dist_b.max()))
    bfs_count = 3
    if approx:
        return lb, bfs_count

    h = lb // 2
    mid = np.flatnonzero((dist_a == h) & (dist_b == lb - h))
    u = int(mid[np.argmax(deg[mid])]) if len(mid) else int(u0)
    dist_u = _bfs(A, u)
    bfs_count += 1
    ecc_u = int(dist_u.max())
    lb = max(lb, ecc_u)

    batch = int(np.clip(4_000_000 // n, 1, 256))
    for i in range(ecc_u, 0, -1):
        if lb >= 2 * i:
            break
        level = np.flatnonzero(dist_u == i)
        n_batches = max(math.ceil(len(level) / batch), min(len(level), workers))
        futures = [executor.submit(_eccentricities, part)
                   for part in np.array_split(level, n_batches) if len(part)]
        # The level must generally be evaluated IN FULL: the
        # iFUB bound diameter <= max(lb, 2*(i-1)) holds only once all nodes
        # in the level have been considered. Stopping midway underestimates
        # the diameter.
        # The only valid exception: for every node x at level j <= i,
        # ecc(x) <= d(x,u) + ecc(u) <= i + ecc(u). If lb reaches this bound,
        # the diameter is known and we can stop immediately.
        cap = i + ecc_u
        done = False
        try:
            for f in as_completed(futures):
                lb = max(lb, int(f.result().max()))
                if lb >= cap:
                    done = True
                    break
        finally:
            for f in futures:
                f.cancel()
        bfs_count += len(level)
        if done or lb > 2 * (i - 1):
            break
    return lb, bfs_count


# --------------------------------------------------------------------------- #
# NetworKit backend
# --------------------------------------------------------------------------- #
def to_networkit(A):
    T = sp.triu(A, k=1).tocoo()
    G = nk.Graph(A.shape[0], weighted=False, directed=False)
    G.addEdges((T.row.astype(np.uint64), T.col.astype(np.uint64)))
    return G


def networkit_clustering(G, approx, trials):
    if approx:
        return float(nk.globals.ClusteringCoefficient.approxAvgLocal(G, trials))
    lcc = nk.centrality.LocalClusteringCoefficient(G, turbo=True)
    lcc.run()
    return float(np.mean(lcc.scores()))


def networkit_diameter(A, labels, lcc_label, approx):
    nodes = np.flatnonzero(labels == lcc_label)
    if len(nodes) <= 1:
        return (0, 0)
    G = to_networkit(A[nodes][:, nodes])
    if approx:
        algo = nk.distance.Diameter(G, algo=nk.distance.DiameterAlgo.ESTIMATED_RANGE, error=0.1)
    else:
        algo = nk.distance.Diameter(G, algo=nk.distance.DiameterAlgo.EXACT)
    algo.run()
    return tuple(int(x) for x in algo.getDiameter())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Fast statistics for a DGL graph.")
    p.add_argument("path", help="DGL file (dgl.save_graphs) or .npz with src/dst/num_nodes")
    p.add_argument("--index", type=int, default=0, help="Index of the graph in the file")
    p.add_argument("--threads", type=int, default=available_cpus(), help="Number of threads/processes")
    p.add_argument("--backend", choices=["auto", "networkit", "scipy"], default="auto")
    p.add_argument("--approx", action="store_true", help="Approximate diameter and clustering")
    p.add_argument("--trials", type=int, default=100_000, help="Samples for approximate clustering")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "networkit" if HAS_NETWORKIT else "scipy"
    if backend == "networkit" and not HAS_NETWORKIT:
        sys.exit("NetworKit is not installed: pip install networkit  (or --backend scipy)")

    t_start = time.perf_counter()
    with Timer("Loading"):
        src, dst, n, m_dgl = load_edges(args.path, args.index)
    with Timer("Adjazenzmatrix"):
        A, n_self_loops = build_adjacency(src, dst, n)
        del src, dst
        deg = np.diff(A.indptr)
        m = int(A.nnz // 2)
    with Timer("Komponenten"):
        n_comp, labels = connected_components(A, directed=False)
        comp_sizes = np.bincount(labels) if n else np.array([0])
        lcc_label = int(np.argmax(comp_sizes))
        lcc_size = int(comp_sizes[lcc_label])

    workers = max(1, args.threads)
    print(f"Backend: {backend}, Threads: {workers}", flush=True)
    diam_note = ""

    if backend == "networkit":
        nk.setNumberOfThreads(workers)
        with Timer("Clustering"):
            clustering = networkit_clustering(to_networkit(A), args.approx, args.trials)
        with Timer("Diameter"):
            lo, hi = networkit_diameter(A, labels, lcc_label, args.approx)
        diameter = str(lo) if hi <= lo else f"{lo} – {hi}"  # EXACT returns (d, 0)
    else:
        if n < 20_000:
            workers = 1  # Starting processes is not worthwhile for small graphs
        rng = np.random.default_rng(args.seed)
        executor = make_executor(A, workers)
        try:
            with Timer("Clustering"):
                clustering = scipy_clustering(A, deg, executor, workers, args.approx, args.trials, rng)
            with Timer("Diameter"):
                d, bfs_count = scipy_diameter(A, deg, labels, lcc_label, executor, workers, args.approx)
        finally:
            executor.shutdown(cancel_futures=True)
        diameter = f"≥ {d}" if args.approx else str(d)
        diam_note = f", {bfs_count} BFS statt {lcc_size}"

    approx_tag = " (approximate)" if args.approx else ""
    lcc_note = "" if n_comp <= 1 else f"largest component: {lcc_size} nodes"
    extra = ", ".join(x for x in [lcc_note, diam_note.lstrip(", ")] if x)

    print("\n" + "=" * 60)
    print(" GRAPH STATISTICS")
    print("=" * 60)
    print(f" Nodes                       : {n}")
    print(f" Edges (DGL, directed)       : {m_dgl}")
    print(f" Edges (undirected, simple)  : {m}")
    print(f" Entfernte Self-Loops        : {n_self_loops}")
    print(f" Zusammenhangskomponenten    : {n_comp}")
    print(f" Density                      : {2 * m / (n * (n - 1)):.6f}" if n > 1 else " Density: n/a")
    print("-" * 60)
    print(f" Diameter{approx_tag:<20}: {diameter}" + (f" ({extra})" if extra else ""))
    print(f" Avg. clustering coefficient : {clustering:.6f}{approx_tag}")
    print("-" * 60)
    if n:
        print(f" Mean degree                 : {deg.mean():.4f}")
        print(f" Median degree               : {np.median(deg):.1f}")
        print(f" Std. degree                 : {deg.std():.4f}")
        print(f" Max degree                  : {int(deg.max())}")
        print(f" Min degree                  : {int(deg.min())}")
    print("=" * 60)
    print(f"Gesamtzeit: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
