#!/usr/bin/env python3
"""
SDT_GNN Streaming Graph Partitioner (SPRING, node-based)
=============================================================
Wrapper around SDT_GNN.Partitioning with a CLI, logging and metrics output,
restricted to the SPRING algorithm proposed in the SDT-GNN paper
(excluding the baseline algorithms Hashing/DBH/Greedy/HDRF/2PSL/Clustering).

SPRING is a NODE partitioner: each node is primarily assigned to exactly
one partition. To support GNN neighborhood aggregation across partition
boundaries, each node's K-hop neighbors are also replicated into its
partition (parameter --K). After the run, the induced subgraph's edges
for each partition are stored in <output>/partition_<k>.txt (format:
"u v" per line). The "Replication Factor" measures the node overhead
introduced by K-hop replication (sum of nodes across all partitions /
number of nodes in the original graph).

Metrics (metrics.json / CODEx_METRICS)
---------------------------------------
As with the random partitioner, standard node-partitioning metrics
are reported in addition to replication-related statistics:

  edge_cut               Number of original edges (u, v) with
                         owner(u) != owner(v).
  edge_cut_ratio         edge_cut / number of considered edges
  node_counts_primary    Nodes per partition WITHOUT K-hop replicas
  node_balance_primary   max(node_counts_primary) / (num_nodes / k)
  edge_replication       num_edges_partitioned / num_edges
  assignment_source      Source of the node-to-partition assignment
  edge_cut_exact         True  = computed from an actual assignment
                         False = derived from the partition files

Where does the assignment come from? (in this order)
--------------------------------------------------
  1. --assignment FILE            Explicitly specified file
                                  (one partition ID per line, line i
                                  = node i; or "node part" per line;
                                  also .npy/.pkl/.json).
  2. Partitioner object          Generic scan of all attributes of
                                  Partitioning and its nested objects
                                  (2 levels) for a plausible assignment:
                                  dict node->part, array of length
                                  num_nodes with values in [0, k),
                                  or k lists of nodes.
  3. Files in output directory   Generic scan of *.txt/*.csv/*.npy/
                                  *.pkl/*.json using the same patterns.
  4. Derivation (fallback)       Majority vote over edges:
                                  a node belongs to the partition whose
                                  partition_<k>.txt contains most of
                                  its incident edges.
                                  -> edge_cut_exact = False.
                                  Controlled by --derive_assignment.

With --verbose, the scan logs which attributes/files were checked and
why they were rejected. This is the fastest way to find the correct
attribute name for a particular SDT_GNN version and then hard-code it.

Usage:
  python partition.py \
      --graph /path/to/project \
      --dataset reddit \
      --parts 4 \
      --output /path/to/partitions/reddit/node/4/raw/

  # Specify the assignment explicitly
  python partition.py ... --assignment /path/to/node_partition.txt

  # Disable majority-vote derivation (edge_cut remains null)
  python partition.py ... --derive_assignment never
"""

import io
import os
import re
import sys
import time
import json
import math
import glob
import types
import pickle
import argparse
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:  # pandas is optional, only used for speed
    pd = None

import SDT_GNN


CHUNK_SIZE = 5_000_000


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
#  Fast edge-list readers (pandas if available, otherwise pure Python)
# ---------------------------------------------------------------------------

def _py_edge_chunks(path: str, delimiter: Optional[str], chunksize: int) -> Iterator[np.ndarray]:
    buf: List[Tuple[int, int]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter) if delimiter else line.split()
            if len(parts) < 2:
                continue
            buf.append((int(parts[0]), int(parts[1])))
            if len(buf) >= chunksize:
                yield np.asarray(buf, dtype=np.int64)
                buf = []
    if buf:
        yield np.asarray(buf, dtype=np.int64)


def edge_chunks(
    path: str,
    delimiter: Optional[str] = None,
    chunksize: int = CHUNK_SIZE,
) -> Iterator[np.ndarray]:
    """Yield the edge list in (n, 2) int64 blocks.

    delimiter=None  -> whitespace-separated (partition_<k>.txt)
    delimiter=","   -> edge_list.csv
    """
    if pd is None:
        yield from _py_edge_chunks(path, delimiter, chunksize)
        return

    kwargs = dict(
        header=None,
        usecols=[0, 1],
        names=["u", "v"],
        dtype=np.int64,
        chunksize=chunksize,
        comment="#",
    )
    try:
        if delimiter:
            reader = pd.read_csv(path, sep=delimiter, engine="c", **kwargs)
        else:
            try:
                reader = pd.read_csv(path, sep=r"\s+", engine="c", **kwargs)
            except (TypeError, ValueError):
                reader = pd.read_csv(path, delim_whitespace=True, **kwargs)
    except Exception:  # noqa: BLE001 - any parser error -> Python fallback
        yield from _py_edge_chunks(path, delimiter, chunksize)
        return

    for chunk in reader:
        arr = np.ascontiguousarray(chunk.to_numpy(dtype=np.int64))
        if arr.size:
            yield arr


def read_edges(path: str, delimiter: Optional[str] = None) -> np.ndarray:
    """Read a complete edge list as an (n, 2) int64 array."""
    chunks = [c for c in edge_chunks(path, delimiter) if c.size]
    if not chunks:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
#  Determine the original graph size (fallback if sp.number_nodes/
#  sp.number_edges are not set)
# ---------------------------------------------------------------------------

def count_original_graph(
    dataset_path: str, dataset: str, log: logging.Logger
) -> Tuple[Optional[int], Optional[int]]:
    """Count nodes/edges in edge_list.csv, as in Partitioner.get_degree()."""
    edge_list_file = os.path.join(dataset_path, dataset, "edge_list.csv")
    if not os.path.exists(edge_list_file):
        log.warning("edge_list.csv not found at %s", edge_list_file)
        return None, None

    max_node = -1
    num_edges = 0
    for chunk in edge_chunks(edge_list_file, delimiter=","):
        max_node = max(max_node, int(chunk.max()))
        num_edges += int(chunk.shape[0])

    return max_node + 1, num_edges


# ---------------------------------------------------------------------------
#  Read partition edge lists
# ---------------------------------------------------------------------------

def partition_file(output_path: str, k: int) -> str:
    return os.path.join(output_path, f"partition_{k}.txt")


def load_partition_edges(output_path: str, k: int) -> np.ndarray:
    """Read partition_<k>.txt (induced subgraph including K-hop replicas,
    format 'u v' per line) as an (n, 2) int array."""
    file_path = partition_file(output_path, k)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Partition file not found: {file_path}")
    return read_edges(file_path)


# ---------------------------------------------------------------------------
#  Normalize candidates to an assignment array and validate
# ---------------------------------------------------------------------------

MIN_COVERAGE = 0.5  # Fraction of nodes that must be assigned

#  Objects that MUST NOT be touched during the scan.
_UNSAFE_TYPES = (
    io.IOBase,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.GeneratorType,
    logging.Logger,
    logging.Handler,
    type,
)

#  Attributes indicating a file/stream/connection object.
_UNSAFE_ATTRS = ("read", "write", "fileno", "close", "seek", "flush", "execute")


def is_safe_candidate(value: Any) -> bool:
    """True if the object can safely be interpreted as an assignment.

    Excluded types:
      * File/stream objects — list() raises an error for closed handles:
        'I/O operation on closed file'; for open handles, it consumes the file.
      * Iterators/generators (including csv.reader) — iterating consumes them
        and would destroy the partitioner state.
      * Modules, classes, functions, loggers.
    """
    if isinstance(value, _UNSAFE_TYPES):
        return False
    if isinstance(value, (np.ndarray, dict, list, tuple, set, frozenset)):
        return True
    if hasattr(value, "__next__"):  # Iterator / Generator / csv.reader
        return False
    if any(hasattr(value, attr) for attr in _UNSAFE_ATTRS):
        return False
    return True


def _finalise(arr: Optional[np.ndarray], num_nodes: int, num_parts: int) -> Optional[np.ndarray]:
    """Check whether a candidate array plausibly represents an assignment."""
    if arr is None or arr.size != num_nodes:
        return None
    assigned = arr >= 0
    coverage = assigned.sum() / num_nodes if num_nodes else 0.0
    if coverage < MIN_COVERAGE:
        return None
    if assigned.any() and int(arr[assigned].max()) >= num_parts:
        return None
    return arr


def as_assignment_array(raw: Any, num_nodes: int, num_parts: int) -> Optional[np.ndarray]:
    """Normalize different representations to an int64 array of
    length num_nodes (value = partition ID, -1 = unknown).

    Recognized representations:
      * dict  node -> part
      * flat array/list of length num_nodes (part per node)
      * (n, 2) array with columns (node, part)
      * sequence of num_parts node lists/sets
    """
    if raw is None or not num_nodes or not is_safe_candidate(raw):
        return None

    # --- dict: node -> part -------------------------------------------------
    if isinstance(raw, dict):
        if not raw:
            return None
        try:
            keys = np.fromiter((int(k) for k in raw.keys()), dtype=np.int64, count=len(raw))
            vals = np.fromiter((int(v) for v in raw.values()), dtype=np.int64, count=len(raw))
        except (TypeError, ValueError):
            return None
        arr = np.full(num_nodes, -1, dtype=np.int64)
        mask = (keys >= 0) & (keys < num_nodes)
        arr[keys[mask]] = vals[mask]
        return _finalise(arr, num_nodes, num_parts)

    # --- torch.Tensor -------------------------------------------------------
    if hasattr(raw, "cpu") and hasattr(raw, "numpy"):
        try:
            raw = raw.cpu().numpy()
        except Exception:  # noqa: BLE001
            return None

    # --- numpy ---------------------------------------------------------------
    if isinstance(raw, np.ndarray):
        if raw.dtype == object:
            raw = list(raw)
        elif raw.ndim == 1:
            try:
                return _finalise(raw.astype(np.int64), num_nodes, num_parts)
            except (TypeError, ValueError):
                return None
        elif raw.ndim == 2 and raw.shape[1] == 2 and raw.shape[0] >= MIN_COVERAGE * num_nodes:
            arr = np.full(num_nodes, -1, dtype=np.int64)
            nodes = raw[:, 0].astype(np.int64)
            parts = raw[:, 1].astype(np.int64)
            mask = (nodes >= 0) & (nodes < num_nodes)
            arr[nodes[mask]] = parts[mask]
            return _finalise(arr, num_nodes, num_parts)
        else:
            return None

    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__iter__"):
        return None

    try:
        seq = list(raw)
    except Exception:  # noqa: BLE001 - simply skip any unusual object
        return None
    if not seq:
        return None

    first = seq[0]

    # --- Sequence of node lists: index == partition ID -------------------
    if isinstance(first, (list, tuple, set, frozenset, np.ndarray)):
        if len(seq) != num_parts:
            return None
        arr = np.full(num_nodes, -1, dtype=np.int64)
        for pid, nodes in enumerate(seq):
            try:
                idx = np.asarray(list(nodes), dtype=np.int64).ravel()
            except (TypeError, ValueError):
                return None
            idx = idx[(idx >= 0) & (idx < num_nodes)]
            arr[idx] = pid
        return _finalise(arr, num_nodes, num_parts)

    # --- flat sequence -----------------------------------------------------
    try:
        arr = np.asarray(seq).astype(np.int64).ravel()
    except (TypeError, ValueError):
        return None
    return _finalise(arr, num_nodes, num_parts)


# ---------------------------------------------------------------------------
#  Source 1: explicit file (--assignment)
# ---------------------------------------------------------------------------

def read_column_or_edges(path: str, delimiter: Optional[str]) -> np.ndarray:
    """Read a text file as a 1D array (one column) or an (n, 2) array."""
    cols = 0
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                cols = len(line.split(delimiter) if delimiter else line.split())
                break
    if cols == 0:
        return np.empty(0, dtype=np.int64)
    if cols == 1:
        return np.loadtxt(path, dtype=np.int64, ndmin=1)
    return read_edges(path, delimiter)


def _load_any(path: str, log: logging.Logger) -> Any:
    """Load txt/csv/npy/pkl/json as tolerantly as possible."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            return np.load(path, allow_pickle=True)
        if ext in (".pkl", ".pickle"):
            with open(path, "rb") as fh:
                return pickle.load(fh)
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return read_column_or_edges(path, "," if ext == ".csv" else None)
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not load %s: %s", path, exc)
        return None


def assignment_from_file(
    path: str, num_nodes: int, num_parts: int, log: logging.Logger
) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        log.error("--assignment: file not found: %s", path)
        return None

    arr = as_assignment_array(_load_any(path, log), num_nodes, num_parts)
    if arr is None:
        log.error(
            "--assignment: %s could not be interpreted as an assignment for %d nodes / "
            "%d partitions.", path, num_nodes, num_parts,
        )
    return arr


# ---------------------------------------------------------------------------
#  Source 2: generic scan of the partitioner object
# ---------------------------------------------------------------------------

def scan_object_for_assignment(
    obj: Any,
    num_nodes: int,
    num_parts: int,
    log: logging.Logger,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 2,
    seen: Optional[set] = None,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Recursively scan an object for a plausible node assignment."""
    if obj is None or depth > max_depth:
        return None, None
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return None, None
    seen.add(id(obj))

    try:
        attrs = sorted(vars(obj).keys())
    except TypeError:
        return None, None

    nested: List[Tuple[str, Any]] = []

    for name in attrs:
        if name.startswith("__"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001
            continue
        if value is None or callable(value) or isinstance(value, (str, bytes, int, float, bool)):
            continue
        if not is_safe_candidate(value):
            log.debug("  skipped: %s%-24s type=%s (file/iterator)",
                      prefix, name, type(value).__name__)
            continue

        try:
            arr = as_assignment_array(value, num_nodes, num_parts)
        except Exception as exc:  # noqa: BLE001 - The scan must never abort the run
            log.debug("  Error at %s%s: %s", prefix, name, exc)
            arr = None
        if arr is not None:
            return arr, f"attribute:{prefix}{name}"

        try:
            size = len(value)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            size = -1
        log.debug("  rejected: %s%-24s type=%-16s len=%s",
                  prefix, name, type(value).__name__, size)

        if hasattr(value, "__dict__"):
            nested.append((name, value))

    for name, value in nested:
        arr, source = scan_object_for_assignment(
            value, num_nodes, num_parts, log,
            prefix=f"{prefix}{name}.", depth=depth + 1, max_depth=max_depth, seen=seen,
        )
        if arr is not None:
            return arr, source

    return None, None


# ---------------------------------------------------------------------------
#  Source 3: generic scan of files in the output directory
# ---------------------------------------------------------------------------

_NODE_FILE_PATTERNS: Tuple[str, ...] = (
    "partition_nodes_{k}.txt",
    "nodes_{k}.txt",
    "partition_{k}_nodes.txt",
    "partition_node_{k}.txt",
)


def scan_files_for_assignment(
    output_path: str, num_nodes: int, num_parts: int, log: logging.Logger
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    # a) Node lists per partition
    for pattern in _NODE_FILE_PATTERNS:
        paths = [os.path.join(output_path, pattern.format(k=k)) for k in range(num_parts)]
        if not all(os.path.exists(p) for p in paths):
            continue
        arr = np.full(num_nodes, -1, dtype=np.int64)
        for pid, p in enumerate(paths):
            nodes = np.loadtxt(p, dtype=np.int64, ndmin=1).ravel()
            nodes = nodes[(nodes >= 0) & (nodes < num_nodes)]
            arr[nodes] = pid
        arr = _finalise(arr, num_nodes, num_parts)
        if arr is not None:
            return arr, f"files:{pattern}"

    # b) Single file containing an assignment
    candidates: List[str] = []
    for ext in ("txt", "csv", "npy", "pkl", "pickle", "json"):
        candidates.extend(sorted(glob.glob(os.path.join(output_path, f"*.{ext}"))))

    for p in candidates:
        base = os.path.basename(p)
        if re.fullmatch(r"partition_\d+\.txt", base) or base == "metrics.json":
            continue
        arr = as_assignment_array(_load_any(p, log), num_nodes, num_parts)
        if arr is not None:
            return arr, f"file:{base}"
        log.debug("  rejected: file %s", base)

    return None, None


# ---------------------------------------------------------------------------
#  Source 4: derivation by majority vote over partition edges
# ---------------------------------------------------------------------------

def derive_assignment_from_partitions(
    output_path: str, num_parts: int, num_nodes: int, log: logging.Logger
) -> Optional[np.ndarray]:
    """Assign each node to the partition containing most of its
    incident edges (ties -> smallest partition ID).

    Approximation: exact as long as the partitioner writes a node's edges
    mostly to its home partition. The resulting
    edge_cut is marked with edge_cut_exact=False.
    """
    counts = np.zeros((num_nodes, num_parts), dtype=np.int64)
    seen_any = False
    t0 = time.perf_counter()

    for k in range(num_parts):
        p = partition_file(output_path, k)
        if not os.path.exists(p):
            log.warning("Derivation aborted: %s is missing.", p)
            return None
        for chunk in edge_chunks(p):
            seen_any = True
            nodes = chunk.ravel()
            nodes = nodes[(nodes >= 0) & (nodes < num_nodes)]
            counts[:, k] += np.bincount(nodes, minlength=num_nodes)[:num_nodes]

    if not seen_any:
        return None

    arr = counts.argmax(axis=1).astype(np.int64)
    arr[counts.sum(axis=1) == 0] = -1  # isolated nodes remain unknown
    log.debug("Assignment derived in %.2f s", time.perf_counter() - t0)
    return _finalise(arr, num_nodes, num_parts)


# ---------------------------------------------------------------------------
#  Determine the assignment (orchestration)
# ---------------------------------------------------------------------------

def resolve_node_assignment(
    partitioner: Any,
    output_path: str,
    num_parts: int,
    num_nodes: Optional[int],
    log: logging.Logger,
    assignment_file: Optional[str] = None,
    derive: str = "auto",
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if not num_nodes:
        log.warning("num_nodes unknown — node assignment will not be determined.")
        return None, None

    if assignment_file:
        arr = assignment_from_file(assignment_file, num_nodes, num_parts, log)
        if arr is not None:
            log.info("Node assignment read from --assignment: %s", assignment_file)
            return arr, f"file:{os.path.basename(assignment_file)}"

    if derive != "always":
        log.debug("Scanning partitioner object for node assignment ...")
        arr, source = scan_object_for_assignment(partitioner, num_nodes, num_parts, log)
        if arr is not None:
            log.info("Node assignment found: %s", source)
            return arr, source

        log.debug("Scanning output directory for node assignment ...")
        arr, source = scan_files_for_assignment(output_path, num_nodes, num_parts, log)
        if arr is not None:
            log.info("Node assignment found: %s", source)
            return arr, source

    if derive == "never":
        log.warning(
            "No node assignment found and --derive_assignment=never — "
            "edge_cut remains undetermined."
        )
        return None, None

    log.warning(
        "No explicit node assignment found — deriving it by "
        "majority vote over partition edges (approximation). "
        "Use --verbose to see which attributes were checked."
    )
    arr = derive_assignment_from_partitions(output_path, num_parts, num_nodes, log)
    if arr is None:
        log.warning("Derivation failed — edge_cut remains undetermined.")
        return None, None
    return arr, "derived:majority_edge_vote"


# ---------------------------------------------------------------------------
#  Edge cut on the original graph
# ---------------------------------------------------------------------------

def compute_edge_cut(
    dataset_path: str, dataset: str, assignment: np.ndarray, log: logging.Logger
) -> Tuple[Optional[int], Optional[int]]:
    """Count edges with owner(u) != owner(v), streamed from edge_list.csv."""
    edge_list_file = os.path.join(dataset_path, dataset, "edge_list.csv")
    if not os.path.exists(edge_list_file):
        log.warning("edge_list.csv not found at %s — skipping edge_cut.",
                    edge_list_file)
        return None, None

    n = assignment.size
    cut = 0
    considered = 0
    skipped = 0
    t0 = time.perf_counter()

    for chunk in edge_chunks(edge_list_file, delimiter=","):
        u = chunk[:, 0]
        v = chunk[:, 1]
        valid = (u >= 0) & (u < n) & (v >= 0) & (v < n)
        pu = assignment[u[valid]]
        pv = assignment[v[valid]]
        known = (pu >= 0) & (pv >= 0)
        n_known = int(known.sum())
        considered += n_known
        skipped += int(chunk.shape[0]) - n_known
        cut += int((pu[known] != pv[known]).sum())

    log.debug("edge_cut computed in %.2f s", time.perf_counter() - t0)
    if skipped:
        log.warning("edge_cut: skipped %d edges (nodes without a valid assignment).", skipped)

    return cut, considered


# ---------------------------------------------------------------------------
#  Balance statistics
# ---------------------------------------------------------------------------

def _log_balance_stats(counts: Sequence[int], label: str, log: logging.Logger) -> None:
    counts = list(counts)
    total = sum(counts)
    num_parts = len(counts)

    log.info("─── %s partition balance ────────────────────────────────", label)
    for pid, n in enumerate(counts):
        pct = 100.0 * n / total if total else 0.0
        bar = "█" * int(pct / 2)
        log.info("  Part %3d : %6d  (%5.1f %%)  %s", pid, n, pct, bar)

    ideal = total / num_parts if num_parts else 0.0
    imbalance = (max(counts) - min(counts)) / ideal * 100 if ideal else 0.0
    log.info("  Imbalance (max-min)/ideal : %.1f %%", imbalance)
    log.info("─────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
#  Formatting
# ---------------------------------------------------------------------------

def _fmt(value: Any, spec: str = ".9f") -> str:
    """Format a metric value; None/NaN become 'nan'."""
    if value is None:
        return "nan"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def _json_safe(value: Any) -> Any:
    """NaN -> None, numpy types -> Python types (valid JSON)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
#  Metrics (node partitioning with K-hop replication, SPRING)
# ---------------------------------------------------------------------------

def compute_partition_metrics(
    output_path: str,
    number_partition: int,
    dataset: str,
    method: str,
    core_time: float,
    total_num_nodes: Optional[int],
    total_num_edges: Optional[int],
    log: logging.Logger,
    assignment: Optional[np.ndarray] = None,
    assignment_source: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> Dict:
    node_counts: List[int] = []
    edge_counts: List[int] = []

    for k in range(number_partition):
        edges = load_partition_edges(output_path, k)
        edge_counts.append(int(edges.shape[0]))
        nodes = np.unique(edges) if edges.size else np.empty(0, dtype=np.int64)
        node_counts.append(int(nodes.size))

    num_edges_partitioned = sum(edge_counts)
    num_nodes_replicated = sum(node_counts)

    if not total_num_nodes:
        log.warning(
            "Total node count of the original graph unknown — "
            "Replication Factor cannot be computed."
        )
        replication_factor = math.nan
    else:
        replication_factor = num_nodes_replicated / total_num_nodes

    edge_balance = (
        max(edge_counts) / (num_edges_partitioned / number_partition)
        if num_edges_partitioned else math.nan
    )
    node_balance = (
        max(node_counts) / (num_nodes_replicated / number_partition)
        if num_nodes_replicated else math.nan
    )
    edge_replication = (
        num_edges_partitioned / total_num_edges if total_num_edges else math.nan
    )

    # --- Primary (non-replicated) node distribution ----------------------
    node_counts_primary: Optional[List[int]] = None
    node_balance_primary: float = math.nan
    if assignment is not None:
        valid = assignment[assignment >= 0]
        node_counts_primary = (
            np.bincount(valid, minlength=number_partition)[:number_partition]
            .astype(int).tolist()
        )
        num_primary = sum(node_counts_primary)
        node_balance_primary = (
            max(node_counts_primary) / (num_primary / number_partition)
            if num_primary else math.nan
        )

    # --- Edge cut on the original graph ----------------------------------
    edge_cut: Optional[int] = None
    edge_cut_ratio: float = math.nan
    considered_edges: Optional[int] = None
    if assignment is not None and dataset_path is not None:
        edge_cut, considered_edges = compute_edge_cut(dataset_path, dataset, assignment, log)
        if edge_cut is not None and considered_edges:
            edge_cut_ratio = edge_cut / considered_edges

    edge_cut_exact = (
        None if edge_cut is None
        else not str(assignment_source or "").startswith("derived:")
    )

    if total_num_edges is not None and num_edges_partitioned != total_num_edges:
        log.info(
            "Sum of edges across partitions (%d) differs from the "
            "original edge count (%d) — expected with K-hop replication "
            "(edge_replication = %.6f).",
            num_edges_partitioned, total_num_edges, edge_replication,
        )

    return {
        "algorithm": method,
        "dataset": dataset,
        "k": number_partition,
        "partitioning_type": "node",
        "num_parts": number_partition,
        "num_nodes": total_num_nodes,
        "num_edges": total_num_edges if total_num_edges is not None else num_edges_partitioned,
        "num_edges_partitioned": num_edges_partitioned,
        "num_edges_considered": considered_edges,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "node_counts_primary": node_counts_primary,
        "node_balance": node_balance,
        "node_balance_primary": node_balance_primary,
        "edge_balance": edge_balance,
        "edge_cut": edge_cut,
        "edge_cut_ratio": edge_cut_ratio,
        "edge_cut_exact": edge_cut_exact,
        "assignment_source": assignment_source,
        "replication_factor": replication_factor,
        "edge_replication": edge_replication,
        "core_time": core_time,
    }


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

METHOD = "SPRING"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDT_GNN Streaming Graph Partitioner (SPRING, node-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--graph", type=str, required=True,
                        help="Project root path (contains 'datasets/', passed to SDT_GNN as 'path')")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g. 'cora'")
    parser.add_argument("--parts", type=int, required=True,
                        help="Number of partitions (>= 1)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for partitions and metrics.json")
    parser.add_argument("--multilabel", action="store_true",
                        help="Set if the dataset is multilabel (default: False)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed, passed directly to SDT_GNN.Partitioning (default: 42)")
    parser.add_argument("--K", type=int, default=1,
                        help="Number of neighbor hops retained after partitioning (default: 1)")
    parser.add_argument("--T", type=float, default=None,
                        help="Memory budget for GNN training in GB "
                             "(default: None, i.e. 2/3 of available GPU memory)")
    parser.add_argument("--assignment", type=str, default=None,
                        help="File containing the primary node->partition assignment "
                             "(one partition ID per line, 'node part' per line, "
                             "or .npy/.pkl/.json). Overrides automatic detection.")
    parser.add_argument("--derive_assignment", choices=["auto", "never", "always"], default="auto",
                        help="Derive a fallback assignment by majority vote over "
                             "partition edges: 'auto' only if nothing was found (default), "
                             "'never' disables derivation (edge_cut remains null), 'always' always derives.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging (shows the attribute/file scan)")
    parser.add_argument("--no_partition_features_file", action="store_false",
                        dest="partition_features_file",
                        help="Do not write a separate features file per partition (enabled by default)")
    parser.add_argument("--no_print_statistics", action="store_false",
                        dest="print_partition_statistics",
                        help="Do not print SDT_GNN partition statistics (enabled by default)")
    parser.add_argument("--no_save_dgl_graph", action="store_false",
                        dest="save_dgl_graph",
                        help="Do not save a DGL graph, only txt edge lists (enabled by default)")
    parser.set_defaults(
        partition_features_file=True,
        print_partition_statistics=True,
        save_dgl_graph=True,
    )
    return parser.parse_args()


def main() -> None:
    main_start = time.perf_counter()

    args = parse_args()
    log = setup_logger(args.verbose)

    number_partition = args.parts
    if number_partition < 1:
        log.error("--parts must be >= 1, got: %d", number_partition)
        sys.exit(1)

    if pd is None:
        log.debug("pandas unavailable — using the slower Python reader.")

    path = os.path.abspath(args.graph)
    dataset_path = os.path.join(path, "datasets") + os.sep
    output_path = os.path.abspath(args.output) + os.sep
    os.makedirs(output_path, exist_ok=True)
    metrics_path = os.path.join(output_path, "metrics.json")

    log.info("Dataset: %s  |  Method: %s  |  Parts: %s", args.dataset, METHOD, args.parts)
    log.info("Path   : %s", path)
    log.info("Output : %s", output_path)
    log.info("Metrics: %s", metrics_path)

    partitioner = SDT_GNN.Partitioning(
        dataset=args.dataset,
        multilabel=args.multilabel,
        number_partition=number_partition,
        seed=args.seed,
        path=dataset_path,
        method=METHOD,
        output_path=output_path,
        partition_features_file=args.partition_features_file,
        print_partition_statistics=args.print_partition_statistics,
        save_dgl_graph=args.save_dgl_graph,
        T=args.T,
        K=args.K,
    )

    log.info("Starting partitioning ...")
    t0 = time.perf_counter()
    partitioner.run()
    core_time = time.perf_counter() - t0
    log.info("Partitioning completed in %.3f s", core_time)

    resolved_parts = getattr(partitioner, "number_partition", number_partition)
    if resolved_parts == 1:
        log.warning("Graph was not partitioned (number_partition=1). No metrics.json written.")
        print("\n[done]  Partitioning skipped (number_partition=1).")
        return

    total_num_nodes = getattr(partitioner.sp, "number_nodes", None) if hasattr(partitioner, "sp") else None
    total_num_edges = getattr(partitioner.sp, "number_edges", None) if hasattr(partitioner, "sp") else None
    if total_num_nodes is None or total_num_edges is None:
        log.debug("sp.number_nodes/number_edges unavailable — reading edge_list.csv as a fallback.")
        total_num_nodes, total_num_edges = count_original_graph(dataset_path, args.dataset, log)

    assignment, assignment_source = resolve_node_assignment(
        partitioner=partitioner,
        output_path=output_path,
        num_parts=resolved_parts,
        num_nodes=total_num_nodes,
        log=log,
        assignment_file=args.assignment,
        derive=args.derive_assignment,
    )

    total_time = time.perf_counter() - main_start
    metrics = compute_partition_metrics(
        output_path=output_path,
        number_partition=resolved_parts,
        dataset=args.dataset,
        method=METHOD,
        core_time=total_time,
        total_num_nodes=total_num_nodes,
        total_num_edges=total_num_edges,
        log=log,
        assignment=assignment,
        assignment_source=assignment_source,
        dataset_path=dataset_path,
    )

    _log_balance_stats(metrics["edge_counts"], "Edge (induced subgraph)", log)
    _log_balance_stats(metrics["node_counts"], "Node (including K-hop replicas)", log)
    if metrics["node_counts_primary"] is not None:
        _log_balance_stats(metrics["node_counts_primary"], "Node (primary, without replicas)", log)

    if metrics["edge_cut"] is not None:
        log.info(
            "Cross-partition edges : %d / %s  (%.1f %%)  [source=%s, exact=%s]",
            metrics["edge_cut"], metrics["num_edges_considered"],
            100.0 * metrics["edge_cut_ratio"],
            metrics["assignment_source"], metrics["edge_cut_exact"],
        )
    else:
        log.warning("Cannot compute edge cut (no node assignment available).")

    log.info("Replication Factor (node overhead from K-hop replication): %.6f",
             metrics["replication_factor"])
    log.info("Edge Replication (edge overhead from K-hop replication)  : %.6f",
             metrics["edge_replication"])

    print("\n[done]  Partitioning completed successfully.")
    print(
        "CODEx_METRICS "
        f"algorithm={METHOD} "
        f"dataset={args.dataset} "
        f"partitioning_type=node "
        f"k={resolved_parts} "
        f"num_nodes={metrics['num_nodes']} "
        f"num_edges={metrics['num_edges']} "
        f"core_time={core_time:.9f} "
        f"edge_cut={_fmt(metrics['edge_cut'], 'd')} "
        f"edge_cut_ratio={_fmt(metrics['edge_cut_ratio'])} "
        f"edge_cut_exact={metrics['edge_cut_exact']} "
        f"assignment_source={metrics['assignment_source']} "
        f"node_balance={_fmt(metrics['node_balance'])} "
        f"node_balance_primary={_fmt(metrics['node_balance_primary'])} "
        f"edge_balance={_fmt(metrics['edge_balance'])} "
        f"replication_factor={_fmt(metrics['replication_factor'])} "
        f"edge_replication={_fmt(metrics['edge_replication'])}"
    )

    metrics["core_time"] = total_time
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(metrics), fh, indent=2)
    log.info("Metrics written to: %s", metrics_path)


if __name__ == "__main__":
    main()
