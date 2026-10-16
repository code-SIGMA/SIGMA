"""
edge_partition_dgl_format_saver.py
==================================

Save a graph partitioned by EDGE-ASSIGNMENT (vertex-cut) into the DGL
on-disk partition format, symmetric to `save_partition_graph` from
`node_partition_dgl_format_saver`.

Targets DGL 2.5; layout follows the convention used by
`dgl.distributed.partition.partition_graph` (from which the original
node-partition saver was adapted).

Scope
-----
- **Homogeneous graphs only.** Uses DGL's canonical `_N` / `_N:_E:_N`
  type names in the metadata JSON, matching what `dgl.distributed`
  expects for a single-type graph.
- **True vertex-cut.** Every edge belongs to exactly one partition
  (as given by `edge_part`). Nodes are replicated: each partition
  receives all endpoints of its edges, with exactly one copy marked
  as the master (`inner_node=1`) and the rest as mirrors.

Master selection
----------------
Vertex-cut needs *one* master copy per node so that node_map can express
contiguous per-partition ID ranges and features have a single physical
home. The master rule follows the common convention from PowerGraph /
GraphX / DistGNN: the master is the partition that holds the most
incident edges of the node; ties are broken by the smallest partition
ID. Override via `master_rule` if you need a different policy.

Local node ordering inside each partition
-----------------------------------------
Each partition's local node ID space is laid out as

    [ masters of p, in ascending reshuffled-NID order ]
    [ mirrors of p, in ascending reshuffled-NID order ]

which guarantees that:
  * `inner_node==1` rows occupy local IDs `[0, n_master)`,
  * the order of master rows matches the slice taken from the global
    feature tensors when writing `node_feat.dgl`,
  * the order of edge rows in `sg.edata` matches the slice taken from
    the global feature tensors when writing `edge_feat.dgl`.

This is the alignment invariant that DGL's `DistGraph` loader assumes.

On-disk layout
--------------
    out_path/
      graph_name.json
      part0/
        graph.dgl        # DGLGraph with structural ndata/edata only
        node_feat.dgl    # tensors for master nodes only
        edge_feat.dgl    # tensors for edges assigned to this partition
        drpa_meta.pt     # OPTIONAL: DRPA metadata (only when
                         #           write_drpa_meta=True)

Structural fields on each subgraph:
    ndata[dgl.NID]       reshuffled global node IDs
    ndata['inner_node']  uint8, 1 iff master of this partition
    ndata['part_id']     master-partition of each node (owner)
    edata[dgl.EID]       reshuffled global edge IDs
    edata['inner_edge']  uint8, always 1 in vertex-cut (every edge
                         in the subgraph was assigned to this part)


DRPA metadata (opt-in)
----------------------
When `write_drpa_meta=True`, an extra `part{p}/drpa_meta.pt` file is
written per partition, holding the tensors the DGL_XEON DRPA (Delayed
Remote Partial Aggregation) C++ kernels consume -- `node_map`, `adj`,
`lf` -- plus `orig_nid`, a per-local-node map back to the original
graph's node space (used by the training script's --validate-drpa
correctness check). See the `write_drpa_metadata` docstring and the
format specification document for details.

NOTE: DRPA's `node_map` is a DIFFERENT object from the
`metadata["node_map"]["_N"]` written into the graph JSON. The JSON one
holds per-partition *master* ranges (DistDGL convention); the DRPA one
holds the cumulative *total* node count per partition (Libra address
space). Both are emitted; they must not be confused.


Implementation notes (memory-optimised rewrite)
-----------------------------------------------
This file produces the same on-disk output as the previous version but
avoids two major memory blow-ups that hurt on graphs the size of
ogbn-papers100M (~111M nodes, ~1.6B edges, ~57GB of node features):

1. The reshuffled global graph is no longer materialised as a DGLGraph
   with reordered ndata/edata. Doing so doubled every feature tensor
   in memory because both `g` (held by the caller) and `g_r` carried
   full copies. Instead we keep just two integer arrays
   (``new_src_np``, ``new_dst_np``) plus the permutations
   ``new_to_old_nid`` / ``new_to_old_eid``, and slice features directly
   out of ``g.ndata[k]`` / ``g.edata[k]`` using composed indices.

2. The per-partition ``g2l`` lookup buffer (length = num_nodes) is now
   allocated once and reset only at the indices that were touched in
   the previous iteration, instead of being reallocated per partition.
   On papers100M this saves a ~890MB allocation/free per partition.

The 3-key lexsort during reshuffle is preserved because the
within-partition (src, dst) ordering it produces is the same ordering
DGL writes when it builds an edge-cut partition; downstream tools that
inspect the on-disk COO will see a layout consistent with that.
"""

import gc
import json
import os
import time

import numpy as np
import torch as th

import dgl
from dgl.data.utils import save_graphs, save_tensors

# Shared DRPA metadata module -- ALSO imported by the training script.
# Single definition of the on-disk format, the canonical dtypes and the
# invariants. The writer function is aliased because this converter's
# public API already has a boolean parameter named `write_drpa_meta`.
from drpa_meta import (
    DRPA_FORMAT_VERSION,
    DRPA_MASTER_MARKER,
    DRPA_META_FILENAME,
    finalize_drpa_tensors,
    register_drpa_meta,
    validate_drpa_meta,
    validate_leaders_consistent_local,
    write_drpa_meta as _write_drpa_meta_file,
)


# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
DGL_NID = dgl.NID
DGL_EID = dgl.EID

# Reserved structural fields that live on the subgraph itself, not in
# node_feat.dgl / edge_feat.dgl. Feature extraction skips these.
_RESERVED_NDATA = {DGL_NID, "inner_node", "part_id"}
_RESERVED_EDATA = {DGL_EID, "inner_edge"}

# Tunable threshold for choosing the dense vs sparse "degree" master rule.
# At int64, 5e8 cells == 4 GB. Adjust if your machine has more/less RAM.
_DENSE_DEGREE_CELL_LIMIT = 5e8


# ====================================================================
# DRPA metadata constants -- single source of truth: drpa_meta.py
# ====================================================================
# DRPA_MASTER_MARKER (-200), DRPA_META_FILENAME ("drpa_meta.pt") and
# DRPA_FORMAT_VERSION are imported from the shared drpa_meta module
# (see imports above), which the training script imports as well, so
# producer and consumer can never disagree on format, dtypes or
# invariants. If the DGL_XEON fork ever uses a different sentinel,
# change it in drpa_meta.py only.
#
# Format history (see drpa_meta.DRPA_FORMAT_VERSION):
#   v1: node_map / adj / lf
#   v2: + orig_nid (per-local-node original NID, for correctness
#       validation against the full graph)
#   v3: lf semantics fixed -- a mirror's leader is now always the
#       node's MASTER partition (previously a free choice among copies,
#       which put ~69% of aggregated embeddings into mirror rows and
#       broke the DRPA forward). node_map / adj / orig_nid unchanged.
#   v4: + "validated" flag (this converter ran the shared invariant
#       checks before writing, so the trainer skips them at start-up),
#       + rank / num_parts / n_total self-description. Tensor content
#       and dtypes unchanged relative to v3.


# --------------------------------------------------------------------
# Master assignment
# --------------------------------------------------------------------
def _assign_masters(graph, edge_part, num_parts, rule="degree"):
    """
    Pick a single "master" partition per node.

    Rules
    -----
    "degree" : master = partition with most incident edges of the node
               (ties: smallest pid). Default. Switches to a memory-light
               per-partition bincount path when (N * num_parts) would be
               too large for a dense counts matrix.
    "first"  : master = partition of the first incident edge seen
               (stable w.r.t. edge order in graph.edges()). Vectorized.
    "hash"   : master = node_id % num_parts.

    Isolated nodes (no incident edges) fall back to node_id % num_parts
    in all rules.
    """
    N = int(graph.num_nodes())
    src, dst = graph.edges()
    src_np = src.numpy().astype(np.int64, copy=False)
    dst_np = dst.numpy().astype(np.int64, copy=False)

    if rule == "hash":
        return np.arange(N, dtype=np.int64) % num_parts

    if rule == "first":
        # Vectorized "first-touch wins": the partition of the smallest
        # edge index in which the node appears (as src or dst).
        #
        # Memory note: this rule allocates ~6 * num_edges * 8 bytes of
        # int64 (concat of nodes/pids/eids + lexsort scratch). For
        # papers100M-class graphs that's >75 GB of peak; prefer "degree"
        # if RAM is tight.
        e_idx = np.arange(edge_part.shape[0], dtype=np.int64)
        nodes = np.concatenate([src_np, dst_np])
        pids = np.concatenate([edge_part, edge_part])
        eids = np.concatenate([e_idx, e_idx])
        order = np.lexsort((eids, nodes))   # within each node, by edge idx asc
        nodes_s = nodes[order]
        pids_s = pids[order]
        # Free the originals; we only need the sorted views from here.
        del nodes, pids, eids, order

        master = np.full(N, -1, dtype=np.int64)
        if nodes_s.size > 0:
            first_mask = np.empty_like(nodes_s, dtype=bool)
            first_mask[0] = True
            first_mask[1:] = nodes_s[1:] != nodes_s[:-1]
            master[nodes_s[first_mask]] = pids_s[first_mask]
        del nodes_s, pids_s

    elif rule == "degree":
        # Choose the dense or sparse path based on memory.
        if float(N) * float(num_parts) <= _DENSE_DEGREE_CELL_LIMIT:
            counts = np.zeros((N, num_parts), dtype=np.int64)
            np.add.at(counts, (src_np, edge_part), 1)
            np.add.at(counts, (dst_np, edge_part), 1)
            master = counts.argmax(axis=1).astype(np.int64)
            isolated = counts.sum(axis=1) == 0
            master[isolated] = -1
            del counts
        else:
            # Memory-light: iterate partitions, keep best (count, pid)
            # per node. First-improvement-wins => smallest pid wins on
            # ties (matches dense argmax).
            #
            # The original allocated `np.concatenate([src[mask], dst[mask]])`
            # per partition. For papers100M that's ~3 GB of scratch
            # per iteration. We instead bincount src and dst separately
            # and add, which avoids the concatenation.
            best_count = np.zeros(N, dtype=np.int64)
            master = np.full(N, -1, dtype=np.int64)
            for p in range(num_parts):
                mask = edge_part == p
                if not np.any(mask):
                    continue
                src_p = src_np[mask]
                dst_p = dst_np[mask]
                c = np.bincount(src_p, minlength=N)
                c += np.bincount(dst_p, minlength=N)
                del src_p, dst_p, mask
                improve = c > best_count
                best_count[improve] = c[improve]
                master[improve] = p
                del c, improve
            del best_count
            # Anything still -1 has zero incident edges in any partition;
            # the fallback below catches it.

    else:
        raise ValueError(f"Unknown master rule: {rule!r}")

    # Fallback for isolated / unreached nodes.
    unassigned = np.where(master < 0)[0]
    if unassigned.size > 0:
        print(
            f"  [WARN] {unassigned.size} isolated/unassigned node(s); "
            f"assigning via round-robin (node_id % num_parts)."
        )
        master[unassigned] = unassigned % num_parts

    return master


# --------------------------------------------------------------------
# Reshuffle (lightweight: numpy arrays only, no DGLGraph copy)
# --------------------------------------------------------------------
def _reshuffle_arrays(graph, master, edge_part):
    """
    Compute the reshuffle permutations and reordered src/dst arrays
    *without* materialising a reshuffled DGLGraph or copying any
    feature tensors.

    Returns
    -------
    new_to_old_nid : np.ndarray, int64, shape (N,)
        new_to_old_nid[i] = original node ID at new position i.
    new_to_old_eid : np.ndarray, int64, shape (E,)
        new_to_old_eid[i] = original edge ID at new position i.
    master_new : np.ndarray, int64, shape (N,)
        Master partition of each node, indexed in the new-NID space.
    edge_part_new : np.ndarray, int64, shape (E,)
        Edge partition of each edge, indexed in the new-EID space.
    new_src_np, new_dst_np : np.ndarray, int64, shape (E,)
        Edge endpoints in the new-NID space, ordered in the new-EID
        space (i.e. blocked by partition, then by (src, dst) within
        a partition -- preserving the layout the previous DGLGraph
        version produced).
    """
    num_nodes = graph.num_nodes()

    # Stable sort: within a partition, original node order is preserved.
    # ``mergesort`` (used by stable=True) is O(N log N) time, O(N) extra
    # space; not the dominant cost for papers100M (~111M nodes).
    new_to_old_nid = np.argsort(master, kind="stable").astype(np.int64)
    old_to_new_nid = np.empty(num_nodes, dtype=np.int64)
    old_to_new_nid[new_to_old_nid] = np.arange(num_nodes, dtype=np.int64)
    master_new = master[new_to_old_nid]

    src, dst = graph.edges()
    src_np = src.numpy().astype(np.int64, copy=False)
    dst_np = dst.numpy().astype(np.int64, copy=False)

    # Translate edges into the new-NID space in place where possible.
    # `take` with `mode='raise'` does the bounds-check but creates a
    # fresh array; we accept that allocation since src/dst are fed to
    # lexsort right after.
    new_src_full = old_to_new_nid[src_np]
    new_dst_full = old_to_new_nid[dst_np]
    # `src_np` / `dst_np` are torch-backed views; no need to delete.

    # Group edges by partition, then order by (new_src, new_dst) within
    # each partition. Three-key lexsort is preserved from the original
    # implementation: that within-partition COO ordering is what DGL
    # writes for edge-cut partitions, so downstream tools see a layout
    # consistent with native dgl.distributed output. Sort key precedence
    # in lexsort goes from last-listed to first-listed.
    sort_key = np.lexsort((new_dst_full, new_src_full, edge_part))
    new_to_old_eid = sort_key.astype(np.int64, copy=False)
    edge_part_new = edge_part[new_to_old_eid]

    # Apply the permutation to the new-NID-space src/dst arrays. After
    # this point we no longer need the unpermuted versions.
    new_src_np = new_src_full[new_to_old_eid]
    new_dst_np = new_dst_full[new_to_old_eid]
    del new_src_full, new_dst_full, sort_key, old_to_new_nid

    return (
        new_to_old_nid,
        new_to_old_eid,
        master_new,
        edge_part_new,
        new_src_np,
        new_dst_np,
    )


# --------------------------------------------------------------------
# Per-partition subgraph (alignment-correct, no g_r required)
# --------------------------------------------------------------------
def _build_partition_subgraph(
    new_src_np,
    new_dst_np,
    p,
    master_new,
    edge_part_new,
    g2l_buf,
    g2l_dirty,
    new_to_old_nid
):
    """
    Build partition `p`'s subgraph with a deterministic local node order:

        local IDs [0, n_master) : masters of p, in ascending reshuffled-NID
                                  order
        local IDs [n_master, _) : mirrors of p, in ascending reshuffled-NID
                                  order

    `g2l_buf` is a caller-provided int64 buffer of length num_nodes used
    as a global-(reshuffled)-NID -> local index lookup. The buffer is
    treated as scratch; on entry the indices listed in `g2l_dirty` (from
    the previous call) are first reset to -1 so we don't carry state
    between partitions. On exit, `g2l_dirty` is replaced with the new
    set of touched indices.

    Vertex-cut semantics: every edge in `sg` was assigned to `p`, so
    `inner_edge` is all-ones.

    Returns
    -------
    sg : DGLGraph
    inner_nids_new : np.ndarray (int64, ascending)
        Master NIDs of p in the reshuffled global NID space.
    inner_eids_new : np.ndarray (int64, ascending)
        EIDs of p in the reshuffled global EID space.
    new_g2l_dirty : np.ndarray
        The local_nids for this partition; pass back in next call.
        NOTE: this array is ALSO this partition's `local_nids` (the
        reshuffled NIDs, masters-then-mirrors). `save_edge_partition_graph`
        relies on that when collecting DRPA metadata.
    """
    # Edges assigned to this partition (already contiguous after
    # reshuffle, but np.where keeps this robust to any future reshuffle
    # change).
    inner_eids_new = np.where(edge_part_new == p)[0].astype(
        np.int64, copy=False
    )

    # Slice src/dst in the new-NID space for this partition's edges.
    p_src = new_src_np[inner_eids_new]
    p_dst = new_dst_np[inner_eids_new]

    # Master nodes of p (contiguous block in new-NID space, ascending).
    inner_nids_new = np.where(master_new == p)[0].astype(
        np.int64, copy=False
    )

    # Endpoints of p's edges, deduped & sorted ascending.
    if p_src.size > 0:
        endpoint_nids = np.unique(np.concatenate([p_src, p_dst]))
    else:
        endpoint_nids = np.empty(0, dtype=np.int64)

    # Mirrors = endpoints that are NOT masters of p. Both arrays are
    # sorted ascending; setdiff1d preserves ascending order.
    mirror_nids = np.setdiff1d(
        endpoint_nids, inner_nids_new, assume_unique=True
    )
    del endpoint_nids

    # Local node list: masters first (in ascending order), mirrors after
    # (also ascending). Isolated masters (owned by p but with no edge in
    # p) are naturally included via inner_nids_new.
    local_nids = np.concatenate([inner_nids_new, mirror_nids])
    del mirror_nids

    # Reset the previously-dirtied entries of the lookup buffer, then
    # write the entries we need this round. This avoids reallocating a
    # full N-int64 array per partition.
    if g2l_dirty is not None:
        g2l_buf[g2l_dirty] = -1
    g2l_buf[local_nids] = np.arange(local_nids.size, dtype=np.int64)

    local_src = g2l_buf[p_src]
    local_dst = g2l_buf[p_dst]
    del p_src, p_dst

    sg = dgl.graph(
        (th.from_numpy(local_src), th.from_numpy(local_dst)),
        num_nodes=int(local_nids.size),
    )
    del local_src, local_dst

    # Structural ndata.
    part_id_arr = master_new[local_nids].astype(np.int64, copy=False)
    inner_node = (part_id_arr == p).astype(np.uint8)
    sg.ndata[DGL_NID]      = th.from_numpy(local_nids)
    sg.ndata["orig_id"]    = th.from_numpy(new_to_old_nid[local_nids])
    sg.ndata["part_id"]    = th.from_numpy(part_id_arr)
    sg.ndata["inner_node"] = th.from_numpy(inner_node)

    # Structural edata.
    sg.edata[DGL_EID]      = th.from_numpy(inner_eids_new)
    sg.edata["inner_edge"] = th.ones(int(inner_eids_new.size), dtype=th.uint8)

    n_in = int(inner_node.sum())
    assert n_in == inner_nids_new.size, \
        f"inner_node={n_in} != inner_nids_new={inner_nids_new.size}"
    assert np.array_equal(new_to_old_nid[local_nids][:n_in],
                          new_to_old_nid[inner_nids_new]), \
        "orig_id does not match feature extraction"

    return sg, inner_nids_new, inner_eids_new, local_nids


# --------------------------------------------------------------------
# Feature extraction (slices straight from the original graph)
# --------------------------------------------------------------------
def _extract_features(
    g,
    new_to_old_nid,
    new_to_old_eid,
    inner_nids_new,
    inner_eids_new,
    exclude_ndata=None,
    exclude_edata=None,
):
    """
    Slice inner-only feature tensors directly out of the *original*
    graph using composed indices, avoiding a second reordered copy.

    Conceptually equivalent to the previous version's

        graph_r.ndata[k][inner_nids_new]
        graph_r.edata[k][inner_eids_new]

    but without ever materialising graph_r.ndata / graph_r.edata.

    Row order:
      - node_feats: rows correspond to inner_nids_new (ascending), which
        matches the master-row order in `sg`.
      - edge_feats: rows correspond to inner_eids_new (ascending), which
        matches the edge order in `sg`.
    """
    exclude_ndata = set(exclude_ndata or [])
    exclude_edata = set(exclude_edata or [])

    skip_n = _RESERVED_NDATA | exclude_ndata
    skip_e = _RESERVED_EDATA | exclude_edata

    nid_old_idx = th.from_numpy(new_to_old_nid[inner_nids_new])
    eid_old_idx = th.from_numpy(new_to_old_eid[inner_eids_new])
    node_feats = {
        f"_N/{k}": v[nid_old_idx]
        for k, v in g.ndata.items() if k not in skip_n
    }
    edge_feats = {
        f"_N:_E:_N/{k}": v[eid_old_idx]
        for k, v in g.edata.items() if k not in skip_e
    }
    return node_feats, edge_feats


# --------------------------------------------------------------------
# Input normalisation
# --------------------------------------------------------------------
def _normalize_edge_part(edge_part, num_edges):
    """Accept torch tensor, numpy array, list, or pandas Series."""
    if hasattr(edge_part, "detach"):  # torch tensor
        arr = edge_part.detach().cpu().numpy()
    elif hasattr(edge_part, "to_numpy"):  # pandas
        arr = edge_part.to_numpy()
    else:
        arr = np.asarray(edge_part)

    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        raise TypeError(
            f"edge_part must be integer-typed; got dtype={arr.dtype}. "
            f"Cast to an integer dtype before calling."
        )
    arr = np.ascontiguousarray(arr, dtype=np.int64).reshape(-1)

    if arr.shape[0] != num_edges:
        raise ValueError(
            f"edge_part has length {arr.shape[0]} but graph has "
            f"{num_edges} edges."
        )
    if arr.size > 0 and arr.min() < 0:
        raise ValueError(
            f"edge_part contains negative values (min={arr.min()})."
        )
    return arr


# ====================================================================
# DRPA metadata writer
# ====================================================================
#
# Produces, for each partition, a `drpa_meta.pt` file with the three
# tensors the DGL_XEON DRPA C++ kernels consume:
#
#     node_map : int32, shape (num_parts,)
#     adj      : int32, shape (n_total_p, num_parts - 1)
#     lf       : int32, shape (n_total_p,)
#
# Address space ("Libra address space")
# --------------------------------------
# DRPA does NOT use the reshuffled-NID space and does NOT use the
# DistDGL `node_map` (per-partition master ranges) in the graph JSON.
# It uses:
#
#     global_libra_id(p, local_idx) = node_map[p - 1] + local_idx
#                                     (node_map[-1] == 0 for p == 0)
#
# where `node_map` is the cumulative sum of the *total* node count
# (masters + mirrors) of each partition. Derived directly from the C++:
#
#   node2partition()    : returns first p with `val < node_map[p]`
#                         => node_map = cumulative EXCLUSIVE upper bounds
#   map2local_index_()  : local_idx = global_id - node_map[pid - 1]
#                         => offset of partition pid is node_map[pid-1]
# --------------------------------------------------------------------


def _choose_leader(reshuffled_nid, copy_parts, master_part):
    """
    Pick the leader partition for one global node: the leader is always
    the node's MASTER partition.

    Why the master, and not a free choice among copies
    ---------------------------------------------------
    DRPA's scatter_reduce_v41 aggregates the gathered remote embeddings
    into `feat[local_index]` of the LEADER copy, and scatter_reduce_v42
    then redistributes from there. If the leader is a mirror, the fully
    aggregated embedding ends up in a mirror row and the master copy
    never receives it -- which makes the master's embedding wrong
    (confirmed empirically: with a free leader choice, ~69% of leaders
    were mirrors and the DRPA forward diverged from the single-machine
    forward).

    Making the leader the master is also automatically globally
    consistent: every node has exactly one master partition, and every
    partition holding a copy computes the same answer from master_part.
    No seed / RNG is needed.

    Parameters
    ----------
    reshuffled_nid : int
        Reshuffled global NID of the node (kept for signature stability
        and debugging; not used for the choice).
    copy_parts : sorted list[int]
        Partitions holding a copy of this node (>= 1 entry).
    master_part : int
        The node's master partition (== master_new[reshuffled_nid]).

    Returns
    -------
    int : the leader partition id (always == master_part).
    """
    if master_part not in copy_parts:
        raise AssertionError(
            f"reshuffled-NID {reshuffled_nid}: master partition "
            f"{master_part} is not among its copy partitions "
            f"{copy_parts}. The partition build is inconsistent."
        )
    return int(master_part)


def _build_copy_index(local_nids_per_part, num_parts):
    """
    Invert the per-partition local_nids arrays into a copy index.

    Parameters
    ----------
    local_nids_per_part : list[np.ndarray]
        local_nids_per_part[p][local_idx] = reshuffled NID of the node
        at local position `local_idx` in partition p. Exactly the
        `local_nids` array returned by `_build_partition_subgraph`.
    num_parts : int

    Returns
    -------
    copies : dict[int, list[tuple[int, int]]]
        reshuffled_nid -> sorted-by-partition list of (partition,
        local_idx) for every copy of that node.
    """
    copies = {}
    for p in range(num_parts):
        lnids = local_nids_per_part[p]
        for local_idx in range(lnids.shape[0]):
            r = int(lnids[local_idx])
            copies.setdefault(r, []).append((p, local_idx))
    for r in copies:
        copies[r].sort(key=lambda pi: pi[0])
    return copies


def write_drpa_metadata(
    out_path_abs,
    num_parts,
    local_nids_per_part,
    master_new,
    new_to_old_nid,
    seed=42,
    verify=True,
    part_metadata=None,
    n_full=None,
):
    """
    Write DRPA metadata (node_map / adj / lf / orig_nid) per partition.

    Per partition the sequence is: finalize dtypes -> validate the
    shared invariants (drpa_meta.validate_drpa_meta) -> write the file
    with validated=True -> register the relative path plus the
    "_drpa_meta" block in `part_metadata`. A partition that fails
    validation aborts the conversion with an AssertionError naming the
    partition and the violated invariant; nothing is registered for it.
    After the last partition, the cross-partition leader-agreement check
    (drpa_meta.validate_leaders_consistent_local) runs in this process
    -- the converter-side replacement for the trainer's all_gather-based
    --check-drpa-consistency, at zero communication cost.

    Called by `save_edge_partition_graph` after all per-partition
    subgraphs have been built and written, when `write_drpa_meta=True`.

    Parameters
    ----------
    out_path_abs : str
        Absolute output directory. DRPA files go to
        <out_path_abs>/part{p}/drpa_meta.pt.
    num_parts : int
    local_nids_per_part : list[np.ndarray]
        Per-partition reshuffled-NID arrays (== sg.ndata[dgl.NID]),
        masters in the leading slots [0, n_master).
    master_new : np.ndarray, int64, shape (N,)
        master_new[r] = master partition of reshuffled-NID r.
    new_to_old_nid : np.ndarray, int64, shape (N,)
        new_to_old_nid[r] = ORIGINAL NID of the node whose reshuffled
        NID is r. Used to emit `orig_nid` so downstream consumers can
        map partition-local nodes back to the original (non-partitioned)
        graph -- needed by the training script's --validate-drpa mode.
    seed : int
        DEPRECATED / no-op. Leader selection no longer uses a seed --
        the leader of every node is its master partition (deterministic
        and globally consistent without randomness). The argument is
        kept for call-site compatibility and is still recorded in the
        file as "_seed" for provenance, but it does not affect output.
    verify : bool
        Run the shared per-partition invariant checks and the
        cross-partition leader check before/after writing (recommended).
        With verify=False the files are written with validated=False,
        and the trainer re-validates them at start-up as it always did.
    part_metadata : dict, optional
        The partition metadata dict (the graph JSON content). When
        given, the per-partition "drpa_meta" paths and the
        "_drpa_meta" block are registered into it via
        drpa_meta.register_drpa_meta.
    n_full : int, optional
        Number of nodes in the ORIGINAL (non-partitioned) graph.
        Enables the orig_nid range check; skipped when None.

    On-disk format (per partition), `torch.save`-d dict (written by
    drpa_meta.write_drpa_meta -- kept in one place, see that module):
        "node_map" : int32 tensor, shape (num_parts,)  [global; identical
                     in every partition file]
        "adj"      : int32 tensor, shape (n_total_p, max(num_parts-1, 1))
        "lf"       : int32 tensor, shape (n_total_p,)  -- -200 for a
                     master; for a mirror, the Libra global id of the
                     node's MASTER copy (the leader is always the
                     master, see _choose_leader). [semantics fixed v3]
        "orig_nid" : int64 tensor, shape (n_total_p,)  -- original NID
                     of each local node (masters then mirrors), i.e.
                     the key into the full graph's node space. [v2+]
        "_drpa_format_version" : int
        "_drpa_master_marker"  : int  [v4+]
        "validated" : bool -- True iff the invariants were checked
                     before writing; the trainer then skips them. [v4+]
        "rank" / "num_parts" / "n_total" : self-description [v4+]
        "_seed"    : int   -- provenance only; no longer affects output

    Returns
    -------
    np.ndarray : the DRPA node_map (int64), for the caller's reference.
    """
    print(f"  [DRPA] Building DRPA metadata "
          f"(leader = master partition) ...")

    new_to_old_nid = np.asarray(new_to_old_nid)
    if new_to_old_nid.ndim != 1:
        raise ValueError(
            f"new_to_old_nid must be 1-D; got shape {new_to_old_nid.shape}."
        )

    # --- node_map: cumulative sum of per-partition TOTAL node counts --
    n_total_per_part = np.array(
        [int(local_nids_per_part[p].shape[0]) for p in range(num_parts)],
        dtype=np.int64,
    )
    node_map_np = np.cumsum(n_total_per_part).astype(np.int64)
    # offset[p] = node_map[p-1], with offset[0] = 0.
    offset = np.concatenate([[0], node_map_np[:-1]]).astype(np.int64)

    if node_map_np[-1] != n_total_per_part.sum():
        raise AssertionError("DRPA node_map cumsum is inconsistent.")
    if node_map_np[-1] >= np.iinfo(np.int32).max:
        raise OverflowError(
            f"Total copies ({node_map_np[-1]}) exceeds int32 range; DRPA "
            f"node_map / adj / lf cannot be int32 for this graph."
        )

    node_map_t = th.from_numpy(node_map_np.astype(np.int32))

    # --- copy index: reshuffled_nid -> [(p, local_idx), ...] ----------
    copies = _build_copy_index(local_nids_per_part, num_parts)

    # --- leader partition per reshuffled_nid (global, once) -----------
    # The leader is always the node's master partition (see
    # _choose_leader for the reasoning). This is deterministic and
    # globally consistent without any seed.
    leader_part = {}
    for r, copy_list in copies.items():
        copy_parts = [p for (p, _) in copy_list]
        leader_part[r] = _choose_leader(r, copy_parts, int(master_new[r]))

    # --- build adj + lf for each partition ----------------------------
    width = max(num_parts - 1, 1)  # >=1 so the tensor is well-formed
    written_parts = []
    leader_parts = []   # (reshuffled_nid, lf, inner_node) per partition
    for p in range(num_parts):
        lnids = local_nids_per_part[p]
        n_total = int(lnids.shape[0])

        adj = np.full((n_total, width), -1, dtype=np.int32)
        lf = np.empty(n_total, dtype=np.int32)

        for local_idx in range(n_total):
            r = int(lnids[local_idx])
            copy_list = copies[r]

            # --- adj row: Libra global ids of the OTHER copies --------
            col = 0
            for (cp, c_local) in copy_list:
                if cp == p:
                    continue
                adj[local_idx, col] = int(offset[cp]) + int(c_local)
                col += 1

            # --- lf entry --------------------------------------------
            if int(master_new[r]) == p:
                lf[local_idx] = DRPA_MASTER_MARKER
            else:
                ldr_p = leader_part[r]
                ldr_local = None
                for (cp, c_local) in copy_list:
                    if cp == ldr_p:
                        ldr_local = c_local
                        break
                if ldr_local is None:
                    raise AssertionError(
                        f"part {p}: leader partition {ldr_p} for "
                        f"reshuffled-NID {r} holds no copy."
                    )
                lf[local_idx] = int(offset[ldr_p]) + int(ldr_local)

        # orig_nid: original NID of every local node (masters then
        # mirrors), aligned with the partition graph's local-ID order.
        # lnids holds reshuffled NIDs; new_to_old_nid maps those back to
        # the original node space.
        orig_nid = new_to_old_nid[lnids].astype(np.int64, copy=False)

        # inner_node vector of this partition (1 = master), aligned with
        # the partition graph's local-ID order (masters then mirrors).
        inner_np = (master_new[lnids] == p).astype(np.uint8)
        inner_t = th.from_numpy(inner_np)

        # --- finalize -> validate -> write -> register ----------------
        # Single source of truth for dtypes and invariants: drpa_meta.py
        # (shared with the trainer). If validation fails, the conversion
        # aborts HERE, before this partition's file is written and before
        # anything is registered in the JSON -- no half-valid artefact.
        t = finalize_drpa_tensors(node_map_t, adj, lf, orig_nid=orig_nid)
        if verify:
            validate_drpa_meta(
                t["node_map"], t["adj"], t["lf"],
                inner_node=inner_t, rank=p, num_parts=num_parts,
                orig_nid=t["orig_nid"], n_full=n_full,
            )

        part_dir = os.path.join(out_path_abs, f"part{p}")
        meta_path = os.path.join(part_dir, DRPA_META_FILENAME)
        _write_drpa_meta_file(
            meta_path,
            t["node_map"], t["adj"], t["lf"], t["orig_nid"],
            rank=p, num_parts=num_parts,
            validated=bool(verify),      # trainer only trusts checked files
            finalize=False,              # dtypes already finalized above
            extra={"_seed": int(seed)},  # provenance key, as before
        )
        if part_metadata is not None:
            register_drpa_meta(
                part_metadata, p, os.path.relpath(meta_path, out_path_abs)
            )
        written_parts.append(p)

        # Input for the cross-partition leader check after the loop.
        leader_parts.append(
            (th.from_numpy(lnids.astype(np.int64, copy=False)),
             t["lf"], inner_t)
        )

        n_master_p = int(inner_np.sum())
        print(
            f"  [DRPA] part {p}: n_total={n_total}, n_master={n_master_p}, "
            f"n_mirror={n_total - n_master_p} -> {DRPA_META_FILENAME} "
            f"(v{DRPA_FORMAT_VERSION}, validated={bool(verify)})"
        )

    # --- completeness + cross-partition leader agreement --------------
    if len(written_parts) != num_parts:
        missing = sorted(set(range(num_parts)) - set(written_parts))
        raise RuntimeError(
            f"DRPA metadata missing for partitions {missing}; the trainer "
            f"would refuse to start."
        )
    if verify:
        # Converter-side replacement for the trainer's all_gather-based
        # --check-drpa-consistency. Since v3 the leader is the master
        # partition, chosen once globally in this process, so a conflict
        # is impossible by construction -- this is defence-in-depth
        # against future changes to the builder above.
        validate_leaders_consistent_local(leader_parts)
        print("  [DRPA] cross-partition leader consistency: PASS")
    else:
        print("  [DRPA] cross-partition leader consistency: SKIPPED "
              "(verify=False; the trainer keeps --check-drpa-consistency)")
    del leader_parts

    if part_metadata is not None:
        glob = part_metadata.setdefault("_drpa_meta", {})
        glob["seed"] = int(seed)          # provenance key, kept from before
        glob["validated"] = bool(verify)  # per-part + cross checks ran

    print(f"  [DRPA] node_map = {node_map_np.tolist()} "
          f"(total copies = {int(node_map_np[-1])})")
    return node_map_np


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def save_edge_partition_graph(
    g,
    edge_part,
    graph_name,
    num_parts,
    out_path,
    partitioner,
    master_rule="degree",
    exclude_ndata_keys=None,
    exclude_edata_keys=("count",),
    return_mapping=False,
    verify=True,
    write_drpa_meta=False,
    drpa_seed=42,
):
    """
    Save a graph partitioned by EDGE ASSIGNMENT (vertex-cut) into DGL's
    on-disk partition format. Homogeneous graphs only.

    Parameters
    ----------
    g : DGLGraph
        The input (global) homogeneous graph.
    edge_part : 1D array-like of int
        Partition assignment per edge, length == g.num_edges(), values
        in [0, num_parts). Aligned with `g.edges()` ordering.
    graph_name : str
        Name for the output metadata JSON (and the name used by
        DistGraph when loading).
    num_parts : int
        Number of partitions.
    out_path : str
        Output directory.
    partitioner : str
        Name of the partitioner, written into the metadata JSON
        (`part_method` field).
    master_rule : {"degree", "first", "hash"}, optional
        How to pick the single master partition per node (see module
        docstring). Default: "degree".
    exclude_ndata_keys : iterable of str, optional
        ndata keys to skip when writing node_feat.dgl.
    exclude_edata_keys : iterable of str, optional
        edata keys to skip when writing edge_feat.dgl. Defaults to
        ("count",) to drop the artifact left by `dgl.to_simple`.
    return_mapping : bool, optional
        If True, also return the (new_to_old_nid, new_to_old_eid) pair
        so the caller can map reshuffled IDs back to the original graph.
    verify : bool, optional
        If True, run cheap correctness assertions on each per-partition
        subgraph before writing it (recommended; catches alignment bugs
        immediately). Also gates the DRPA metadata verification: with
        verify=True the drpa_meta.pt files are stamped validated=True and
        the trainer skips its start-up checks; with verify=False they
        carry validated=False and the trainer validates as before.
        Default: True.
    write_drpa_meta : bool, optional
        If True, additionally write `part{p}/drpa_meta.pt` for each
        partition, containing the DRPA `node_map` / `adj` / `lf`
        tensors. Default: False (opt-in). See `write_drpa_metadata`.
    drpa_seed : int, optional
        Fixed seed for deterministic DRPA leader selection. Only used
        when `write_drpa_meta=True`. Default: 42.

    Returns
    -------
    None, or (new_to_old_nid, new_to_old_eid) if `return_mapping=True`.
    """
    if not g.is_homogeneous:
        raise ValueError(
            "save_edge_partition_graph supports homogeneous graphs only. "
            "Run `dgl.to_homogeneous(g)` before calling."
        )
    all_formats = sum(g.formats().values(), [])
    assert "coo" in all_formats, \
        "'coo' format must be available for partitioning."

    edge_part = _normalize_edge_part(edge_part, g.num_edges())

    # Sanity: partition count.
    if edge_part.size > 0 and edge_part.max() >= num_parts:
        raise ValueError(
            f"edge_part contains pid={edge_part.max()} >= num_parts={num_parts}."
        )

    present = (np.unique(edge_part) if edge_part.size > 0
               else np.empty(0, dtype=np.int64))
    missing = np.setdiff1d(np.arange(num_parts), present)
    if missing.size > 0:
        print(
            f"  [INFO] {missing.size} of {num_parts} partitions have no "
            f"assigned edges: {missing.tolist()}"
        )
    del present

    # --- Master assignment ----------------------------------------------
    start = time.time()
    master = _assign_masters(g, edge_part, num_parts, rule=master_rule)
    print(
        f"Assigning masters ({master_rule}) takes {time.time() - start:.3f}s. "
        f"Masters per partition: "
        f"{np.bincount(master, minlength=num_parts).tolist()}"
    )

    # --- Reshuffle ------------------------------------------------------
    # Lightweight reshuffle: only permutations + reordered src/dst, no
    # DGLGraph copy and no feature-tensor copy.
    start = time.time()
    (
        new_to_old_nid,
        new_to_old_eid,
        master_new,
        edge_part_new,
        new_src_np,
        new_dst_np,
    ) = _reshuffle_arrays(g, master, edge_part)
    # `master` and `edge_part` (in old-EID space) are no longer needed.
    del master, edge_part
    gc.collect()
    print(f"Reshuffling graph takes {time.time() - start:.3f}s.")

    # Sanity: per-partition master ranges are contiguous in new-NID
    # space. Since master_new is the result of np.argsort(master), the
    # check is essentially a guard against a future change to
    # _reshuffle_arrays.
    for p in range(num_parts):
        nids = np.where(master_new == p)[0]
        if nids.size > 0 and not np.array_equal(
            nids, np.arange(nids.min(), nids.max() + 1)
        ):
            raise AssertionError(
                f"Reshuffle invariant broken: partition {p} master NIDs "
                f"are not contiguous."
            )
        del nids

    # --- Build metadata scaffold ----------------------------------------
    metadata = {
        "graph_name": graph_name,
        "num_nodes": int(g.num_nodes()),
        "num_edges": int(g.num_edges()),
        "part_method": partitioner,
        "num_parts": int(num_parts),
        # No BFS halo in vertex-cut: mirrors come from the edge assignment.
        # Some DGL versions look for "num_hops"; write both for safety.
        "halo_hops": 0,
        "num_hops": 0,
        "node_map": {"_N": []},
        "edge_map": {"_N:_E:_N": []},
        "ntypes": {"_N": 0},
        "etypes": {"_N:_E:_N": 0},
        # Markers so consumers can tell this apart from an edge-cut dump.
        "_partition_type": "vertex-cut",
        "_master_rule": master_rule,
    }
    # NOTE: the "_drpa_meta" block and the per-partition "drpa_meta"
    # paths are registered by write_drpa_metadata() -- AFTER the metadata
    # has been validated and written. A JSON claiming present=True can
    # therefore only exist for a complete, checked set of drpa_meta.pt
    # files. With write_drpa_meta=False nothing is registered and
    # "_drpa_meta" stays absent, exactly as before.

    os.makedirs(out_path, exist_ok=True)
    out_path_abs = os.path.abspath(out_path)

    curr_n = 0
    curr_e = 0
    tot_master = 0
    tot_inner_edges = 0
    tot_copies = 0

    # Persistent scratch buffer for the global->local NID map. Re-used
    # across partitions; only the touched entries are reset between
    # iterations (see _build_partition_subgraph).
    g2l_buf = np.full(g.num_nodes(), -1, dtype=np.int64)
    g2l_dirty = None

    # Collected only when DRPA metadata is requested. Holds, per
    # partition, the reshuffled-NID array (== sg.ndata[dgl.NID]).
    drpa_local_nids = [] if write_drpa_meta else None

    # --- Per-partition build & save -------------------------------------
    start = time.time()
    for p in range(num_parts):
        sg, inner_nids_new, inner_eids_new, g2l_dirty = (
            _build_partition_subgraph(
                new_src_np,
                new_dst_np,
                p,
                master_new,
                edge_part_new,
                g2l_buf,
                g2l_dirty,
                new_to_old_nid,
            )
        )

        # g2l_dirty IS this partition's local_nids array. Copy it before
        # it gets handed back in as scratch on the next iteration. The
        # copy is cheap relative to feature extraction and decouples us
        # from _build_partition_subgraph's internal buffer reuse.
        if write_drpa_meta:
            drpa_local_nids.append(np.array(g2l_dirty, dtype=np.int64))

        node_feats, edge_feats = _extract_features(
            g,
            new_to_old_nid,
            new_to_old_eid,
            inner_nids_new,
            inner_eids_new,
            exclude_ndata=exclude_ndata_keys,
            exclude_edata=exclude_edata_keys,
        )

        n_inner = int(inner_nids_new.size)
        e_inner = int(inner_eids_new.size)
        n_total = int(sg.num_nodes())
        tot_master += n_inner
        tot_inner_edges += e_inner
        tot_copies += n_total

        # Contiguity sanity (in the global new-NID / new-EID spaces).
        if n_inner > 0:
            assert (inner_nids_new[0] == curr_n
                    and inner_nids_new[-1] == curr_n + n_inner - 1), \
                f"Master NID range not contiguous in partition {p}."
        if e_inner > 0:
            assert (inner_eids_new[0] == curr_e
                    and inner_eids_new[-1] == curr_e + e_inner - 1), \
                f"Inner EID range not contiguous in partition {p}."

        # Alignment invariants (cheap, catch feature-row misalignment).
        if verify:
            n_inner_in_sg = int((sg.ndata["inner_node"] == 1).sum().item())
            assert n_inner_in_sg == n_inner, (
                f"part {p}: inner_node count {n_inner_in_sg} "
                f"!= masters expected {n_inner}."
            )
            if n_inner > 0:
                masters_in_sg = sg.ndata[DGL_NID][:n_inner].numpy()
                assert np.array_equal(masters_in_sg, inner_nids_new), (
                    f"part {p}: master ordering in subgraph does not match "
                    f"node_feat row order; node features would be misaligned."
                )
            if e_inner > 0:
                eids_in_sg = sg.edata[DGL_EID].numpy()
                assert np.array_equal(eids_in_sg, inner_eids_new), (
                    f"part {p}: edge ordering in subgraph does not match "
                    f"edge_feat row order; edge features would be misaligned."
                )

        # Write files.
        part_dir = os.path.join(out_path_abs, f"part{p}")
        os.makedirs(part_dir, exist_ok=True)
        graph_path = os.path.join(part_dir, "graph.dgl")
        nf_path = os.path.join(part_dir, "node_feat.dgl")
        ef_path = os.path.join(part_dir, "edge_feat.dgl")
        save_graphs(graph_path, [sg])
        save_tensors(nf_path, node_feats)
        save_tensors(ef_path, edge_feats)

        metadata["node_map"]["_N"].append(
            [int(curr_n), int(curr_n + n_inner)]
        )
        metadata["edge_map"]["_N:_E:_N"].append(
            [int(curr_e), int(curr_e + e_inner)]
        )
        metadata[f"part-{p}"] = {
            "node_feats": os.path.relpath(nf_path, out_path_abs),
            "edge_feats": os.path.relpath(ef_path, out_path_abs),
            "part_graph": os.path.relpath(graph_path, out_path_abs),
        }

        curr_n += n_inner
        curr_e += e_inner

        print(
            f"part {p}: {n_total} nodes ({n_inner} master, "
            f"{n_total - n_inner} mirror), {e_inner} edges."
        )

        # Free this partition's transients before the next iteration.
        # On papers100M these can each be hundreds of megabytes (the
        # node_feats dict is the largest, since it holds the master-row
        # slice of every ndata tensor including the 128-d float feature).
        del sg, inner_nids_new, inner_eids_new, node_feats, edge_feats
        gc.collect()

    # Reshuffle scratch is no longer needed.
    # NOTE: master_new is intentionally NOT deleted here -- write_drpa_metadata
    # needs it below. It is freed right after the DRPA block.
    del new_src_np, new_dst_np, edge_part_new, g2l_buf, g2l_dirty
    gc.collect()

    print(f"Save partitions: {time.time() - start:.3f} seconds.")

    # --- DRPA metadata (opt-in) ----------------------------------------
    # Runs BEFORE the metadata JSON is dumped: if any partition fails the
    # shared invariant checks, the conversion aborts here and no JSON
    # claiming _drpa_meta.present=True is ever written.
    if write_drpa_meta:
        start_drpa = time.time()
        write_drpa_metadata(
            out_path_abs=out_path_abs,
            num_parts=num_parts,
            local_nids_per_part=drpa_local_nids,
            master_new=master_new,
            new_to_old_nid=new_to_old_nid,
            seed=drpa_seed,
            verify=verify,
            part_metadata=metadata,
            n_full=int(g.num_nodes()),
        )
        print(f"Write DRPA metadata: {time.time() - start_drpa:.3f} seconds.")
        del drpa_local_nids

    # --- Metadata JSON --------------------------------------------------
    json_path = os.path.join(out_path_abs, f"{graph_name}.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    del master_new
    gc.collect()

    # --- Final consistency checks --------------------------------------
    if tot_master != g.num_nodes():
        raise AssertionError(
            f"Total masters ({tot_master}) != num_nodes ({g.num_nodes()})."
        )
    if tot_inner_edges != g.num_edges():
        raise AssertionError(
            f"Total inner edges ({tot_inner_edges}) != num_edges ({g.num_edges()})."
        )

    rf = tot_copies / max(g.num_nodes(), 1)
    print(
        f"Vertex-cut summary: {g.num_edges()} edges across {num_parts} "
        f"partitions, node replication factor = {rf:.3f} "
        f"(copies/N = {tot_copies}/{g.num_nodes()})."
    )

    if return_mapping:
        return new_to_old_nid, new_to_old_eid
    return None
