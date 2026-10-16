import os
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch as th


# ====================================================================
# Node partitioner loader (unchanged semantics, returned type already OK)
# ====================================================================
def load_node_map_flat(input_file_path: Path):
    """One integer per line, line i = partition of node i.

    Returns
    -------
    torch.Tensor, int64, shape (num_nodes,)
    """
    if not input_file_path.is_file():
        raise Exception(f"Not a file: {input_file_path}")

    with open(input_file_path) as f:
        arr = np.fromiter(
            (int(line) for line in f if line.strip().isdigit()),
            dtype=np.int64,
        )

    return th.from_numpy(arr)

import pickle


def load_sdtgnn_node_partition(
    input_dir_path: Path,
    num_nodes: int = None,
    fill_missing: str = "balanced",
):
    partition_file = input_dir_path
    if not partition_file.is_file():
        raise Exception(f"Not a file: {partition_file}")
    with open(partition_file, "rb") as f:
        node_partition_dict = pickle.load(f)
    # Keys may be int or str depending on how the dict was built upstream.
    node_partition_dict = {int(k): int(v) for k, v in node_partition_dict.items()}
    inferred_n = max(node_partition_dict.keys()) + 1
    if num_nodes is None:
        num_nodes = inferred_n
    elif inferred_n > num_nodes:
        raise Exception(
            f"load_sdtgnn_node_partition: max node id {inferred_n - 1} "
            f">= expected num_nodes {num_nodes}."
        )
    arr = np.full(num_nodes, -1, dtype=np.int64)
    for node_id, part_id in node_partition_dict.items():
        arr[node_id] = part_id

    missing = np.flatnonzero(arr < 0)
    if missing.size > 0:
        if fill_missing == "strict":
            raise Exception(
                f"load_sdtgnn_node_partition: {missing.size} node(s) missing from "
                f"partition.json (out of {num_nodes})."
            )
        num_parts = int(arr.max()) + 1
        counts = np.bincount(arr[arr >= 0], minlength=num_parts)
        # Nodes absent from the partition file are isolated (degree 0), so their
        # placement affects neither edge cut nor replication. Assign each to the
        # currently smallest partition; ties go to the lowest part id. Iterating
        # over ascending node ids keeps this reproducible across runs.
        for node_id in missing:
            target = int(np.argmin(counts))
            arr[node_id] = target
            counts[target] += 1
        print(
            f"  load_sdtgnn_node_partition: filled {missing.size} unassigned "
            f"node(s) (isolated), strategy={fill_missing}."
        )

    print(
        f"  load_sdtgnn_node_partition: loaded primary partition for "
        f"{num_nodes} nodes across {len(set(arr.tolist()))} partitions."
    )
    return th.from_numpy(arr)


# ====================================================================
# Helpers for edge loaders
# ====================================================================
def _build_edge_key_to_index(graph):
    """Build a dict `(src, dst) -> eid` over all directed edges of `graph`.

    Used by the edge-assignment loaders to map (src, dst) pairs back to
    the edge index aligned with `graph.edges()` ordering, which is what
    `save_edge_partition_graph` expects.
    """
    g_src, g_dst = graph.edges()
    src_np = g_src.numpy().astype(np.int64)
    dst_np = g_dst.numpy().astype(np.int64)
    # dict lookup is O(E) memory but O(1) per query; fine for offline tools.
    return {
        (int(s), int(d)): i for i, (s, d) in enumerate(zip(src_np, dst_np))
    }, src_np, dst_np


def _build_undirected_key_to_indices(graph):
    """Map (min(u,v), max(u,v)) -> list of edge indices.

    An undirected edge (u, v) may appear in `graph.edges()` as both
    (u, v) and (v, u); both directed edges must get the same partition
    ID when the source file stores one partition per undirected edge.
    """
    g_src, g_dst = graph.edges()
    src_np = g_src.numpy().astype(np.int64)
    dst_np = g_dst.numpy().astype(np.int64)

    lookup = {}
    for i, (s, d) in enumerate(zip(src_np, dst_np)):
        key = (int(min(s, d)), int(max(s, d)))
        lookup.setdefault(key, []).append(i)
    return lookup, src_np, dst_np


# ====================================================================
# Edge partitioner loaders (all return 1-D int64 tensor, length = num_edges,
# aligned with graph.edges())
# ====================================================================
def load_dgl_style_multi(graph, input_dir_path: Path):
    """Vertex-cut node-replication format: `part{k}/nodes.txt` lists the
    global node IDs that partition k holds. A node may appear in multiple
    partitions (replicated masters/mirrors).

    The edge assignment is derived from the node membership as follows:

      1. An edge (u, v) can go to any partition k that contains BOTH u
         and v in `part{k}/nodes.txt`. Call that set `C(u,v)`.
      2. If |C(u,v)| >= 1, pick the partition from C(u,v) with the
         currently lowest assigned-edge count (greedy balance, ties
         broken by smallest part-id). This is the HDRF/balanced-choice
         rule commonly used for vertex-cut tie-breaking.
      3. If |C(u,v)| == 0 (no partition contains both endpoints), emit
         a warning and fall back to `C_fallback = partitions(u) \u222a
         partitions(v)`, again picking by lowest edge-count. This is
         the "least-overlapping-node" fallback.
      4. If a node has no partition at all, fall back to node_id %
         num_parts so we don't crash on isolates / partitioner bugs.

    Returns
    -------
    torch.Tensor, int64, shape (num_edges,)
        Partition of each directed edge, aligned with `graph.edges()`.
    """
    if not input_dir_path.is_dir():
        raise Exception(f"Not a folder: {input_dir_path}")

    # Collect all part* folders and sort NUMERICALLY (not lexicographically),
    # so that part2 < part10 < part33 etc.
    part_pattern = re.compile(r"^part(\d+)$")
    found_parts = {}  # part_id -> folder name
    for f in os.listdir(input_dir_path):
        m = part_pattern.match(f)
        if m and (input_dir_path / f).is_dir():
            pid = int(m.group(1))
            if pid in found_parts:
                raise Exception(
                    f"Duplicate partition id {pid}: "
                    f"'{found_parts[pid]}' and '{f}' both map to part{pid}."
                )
            found_parts[pid] = f

    if not found_parts:
        raise Exception(f"load_dgl_style_multi: no part* folders under {input_dir_path}.")

    num_parts = max(found_parts.keys()) + 1

    # Explicitly check that every expected folder part0 ... part{num_parts-1}
    # exists. This catches gaps like {part0, part1, part3} (part2 missing).
    missing = [pid for pid in range(num_parts) if pid not in found_parts]
    if missing:
        raise Exception(
            f"Missing partition folder(s) under {input_dir_path}: "
            f"{[f'part{p}' for p in missing]}. "
            f"Found parts {sorted(found_parts.keys())}, expected contiguous "
            f"range [0, {num_parts})."
        )

    # Build the ordered list of folder names by numeric part-id.
    folders = [found_parts[pid] for pid in range(num_parts)]

    num_nodes = int(graph.num_nodes())

    # --- Step 1: read per-partition node membership ---------------------
    # Vectorized: store membership as a list of sorted np.int64 arrays
    # (one per node) built from a CSR-style accumulation, instead of one
    # Python set per node. Semantics identical: each node maps to the
    # set of partition ids that list it.
    membership_per_part = []  # membership_per_part[k] = np.int64 array of node ids in part k
    for part_id, folder in enumerate(tqdm(folders, desc="Reading node-partition files")):
        file_path = input_dir_path / folder / "nodes.txt"
        if not file_path.is_file():
            raise Exception(f"Expected file not found: {file_path}")

        with open(file_path) as f:
            raw = f.read()
        ids = np.fromstring(raw, dtype=np.int64, sep=" ") if raw.strip() else np.empty(0, np.int64)
        # `np.fromstring` with sep=" " also splits on newlines.
        if ids.size:
            if ids.min() < 0 or ids.max() >= num_nodes:
                bad = ids[(ids < 0) | (ids >= num_nodes)][0]
                raise Exception(
                    f"{file_path}: node id {int(bad)} out of range "
                    f"[0, {num_nodes})."
                )
        membership_per_part.append(ids)

    # Build, per node, the sorted list of partitions holding it.
    # node_parts_count[v] = number of partitions holding v.
    node_parts_count = np.zeros(num_nodes, dtype=np.int64)
    for ids in membership_per_part:
        if ids.size:
            np.add.at(node_parts_count, ids, 1)
    # node_parts_list[v] = sorted np.int64 array of partition ids (may be empty).
    node_parts_list = [None] * num_nodes
    # Fill via per-partition scatter into Python lists only where needed.
    _tmp = [[] for _ in range(num_nodes)]
    for part_id, ids in enumerate(membership_per_part):
        for nid in ids.tolist():
            _tmp[nid].append(part_id)
    for v in range(num_nodes):
        node_parts_list[v] = np.asarray(_tmp[v], dtype=np.int64)
    del _tmp, membership_per_part

    # Diagnostics: replication factor, unassigned nodes.
    total_copies = int(node_parts_count.sum())
    n_unassigned = int((node_parts_count == 0).sum())
    rf = total_copies / max(num_nodes, 1)
    print(
        f"  load_dgl_style_multi: read {num_parts} partitions, "
        f"node replication factor = {rf:.3f} ({total_copies}/{num_nodes}); "
        f"{n_unassigned} node(s) have no partition assignment."
    )
    if n_unassigned > 0:
        print(
            f"  [WARN] Falling back to `node_id % num_parts` for "
            f"{n_unassigned} unassigned node(s)."
        )

    # --- Step 2: derive per-edge partition via greedy balance -----------
    g_src, g_dst = graph.edges()
    src_np = g_src.numpy().astype(np.int64)
    dst_np = g_dst.numpy().astype(np.int64)
    num_edges = src_np.shape[0]

    edge_parts = np.empty(num_edges, dtype=np.int64)
    edge_counts = np.zeros(num_parts, dtype=np.int64)

    n_no_common = 0           # edges where no partition had both endpoints
    n_multi_candidate = 0     # edges with >= 2 candidates (balance kicks in)

    def _pick_lowest_count(candidates):
        """Return part-id from `candidates` with the smallest current
        edge_count; ties broken by smallest part-id (natural via min())."""
        return min(candidates, key=lambda p: (int(edge_counts[p]), p))

    # Helper returning the candidate set for a node, with the
    # `node_id % num_parts` fallback for nodes with no assignment.
    # IMPORTANT: this returns a Python set to preserve the exact set
    # semantics (& and |) used by the original code.
    def _parts_of(node):
        arr = node_parts_list[node]
        if arr.size:
            return set(arr.tolist())
        return {node % num_parts}

    # The greedy balance is order-dependent (edge_counts mutates), so the
    # multi-candidate edges MUST stay in original edge order. We cannot
    # reorder them. But the *single-candidate* common case is independent
    # of edge_counts: when |common| == 1 the chosen partition is fixed
    # regardless of counts. We still process every edge in order so the
    # mutation sequence of edge_counts is bit-identical to the original.
    for e in range(num_edges):
        u = int(src_np[e]); v = int(dst_np[e])
        au = node_parts_list[u]; av = node_parts_list[v]
        pu = set(au.tolist()) if au.size else {u % num_parts}
        pv = set(av.tolist()) if av.size else {v % num_parts}

        common = pu & pv
        if common:
            if len(common) > 1:
                n_multi_candidate += 1
                chosen = _pick_lowest_count(common)
            else:
                # Single candidate: count-independent, but we still
                # increment edge_counts in order to keep behaviour identical.
                chosen = next(iter(common))
        else:
            n_no_common += 1
            chosen = _pick_lowest_count(pu | pv)

        edge_parts[e] = chosen
        edge_counts[chosen] += 1

    if n_no_common > 0:
        print(
            f"  [WARN] {n_no_common} edge(s) had no partition containing "
            f"both endpoints; assigned via union-of-endpoints fallback."
        )
    print(
        f"  load_dgl_style_multi: edges per partition = "
        f"{edge_counts.tolist()} "
        f"(tie-broken greedily for {n_multi_candidate} multi-candidate edges)."
    )

    return th.from_numpy(edge_parts)


def load_edge_map_flat(graph, input_file_path: Path):
    """Flat file: one partition ID per line, aligned with the list of
    unique undirected edges (self-loops excluded), in the order produced
    by `torch.unique` over `(min(u,v), max(u,v))` pairs.

    Returns a partition-per-directed-edge tensor: both directions of an
    undirected edge receive the same partition.
    """
    if not input_file_path.is_file():
        raise Exception(f"Not a file: {input_file_path}")

    src, dst = graph.edges()
    src_np = src.numpy().astype(np.int64)
    dst_np = dst.numpy().astype(np.int64)
    num_edges = src_np.shape[0]

    # Same deduplication logic as the original loader, so the file's
    # line ordering matches.
    mask = src != dst
    u, v = src[mask], dst[mask]
    a, b = th.minimum(u, v), th.maximum(u, v)
    unique_pairs = th.unique(th.stack((a, b), dim=1), dim=0)

    with open(input_file_path) as f:
        raw = [int(x) for x in f.read().split() if x.lstrip("-").isdigit()]

    valid_lines = th.tensor(raw, dtype=unique_pairs.dtype)
    if valid_lines.numel() != unique_pairs.size(0):
        raise ValueError(
            f"load_edge_map_flat: {valid_lines.numel()} partition IDs vs "
            f"{unique_pairs.size(0)} unique undirected edges."
        )

    # Vectorized key matching via integer-encoded undirected keys.
    # Encode (min,max) as a single int64: key = min * N + max, where N is
    # an upper bound on node ids. This is collision-free for valid ids.
    u_pairs = unique_pairs[:, 0].numpy().astype(np.int64)
    v_pairs = unique_pairs[:, 1].numpy().astype(np.int64)
    part_arr = valid_lines.numpy().astype(np.int64)

    N = int(graph.num_nodes())
    file_keys = u_pairs * N + v_pairs  # already min<=max by construction

    # Sort file keys for searchsorted lookup.
    order = np.argsort(file_keys, kind="stable")
    file_keys_sorted = file_keys[order]
    part_sorted = part_arr[order]

    # Build undirected keys for every graph edge.
    lo = np.minimum(src_np, dst_np)
    hi = np.maximum(src_np, dst_np)
    self_loop_mask = (src_np == dst_np)
    edge_keys = lo * N + hi

    edge_parts = np.full(num_edges, -1, dtype=np.int64)

    # Look up only the non-self-loop edges.
    nz = ~self_loop_mask
    q = edge_keys[nz]
    pos = np.searchsorted(file_keys_sorted, q)
    pos_clipped = np.clip(pos, 0, file_keys_sorted.size - 1)
    found = file_keys_sorted[pos_clipped] == q
    # Assign matches.
    matched_parts = np.full(q.shape[0], -1, dtype=np.int64)
    matched_parts[found] = part_sorted[pos_clipped[found]]

    tmp = edge_parts[nz]
    tmp[:] = matched_parts
    edge_parts[nz] = tmp

    missing_count = int((matched_parts < 0).sum())
    self_loop_count = int(self_loop_mask.sum())

    if self_loop_count > 0:
        print(
            f"[WARN] load_edge_map_flat: {self_loop_count} self-loop(s) "
            f"not covered by the edge-map file; assigning them to partition 0."
        )
        edge_parts[self_loop_mask] = 0
    if missing_count > 0:
        raise Exception(
            f"load_edge_map_flat: {missing_count} graph edges have no match "
            f"in the edge-map file (check dedup / orientation)."
        )

    return th.from_numpy(edge_parts)


def _detect_node_id_offset(records, graph):
    """Detect whether file uses 0- or 1-indexed node IDs."""
    N = int(graph.num_nodes())
    g_src, g_dst = graph.edges()
    existing_keys = set(
        (g_src.numpy().astype(np.int64) * N + g_dst.numpy().astype(np.int64)).tolist()
    )

    sample = records[: min(2000, len(records))]
    if sample:
        s_arr = np.fromiter((r[0] for r in sample), dtype=np.int64, count=len(sample))
        d_arr = np.fromiter((r[1] for r in sample), dtype=np.int64, count=len(sample))
    else:
        s_arr = d_arr = np.empty(0, dtype=np.int64)

    existing_arr = np.fromiter(existing_keys, dtype=np.int64, count=len(existing_keys))
    existing_arr.sort()

    def hit_rate(offset):
        if s_arr.size == 0:
            return 0.0
        keys = (s_arr - offset) * N + (d_arr - offset)
        pos = np.searchsorted(existing_arr, keys)
        pos_c = np.clip(pos, 0, existing_arr.size - 1)
        hits = int((existing_arr[pos_c] == keys).sum())
        return hits / s_arr.size

    rates = {o: hit_rate(o) for o in (0, 1)}
    best_offset = max(rates, key=rates.get)
    best_rate = rates[best_offset]

    if best_rate < 0.5:
        raise Exception(
            f"load_edge_list_labeled: edge-match rate is only {best_rate:.1%} "
            f"(offset 0: {rates[0]:.1%}, offset 1: {rates[1]:.1%}). "
            f"Node IDs in the file don't align with the graph's node IDs."
        )

    if best_offset != 0:
        print(
            f"  load_edge_list_labeled: detected node-ID offset={best_offset} "
            f"(hit rate {best_rate:.1%}). Auto-correcting."
        )
    else:
        print(
            f"  load_edge_list_labeled: node IDs aligned with graph "
            f"(hit rate {best_rate:.1%})."
        )

    return best_offset


def load_edge_list_labeled(graph, input_file_path: Path):
    """File format: each non-comment line = `src dst part_id`.

    May list only one direction per undirected edge; we mirror the
    partition onto the reverse direction if it exists in the graph.
    Every directed edge of `graph` must end up assigned.
    """
    if not input_file_path.is_file():
        raise Exception(f"Not a file: {input_file_path}")

    records = []
    with open(input_file_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"[WARN] Skipping malformed line {lineno}: {line!r}")
                continue
            try:
                s, d, p = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                print(f"[WARN] Non-integer values on line {lineno}: {line!r}")
                continue
            records.append((s, d, p))

    if not records:
        raise Exception("load_edge_list_labeled: no valid edges found in file.")

    offset = _detect_node_id_offset(records, graph)

    num_edges = int(graph.num_edges())
    key_to_eid, src_np, dst_np = _build_edge_key_to_index(graph)
    edge_parts = np.full(num_edges, -1, dtype=np.int64)

    n_forward_hits = 0
    n_reverse_hits = 0
    n_missing = 0

    for raw_s, raw_d, pid in records:
        s = raw_s - offset
        d = raw_d - offset

        hit_any = False
        fwd_eid = key_to_eid.get((s, d))
        if fwd_eid is not None:
            edge_parts[fwd_eid] = pid
            n_forward_hits += 1
            hit_any = True

        if s != d:
            rev_eid = key_to_eid.get((d, s))
            if rev_eid is not None:
                edge_parts[rev_eid] = pid
                n_reverse_hits += 1
                hit_any = True

        if not hit_any:
            n_missing += 1

    print(
        f"  load_edge_list_labeled: {n_forward_hits} forward, "
        f"{n_reverse_hits} reverse hits; {n_missing} file edges not in graph."
    )

    unassigned = int((edge_parts < 0).sum())
    if unassigned > 0:
        raise Exception(
            f"load_edge_list_labeled: {unassigned} graph edges have no "
            f"partition assignment after processing the file."
        )

    return th.from_numpy(edge_parts)


def load_edge_list_paired(graph, input_file_path: Path):
    """File format: each non-comment line = `src dst part_id`, one line
    per directed edge, in the same order as `graph.edges()`.
    """
    if not input_file_path.is_file():
        raise Exception(f"Not a file: {input_file_path}")

    part_ids = []
    with open(input_file_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                print(f"[WARN] Skipping malformed line {lineno}: {line!r}")
                continue
            try:
                part_ids.append(int(parts[2]))
            except ValueError:
                print(f"[WARN] Non-integer part_id on line {lineno}: {line!r}")

    part_ids_arr = np.asarray(part_ids, dtype=np.int64)
    num_edges = int(graph.num_edges())
    if part_ids_arr.shape[0] != num_edges:
        raise Exception(
            f"load_edge_list_paired: {part_ids_arr.shape[0]} partition entries "
            f"vs {num_edges} graph edges."
        )

    if part_ids_arr.min() < 0:
        raise Exception(
            f"load_edge_list_paired: negative part_id found "
            f"(min={part_ids_arr.min()})."
        )

    print(
        f"  load_edge_list_paired: paired {part_ids_arr.shape[0]} partition "
        f"assignments with graph.edges() in native order."
    )

    return th.from_numpy(part_ids_arr)
