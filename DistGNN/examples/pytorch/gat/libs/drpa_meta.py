#!/usr/bin/env python3
"""
drpa_meta.py -- single source of truth for DRPA partition metadata.

Extracted from the GraphSAGE DistGNN training script so that the
*producer* (the vertex-cut converter, edge_partition_dgl_format_saver.py)
and the *consumer* (train_dist_sym.py) share one definition of the
on-disk format, the dtypes and the invariants.

Layout of the responsibilities after the extraction
---------------------------------------------------
Converter (producer), once, offline (inside write_drpa_metadata):
    t = finalize_drpa_tensors(node_map, adj, lf, orig_nid)
    validate_drpa_meta(t["node_map"], t["adj"], t["lf"],
                       inner_node=inner, rank=p, num_parts=P,
                       orig_nid=t["orig_nid"], n_full=N)   # per partition
    write_drpa_meta(out_dir / f"part{p}" / DRPA_META_FILENAME,
                    **t, rank=p, num_parts=P, validated=True,
                    finalize=False)
    register_drpa_meta(part_metadata, p, rel_path)          # into the JSON
    ...
    validate_leaders_consistent_local(parts)                # across parts

Training (consumer), every run:
    node_map = attach_drpa_meta_to_graph(graph, part_metadata,
                                         part_config_parent, rank,
                                         num_parts)
    -> no dtype casts, no assertions, no all_gather in the hot path.

The trainer keeps the ability to re-check (`trust_meta=False`, exposed
as --recheck-drpa), but the default is to trust metadata that the
converter has already validated (`validated=True` in the file).

Format history (matches edge_partition_dgl_format_saver.py)
-----------------------------------------------------------
v1: node_map / adj / lf
v2: + orig_nid (per-local-node original NID, for correctness
    validation against the full graph)
v3: lf semantics fixed -- a mirror's leader is now always the node's
    MASTER partition (previously a free choice among copies).
    node_map / adj / orig_nid unchanged.
v4: + "validated" flag (converter ran validate_drpa_meta before
    writing), + rank / num_parts / n_total self-description,
    + "_drpa_master_marker" recorded in the file. Tensor content and
    dtypes are unchanged relative to v3 (the converter has produced
    int32/int32/int32/int64 since v1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch as th

__all__ = [
    "DRPA_MASTER_MARKER",
    "DRPA_FORMAT_VERSION",
    "DRPA_META_FILENAME",
    "finalize_drpa_tensors",
    "write_drpa_meta",
    "register_drpa_meta",
    "validate_drpa_meta",
    "validate_leaders_consistent_local",
    "validate_leaders_consistent_dist",
    "drpa_meta_path",
    "load_drpa_meta",
    "attach_drpa_meta_to_graph",
]

# Master marker in `lf`. The C++ asserts `lf_val != -200` for every
# selected (mirror) node, i.e. -200 means "this node is a master".
# Lives in the DGL_XEON C++ side (libra2dgl_built_adj_v2 /
# fdrpa_get_buckets_v4); if the fork ever uses a different sentinel,
# change it HERE only -- converter and trainer both import it.
DRPA_MASTER_MARKER = -200

DRPA_FORMAT_VERSION = 4

# Filename written into each part{p}/ directory. Do not rename: the
# trainer's no-JSON-entry fallback probes for exactly this name.
DRPA_META_FILENAME = "drpa_meta.pt"

# Canonical dtypes. DRPA's C kernels expect 32-bit ids; orig_nid indexes
# the full graph and stays 64-bit.
_DT_NODE_MAP = th.int32
_DT_ADJ = th.int32
_DT_LF = th.int32
_DT_ORIG_NID = th.int64


# ===========================================================================
# Producer side -- converter
# ===========================================================================
def finalize_drpa_tensors(node_map,
                          adj,
                          lf,
                          orig_nid=None,
                          ) -> Dict[str, Optional[th.Tensor]]:
    """
    Normalise dtypes / contiguity once, in the converter.

    Previously the trainer did `.to(th.int32)` on every tensor at every
    start-up. Doing it here means the file on disk is already in the
    shape DRPA wants, and the load path becomes a pure read.

    Accepts torch tensors or numpy arrays. Raises on values that do not
    fit the target dtype rather than silently wrapping around.
    """
    def _to_tensor(t, name):
        if isinstance(t, th.Tensor):
            return t
        try:
            return th.as_tensor(t)
        except Exception as e:
            raise TypeError(f"{name} must be tensor-like, got "
                            f"{type(t)}: {e}")

    def _cast(t, dt: th.dtype, name: str) -> th.Tensor:
        t = _to_tensor(t, name)
        if t.numel() and dt in (th.int32,):
            info = th.iinfo(dt)
            lo, hi = int(t.min()), int(t.max())
            if lo < info.min or hi > info.max:
                raise OverflowError(
                    f"{name} range [{lo}, {hi}] does not fit into {dt}")
        return t.to(dt).contiguous()

    out: Dict[str, Optional[th.Tensor]] = {
        "node_map": _cast(_to_tensor(node_map, "node_map").reshape(-1),
                          _DT_NODE_MAP, "node_map"),
        "adj": _cast(adj, _DT_ADJ, "adj"),
        "lf": _cast(_to_tensor(lf, "lf").reshape(-1), _DT_LF, "lf"),
        "orig_nid": None,
    }
    if orig_nid is not None:
        out["orig_nid"] = _cast(_to_tensor(orig_nid, "orig_nid").reshape(-1),
                                _DT_ORIG_NID, "orig_nid")
    return out


def write_drpa_meta(path,
                    node_map: th.Tensor,
                    adj: th.Tensor,
                    lf: th.Tensor,
                    orig_nid: Optional[th.Tensor] = None,
                    *,
                    rank: Optional[int] = None,
                    num_parts: Optional[int] = None,
                    validated: bool = False,
                    finalize: bool = True,
                    extra: Optional[dict] = None) -> Path:
    """
    Write one partition's drpa_meta.pt. Returns the path written.

    `finalize=True` runs finalize_drpa_tensors first, so callers can hand
    over whatever dtype their partitioner produced.

    `extra` is merged into the saved dict AFTER the standard keys --
    used by the converter to keep provenance fields like "_seed" that
    are not part of the format proper. Extra keys must not collide with
    standard keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if finalize:
        t = finalize_drpa_tensors(node_map, adj, lf, orig_nid)
    else:
        t = {"node_map": node_map, "adj": adj, "lf": lf, "orig_nid": orig_nid}

    blob = {
        "_drpa_format_version": DRPA_FORMAT_VERSION,
        "_drpa_master_marker": DRPA_MASTER_MARKER,
        "node_map": t["node_map"],
        "adj": t["adj"],
        "lf": t["lf"],
        "validated": bool(validated),
        "rank": rank,
        "num_parts": num_parts if num_parts is not None
        else int(t["node_map"].numel()),
        "n_total": int(t["lf"].numel()),
    }
    if t["orig_nid"] is not None:
        blob["orig_nid"] = t["orig_nid"]
    if extra:
        for k in extra:
            if k in blob:
                raise ValueError(f"extra key {k!r} collides with a "
                                 f"standard drpa_meta key")
        blob.update(extra)

    th.save(blob, path.as_posix())
    return path


def register_drpa_meta(part_metadata: dict,
                       rank: int,
                       rel_path) -> None:
    """
    Record the metadata in the partition JSON so the trainer never has to
    guess a path. Additive only: existing key names from older converter
    versions ("present", "filename", "format_version", "seed") are kept.

    Writes:
        part_metadata["part-<rank>"]["drpa_meta"]      = "<rel_path>"
        part_metadata["_drpa_meta"]["present"]         = True
        part_metadata["_drpa_meta"]["filename"]        = basename
        part_metadata["_drpa_meta"]["format_version"]  = DRPA_FORMAT_VERSION
        part_metadata["_drpa_meta"]["master_marker"]   = DRPA_MASTER_MARKER
        part_metadata["_drpa_meta"]["parts"]           = sorted list of ranks
    """
    rel_path = Path(rel_path)
    entry = part_metadata.setdefault(f"part-{rank}", {})
    entry["drpa_meta"] = rel_path.as_posix()

    glob = part_metadata.setdefault("_drpa_meta", {})
    glob["present"] = True
    glob.setdefault("filename", rel_path.name)
    glob["format_version"] = DRPA_FORMAT_VERSION
    glob["master_marker"] = DRPA_MASTER_MARKER
    written = set(glob.get("parts", []))
    written.add(int(rank))
    glob["parts"] = sorted(written)


# ===========================================================================
# Validation -- runs in the converter, optionally in the trainer
# ===========================================================================
def validate_drpa_meta(node_map: th.Tensor,
                       adj: th.Tensor,
                       lf: th.Tensor,
                       inner_node: th.Tensor,
                       rank: int,
                       num_parts: int,
                       orig_nid: Optional[th.Tensor] = None,
                       n_full: Optional[int] = None) -> None:
    """
    Per-partition invariant check. Pure tensor code -- no DGL graph, no
    torch.distributed -- so the converter can call it before writing.

    These are exactly the invariants the trainer's old
    validate_drpa_metadata asserted at every start-up, plus:
      * negative adj entries must be exactly -1 (the padding value),
      * optional orig_nid length / range checks (`n_full` = number of
        nodes in the original graph; skipped when None).

    `inner_node` is the partition's inner_node vector (1 = master).
    Raises AssertionError naming the partition and the violated
    invariant.
    """
    n_total = int(inner_node.numel())
    inner_mask = inner_node.bool()

    if node_map.dtype != _DT_NODE_MAP or tuple(node_map.shape) != (num_parts,):
        raise AssertionError(
            f"part {rank}: node_map must be {_DT_NODE_MAP} of shape "
            f"({num_parts},), got {node_map.dtype} {tuple(node_map.shape)}")

    nm = node_map.tolist()
    if any(nm[i] > nm[i + 1] for i in range(num_parts - 1)):
        raise AssertionError(f"part {rank}: node_map not non-decreasing: {nm}")

    expected = nm[rank] - (nm[rank - 1] if rank > 0 else 0)
    if expected != n_total:
        raise AssertionError(
            f"part {rank}: node_map says this part has {expected} nodes, "
            f"got {n_total}")

    width = max(num_parts - 1, 1)
    if adj.dtype != _DT_ADJ or tuple(adj.shape) != (n_total, width):
        raise AssertionError(
            f"part {rank}: adj must be {_DT_ADJ} of shape "
            f"({n_total}, {width}), got {adj.dtype} {tuple(adj.shape)}")
    if lf.dtype != _DT_LF or tuple(lf.shape) != (n_total,):
        raise AssertionError(
            f"part {rank}: lf must be {_DT_LF} of shape ({n_total},), got "
            f"{lf.dtype} {tuple(lf.shape)}")

    # Masters carry the marker, mirrors carry a real leader id.
    m_lf = lf[inner_mask]
    if m_lf.numel() > 0 and (int(m_lf.min()) != DRPA_MASTER_MARKER or
                             int(m_lf.max()) != DRPA_MASTER_MARKER):
        raise AssertionError(
            f"part {rank}: masters must all carry the "
            f"{DRPA_MASTER_MARKER} marker in lf")
    mir_lf = lf[~inner_mask]
    if mir_lf.numel() > 0 and int(mir_lf.min()) < 0:
        raise AssertionError(
            f"part {rank}: mirrors must have non-negative lf")

    # A mirror's leader must appear in its own adj row when it is remote.
    # (Since format v3 the leader is always the master, so it is always
    # remote from a mirror's point of view -- this covers every mirror.)
    mir_idx = th.where(~inner_mask)[0]
    if mir_idx.numel() > 0:
        lf_vals = lf[mir_idx]
        leader_part = th.searchsorted(node_map.to(th.int64),
                                      lf_vals.to(th.int64), right=True)
        remote = leader_part != rank
        ci, cl = mir_idx[remote], lf_vals[remote]
        if ci.numel() > 0:
            rows = adj[ci]
            hit = (rows == cl.unsqueeze(1)).any(dim=1)
            if not bool(hit.all().item()):
                bad = int((~hit).sum().item())
                first = int(ci[~hit][0].item())
                raise AssertionError(
                    f"part {rank}: {bad} mirror(s) with lf not in their adj "
                    f"row (first: local {first}, lf={int(lf[first])}, "
                    f"adj={adj[first].tolist()}). This would trip the "
                    f"assert(flg == 0) in fdrpa_get_buckets_v4.")

    # adj holds reshuffled ids into the concatenated copy space; padding
    # must be exactly -1.
    total_copies = nm[-1]
    flat = adj.reshape(-1)
    neg = flat[flat < 0]
    if neg.numel() > 0 and int(neg.min()) != -1:
        raise AssertionError(
            f"part {rank}: adj padding must be -1, found {int(neg.min())}")
    nonpad = flat[flat >= 0]
    if nonpad.numel() > 0 and int(nonpad.max()) >= total_copies:
        raise AssertionError(
            f"part {rank}: adj id {int(nonpad.max())} out of range "
            f"(total copies {total_copies})")

    if orig_nid is not None:
        if int(orig_nid.numel()) != n_total:
            raise AssertionError(
                f"part {rank}: orig_nid has {int(orig_nid.numel())} entries, "
                f"expected {n_total}")
        if orig_nid.numel() and int(orig_nid.min()) < 0:
            raise AssertionError(f"part {rank}: orig_nid must be non-negative")
        if n_full is not None and orig_nid.numel() and \
                int(orig_nid.max()) >= n_full:
            raise AssertionError(
                f"part {rank}: orig_nid max {int(orig_nid.max())} >= "
                f"n_full {n_full}")


def validate_leaders_consistent_local(
        parts: Sequence[Tuple[th.Tensor, th.Tensor, th.Tensor]]) -> None:
    """
    Cross-partition leader agreement, checked in ONE process.

    This is the converter-side replacement for the trainer's
    all_gather-based check: the converter already holds every partition,
    so the same guarantee costs no communication at all.

    `parts` is an iterable of (reshuffled_nid, lf, inner_node) triples,
    one per partition, in rank order.
    """
    pairs: List[th.Tensor] = []
    for gids, lf, inner in parts:
        mir = th.where(~inner.bool())[0]
        if mir.numel() == 0:
            continue
        pairs.append(th.stack([gids[mir].to(th.int64),
                               lf[mir].to(th.int64)], dim=1))
    if not pairs:
        return
    _assert_no_leader_conflict(th.cat(pairs, dim=0))


def validate_leaders_consistent_dist(lf: th.Tensor,
                                     gids: th.Tensor,
                                     inner_node: th.Tensor,
                                     rank: int,
                                     world_size: int,
                                     dist) -> None:
    """
    Distributed variant, kept for the trainer's --check-drpa-consistency
    escape hatch. `dist` is torch.distributed, injected so this module
    never imports it. Prefer validate_leaders_consistent_local() in the
    converter.
    """
    if world_size <= 1:
        return
    inner = inner_node.bool()
    mir = th.where(~inner)[0]
    local = th.stack([gids[mir].to(th.int64), lf[mir].to(th.int64)], dim=1) \
        if mir.numel() > 0 else th.zeros((0, 2), dtype=th.int64)

    counts = [th.zeros(1, dtype=th.int64) for _ in range(world_size)]
    dist.all_gather(counts, th.tensor([local.shape[0]], dtype=th.int64))
    max_n = max(int(c.item()) for c in counts)
    if max_n == 0:
        return

    padded = th.full((max_n, 2), -1, dtype=th.int64)
    if local.shape[0] > 0:
        padded[:local.shape[0]] = local
    gathered = [th.zeros((max_n, 2), dtype=th.int64) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    all_pairs = th.cat([gathered[r][:int(counts[r].item())]
                        for r in range(world_size)], dim=0)
    if all_pairs.numel():
        _assert_no_leader_conflict(all_pairs)


def _assert_no_leader_conflict(all_pairs: th.Tensor) -> None:
    """Sort (nid, leader) pairs; flag equal nids with differing leaders."""
    order = th.argsort(all_pairs[:, 0])
    sp = all_pairs[order]
    conflict = (sp[1:, 0] == sp[:-1, 0]) & (sp[1:, 1] != sp[:-1, 1])
    n_conflict = int(conflict.sum().item())
    if n_conflict:
        i = int(conflict.nonzero(as_tuple=True)[0][0].item())
        raise AssertionError(
            f"DRPA leader consistency FAILED: {n_conflict} split nodes have "
            f"inconsistent leaders. First conflict: reshuffled-NID "
            f"{int(sp[i, 0])} has leaders {int(sp[i, 1])} vs "
            f"{int(sp[i + 1, 1])}.")


# ===========================================================================
# Consumer side -- training
# ===========================================================================
def drpa_meta_path(part_metadata: dict, part_config_parent,
                   rank: int) -> Path:
    parent = Path(part_config_parent)
    rel = part_metadata.get(f"part-{rank}", {}).get("drpa_meta")
    return (parent / rel) if rel \
        else (parent / f"part{rank}" / DRPA_META_FILENAME)


def load_drpa_meta(part_metadata: dict,
                   part_config_parent,
                   rank: int,
                   num_parts: int,
                   require_orig_nid: bool = False,
                   ) -> Dict[str, Optional[th.Tensor]]:
    """
    Read one partition's metadata. v4 files are already in the canonical
    dtypes AND already validated; older files (v1-v3, written by the
    previous converter) are cast here and carry validated=False, so the
    caller re-validates -- exactly the old trainer behaviour. Old
    artefacts therefore run without re-conversion; they only lose the
    start-up saving.

    Returns a dict with keys: node_map, adj, lf, orig_nid, version,
    validated.
    """
    glob = part_metadata.get("_drpa_meta")
    if glob is not None and not glob.get("present", False):
        raise FileNotFoundError(
            "JSON says DRPA metadata is absent; re-run the converter with "
            "write_drpa_meta=True or pass --nr -1.")

    path = drpa_meta_path(part_metadata, part_config_parent, rank)
    if not path.exists():
        raise FileNotFoundError(
            f"DRPA enabled but no metadata at {path}. Re-run the vertex-cut "
            f"converter with write_drpa_meta=True, or pass --nr -1 to "
            f"disable DRPA.")

    blob = th.load(path.as_posix())
    for k in ("node_map", "adj", "lf"):
        if k not in blob:
            raise KeyError(f"DRPA meta {path} missing '{k}'.")

    version = blob.get("_drpa_format_version", 1)
    marker = blob.get("_drpa_master_marker")
    if marker is not None and int(marker) != DRPA_MASTER_MARKER:
        raise ValueError(
            f"DRPA meta {path} was written with master marker {marker}, "
            f"this build expects {DRPA_MASTER_MARKER}.")

    orig_nid = blob.get("orig_nid")
    if orig_nid is None and require_orig_nid:
        raise KeyError(
            f"DRPA meta v{version} has no orig_nid; --validate-drpa needs "
            f"it. Re-run the converter (it now writes v{DRPA_FORMAT_VERSION}).")

    stored_parts = blob.get("num_parts")
    if stored_parts is not None and int(stored_parts) != int(num_parts):
        raise ValueError(
            f"DRPA meta was built for {stored_parts} partitions, this run "
            f"has {num_parts}.")
    stored_rank = blob.get("rank")
    if stored_rank is not None and int(stored_rank) != int(rank):
        raise ValueError(
            f"DRPA meta at {path} says rank {stored_rank}, loaded as "
            f"rank {rank}.")

    return {
        "node_map": blob["node_map"].to(_DT_NODE_MAP),
        "adj": blob["adj"].to(_DT_ADJ),
        "lf": blob["lf"].to(_DT_LF),
        "orig_nid": None if orig_nid is None else orig_nid.to(_DT_ORIG_NID),
        "version": version,
        "validated": bool(blob.get("validated", False)),
    }


def attach_drpa_meta_to_graph(graph,
                              part_metadata: dict,
                              part_config_parent,
                              rank: int,
                              num_parts: int,
                              *,
                              require_orig_nid: bool = False,
                              trust_meta: bool = True,
                              log=None) -> th.Tensor:
    """
    Load + attach in one call. Returns node_map.

    `trust_meta=True` (default) skips the invariant check when the file
    carries validated=True (the converter already ran it). Files without
    the flag -- everything the previous converter wrote -- are always
    validated, as before. Pass trust_meta=False (--recheck-drpa) to
    re-validate v4 files at start-up too.
    """
    meta = load_drpa_meta(part_metadata, part_config_parent, rank, num_parts,
                          require_orig_nid=require_orig_nid)

    if not (trust_meta and meta["validated"]):
        validate_drpa_meta(meta["node_map"], meta["adj"], meta["lf"],
                           graph.ndata["inner_node"], rank, num_parts,
                           orig_nid=meta["orig_nid"])

    if not hasattr(graph, "dstdata"):
        graph.dstdata = {}
    graph.dstdata["adj"] = meta["adj"]
    graph.dstdata["lf"] = meta["lf"]

    if meta["orig_nid"] is not None:
        if int(meta["orig_nid"].numel()) != int(graph.num_nodes()):
            raise RuntimeError("orig_nid length mismatch")
        graph.ndata["orig_nid"] = meta["orig_nid"]

    if log is not None and rank == 0:
        log(f"DRPA meta v{meta['version']} "
            f"(validated={meta['validated']}, "
            f"revalidated={not (trust_meta and meta['validated'])}): "
            f"node_map={meta['node_map'].tolist()}, "
            f"adj{tuple(meta['adj'].shape)}, lf{tuple(meta['lf'].shape)}, "
            f"orig_nid={'yes' if meta['orig_nid'] is not None else 'no'}")
    return meta["node_map"]
