#!/usr/bin/env python3
"""
graph_prep.py -- the *non*-DRPA preprocessing that the training script was
doing at every start-up, moved to where it can be done once.

These things are partition-independent, so they belong in the graph
download / preparation scripts (before partitioning):

  1. Feature-name canonicalisation to `feat` / `label` / `train_mask` /
     `val_mask` / `test_mask` -- the trainer's canonical role names. The
     converter writes namespaced keys ("_N/<key>") derived from whatever
     the full graph carries; if the names are canonical on disk, the
     trainer's alias table is dead code in the hot path.
  2. float64 -> float32 downcast. Halves the largest tensor in the job.
     Doing it once on disk also removes the risk of the partition and the
     full graph disagreeing on dtype.
  3. CSC materialisation. Without CSC, SAGEConv's update_all falls back to
     materialising per-edge messages (~70 GB/rank on a 28M-edge partition
     with 602-d features).

     IMPORTANT: for the FULL downloaded graph, CSC must be materialised
     WITHOUT dropping COO -- the vertex-cut converter asserts that 'coo'
     is available for partitioning (`save_edge_partition_graph`). The
     CSC-*only* restriction is correct for the per-partition graphs and
     is applied by the converter (and, as a fallback, by the trainer's
     load_partition). Hence to_csc(restrict=...) below.

The alias resolver stays exported so the trainer can keep a
backwards-compatible fallback for graphs prepared with an older script.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch as th

__all__ = [
    "FEATURE_ALIASES",
    "NS_PREFIXES",
    "ROLES",
    "resolve_feature_key",
    "canonicalize_ndata",
    "downcast_float64",
    "to_csc",
    "prepare_graph_for_training",
]

FEATURE_ALIASES = {
    "feat":       ["feat", "feats", "features", "feature", "x", "X", "node_feat"],
    "label":      ["label", "labels", "y", "Y", "node_label", "class"],
    "train_mask": ["train_mask", "train_idx", "train"],
    "val_mask":   ["val_mask", "valid_mask", "val_idx", "valid_idx", "val", "valid"],
    "test_mask":  ["test_mask", "test_idx", "test"],
}
NS_PREFIXES = ("", "_N/", "_N:")
ROLES = ("feat", "label", "train_mask", "val_mask", "test_mask")

_REQUIRED = ("feat", "label", "train_mask")


def resolve_feature_key(node_feats, role: str) -> Optional[str]:
    """First key in `node_feats` matching `role` under any alias/prefix."""
    for alias in FEATURE_ALIASES.get(role, [role]):
        for p in NS_PREFIXES:
            k = p + alias
            if k in node_feats:
                return k
    return None


def canonicalize_ndata(ndata, roles: Iterable[str] = ROLES,
                       required: Iterable[str] = _REQUIRED,
                       bool_masks: bool = True,
                       int64_labels: bool = True,
                       squeeze_labels: bool = True,
                       log=None) -> None:
    """
    Rename every recognised key to its canonical role name, in place.
    Also normalises mask dtype to bool and label dtype to int64 -- both
    were being fixed up at training start-up -- and squeezes (N, 1)
    label matrices to (N,), which the old download scripts did in their
    normalize_keys(). Genuine multilabel (N, C>1) labels are left alone.
    """
    required = set(required)
    for role in roles:
        key = resolve_feature_key(ndata, role)
        if key is None:
            if role in required:
                raise KeyError(
                    f"Could not resolve role '{role}'. Available keys: "
                    f"{list(ndata.keys())}")
            continue
        if key != role:
            ndata[role] = ndata[key]
            del ndata[key]
            if log:
                log(f"renamed '{key}' -> '{role}'")

    if squeeze_labels and "label" in ndata:
        lbl = ndata["label"]
        if lbl.dim() == 2 and lbl.shape[1] == 1:
            ndata["label"] = lbl.squeeze(1)
            if log:
                log("'label': squeezed (N, 1) -> (N,)")
    if bool_masks:
        for m in ("train_mask", "val_mask", "test_mask"):
            if m in ndata and ndata[m].dtype != th.bool:
                ndata[m] = ndata[m].bool()
                if log:
                    log(f"'{m}': cast to bool")
    if int64_labels and "label" in ndata and ndata["label"].dtype != th.int64:
        ndata["label"] = ndata["label"].long()
        if log:
            log("'label': cast to int64")


def downcast_float64(ndata, keys: Iterable[str] = ("feat",), log=None) -> None:
    """float64 -> float32 for the named tensors, in place."""
    for k in keys:
        t = ndata.get(k)
        if t is not None and t.dtype == th.float64:
            ndata[k] = t.to(th.float32)
            if log:
                log(f"'{k}': downcast float64 -> float32")


def to_csc(graph, restrict: bool = True, log=None):
    """
    Materialise CSC. Returns the graph -- DGL's .formats() returns a new
    object.

    restrict=True  : CSC only, drop CSR/COO. Correct for per-partition
                     graphs (SAGEConv only needs CSC; keeping COO/CSR
                     wastes RAM). Used by the converter and by the
                     trainer's load_partition fallback.
    restrict=False : materialise CSC while KEEPING COO available.
                     Correct for the full downloaded graph -- the
                     vertex-cut converter needs COO for partitioning.

    Fallback chain (target DGL fork may not have all three APIs):
    .formats([...]) -> create_formats_() -> warn.
    """
    if restrict:
        try:
            graph = graph.formats(["csc"])
            if log:
                log("graph format set to CSC only")
        except Exception:
            try:
                graph.create_formats_()
                if log:
                    log("create_formats_() fallback used")
            except AttributeError:
                if log:
                    log("WARNING: no format API in this DGL; the first "
                        "forward will materialise per-edge messages and "
                        "likely OOM")
        return graph

    # Full-graph path: keep COO for the converter, add CSC for eval.
    try:
        graph = graph.formats(["coo", "csc"])
        try:
            graph.create_formats_()
            if log:
                log("materialised CSC (COO kept for the converter)")
        except AttributeError:
            if log:
                log("WARNING: create_formats_() unavailable; allowed "
                    "formats restricted to coo+csc but CSC not "
                    "materialised (trainer builds it at load)")
    except AttributeError:
        try:
            graph.create_formats_()
            if log:
                log("create_formats_() fallback: all formats materialised")
        except AttributeError:
            if log:
                log("WARNING: no format API in this DGL; CSC not "
                    "materialised (trainer builds it at load)")
    return graph


def prepare_graph_for_training(graph, *, fp32: bool = True, csc: bool = True,
                               restrict_csc: bool = False, log=None):
    """
    One-call preparation for the download / preparation scripts.
    Canonical names + optional fp32 downcast + optional CSC.

    restrict_csc=False (default) keeps COO available, which the
    vertex-cut converter requires on the full graph. Only pass
    restrict_csc=True for a graph that will never be partitioned again.
    """
    canonicalize_ndata(graph.ndata, log=log)
    if fp32:
        downcast_float64(graph.ndata, log=log)
    if csc:
        graph = to_csc(graph, restrict=restrict_csc, log=log)
    return graph
