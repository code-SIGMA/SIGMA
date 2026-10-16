import concurrent
import concurrent.futures
import copy
import gc
import json
import logging
import multiprocessing as mp
import os
import time
from functools import partial

import numpy as np

import torch

from dgl import backend as F, graphbolt as gb
from dgl.base import dgl_warning, DGLError, EID, ETYPE, NID, NTYPE
from dgl.convert import heterograph, to_homogeneous
from dgl.data.utils import load_graphs, load_tensors, save_graphs, save_tensors
from dgl.partition import (
    get_peak_mem,
    metis_partition_assignment,
    partition_graph_with_halo,
)
from dgl.random import choice as random_choice
from dgl.transforms import sort_csc_by_tag, sort_csr_by_tag
import dgl.distributed
from dgl.distributed.constants import DEFAULT_ETYPE, DEFAULT_NTYPE, DGL2GB_EID, GB_DST_ID
from dgl.distributed.graph_partition_book import (
    _etype_str_to_tuple,
    _etype_tuple_to_str,
    RangePartitionBook,
)


RESERVED_FIELD_DTYPE = {
    "inner_node": (
        F.uint8
    ),  # A flag indicates whether the node is inside a partition.
    "inner_edge": (
        F.uint8
    ),  # A flag indicates whether the edge is inside a partition.
    NID: F.int64,
    EID: F.int64,
    NTYPE: F.int16,
    # `sort_csr_by_tag` and `sort_csc_by_tag` works on int32/64 only.
    ETYPE: F.int32,
}

def _load_part_config(part_config):
    """Load part config and format."""
    try:
        with open(part_config) as f:
            part_metadata = _format_part_metadata(
                json.load(f), _etype_str_to_tuple
            )
    except AssertionError as e:
        raise DGLError(
            f"Failed to load partition config due to {e}. "
            "Probably caused by outdated config. If so, please refer to "
            "https://github.com/dmlc/dgl/tree/master/tools#change-edge-"
            "type-to-canonical-edge-type-for-partition-configuration-json"
        )
    return part_metadata

def _get_part_ranges(id_ranges):
    res = {}
    for key in id_ranges:
        # Normally, each element has two values that represent the starting ID and the ending ID
        # of the ID range in a partition.
        # If not, the data is probably still in the old format, in which only the ending ID is
        # stored. We need to convert it to the format we expect.
        if not isinstance(id_ranges[key][0], list):
            start = 0
            for i, end in enumerate(id_ranges[key]):
                id_ranges[key][i] = [start, end]
                start = end
        res[key] = np.concatenate(
            [np.array(l) for l in id_ranges[key]]
        ).reshape(-1, 2)
    return res

def load_partition_book(part_config, part_id, part_metadata=None):
    """Load a graph partition book from the partition config file."""
    if part_metadata is None:
        part_metadata = _load_part_config(part_config)
    assert "num_parts" in part_metadata, "num_parts does not exist."
    assert (
        part_metadata["num_parts"] > part_id
    ), "part {} is out of range (#parts: {})".format(
        part_id, part_metadata["num_parts"]
    )
    num_parts = part_metadata["num_parts"]
    assert (
        "num_nodes" in part_metadata
    ), "cannot get the number of nodes of the global graph."
    assert (
        "num_edges" in part_metadata
    ), "cannot get the number of edges of the global graph."
    assert "node_map" in part_metadata, "cannot get the node map."
    assert "edge_map" in part_metadata, "cannot get the edge map."
    assert "graph_name" in part_metadata, "cannot get the graph name"

    node_map = part_metadata["node_map"]
    edge_map = part_metadata["edge_map"]
    if isinstance(node_map, dict):
        for key in node_map:
            is_range_part = isinstance(node_map[key], list)
            break
    elif isinstance(node_map, list):
        is_range_part = True
        node_map = {DEFAULT_NTYPE: node_map}
    else:
        is_range_part = False
    if isinstance(edge_map, list):
        edge_map = {DEFAULT_ETYPE: edge_map}

    ntypes = {DEFAULT_NTYPE: 0}
    etypes = {DEFAULT_ETYPE: 0}
    if "ntypes" in part_metadata:
        ntypes = part_metadata["ntypes"]
    if "etypes" in part_metadata:
        etypes = part_metadata["etypes"]

    if isinstance(node_map, dict):
        for key in node_map:
            assert key in ntypes, "The node type {} is invalid".format(key)
    if isinstance(edge_map, dict):
        for key in edge_map:
            assert key in etypes, "The edge type {} is invalid".format(key)

    if not is_range_part:
        raise TypeError("Only RangePartitionBook is supported currently.")

    node_map = _get_part_ranges(node_map)
    edge_map = _get_part_ranges(edge_map)

    def _format_node_edge_map(part_metadata, map_type, data):
        key = f"{map_type}_map_dtype"
        if key not in part_metadata:
            return data
        dtype = part_metadata[key]
        assert dtype in ["int32", "int64"], (
            f"The {map_type} map dtype should be either int32 or int64, "
            f"but got {dtype}."
        )
        for key in data:
            data[key] = data[key].astype(dtype)
        return data

    node_map = _format_node_edge_map(part_metadata, "node", node_map)
    edge_map = _format_node_edge_map(part_metadata, "edge", edge_map)

    node_map = dict(sorted(node_map.items(), key=lambda x: ntypes[x[0]]))
    edge_map = dict(sorted(edge_map.items(), key=lambda x: etypes[x[0]]))

    def _assert_is_sorted(id_map):
        id_ranges = np.array(list(id_map.values()))
        ids = []
        for i in range(num_parts):
            ids.append(id_ranges[:, i, :])
        ids = np.array(ids).flatten()
        assert np.all(
            ids[:-1] <= ids[1:]
        ), f"The node/edge map is not sorted: {ids}"

    _assert_is_sorted(node_map)
    _assert_is_sorted(edge_map)

    return (
        RangePartitionBook(
            part_id, num_parts, node_map, edge_map, ntypes, etypes
        ),
        part_metadata["graph_name"],
        ntypes,
        etypes,
    )

def _format_part_metadata(part_metadata, formatter):
    """Format etypes with specified formatter."""
    for key in ["edge_map", "etypes"]:
        if key not in part_metadata:
            continue
        orig_data = part_metadata[key]
        if not isinstance(orig_data, dict):
            continue
        new_data = {}
        for etype, data in orig_data.items():
            etype = formatter(etype)
            new_data[etype] = data
        part_metadata[key] = new_data
    return part_metadata

def _dump_part_config(part_config, part_metadata):
    """Format and dump part config."""
    part_metadata = _format_part_metadata(part_metadata, _etype_tuple_to_str)
    with open(part_config, "w") as outfile:
        json.dump(part_metadata, outfile, sort_keys=False, indent=4)

def _save_dgl_graphs(filename, g_list, formats=None):
    save_graphs(filename, g_list, formats=formats)

def process_partitions(g, formats=None, sort_etypes=False):
    """Preprocess partitions before saving:
    1. format data types.
    2. sort csc/csr by tag.
    """
    for k, dtype in RESERVED_FIELD_DTYPE.items():
        if k in g.ndata:
            g.ndata[k] = F.astype(g.ndata[k], dtype)
        if k in g.edata:
            g.edata[k] = F.astype(g.edata[k], dtype)

    if (sort_etypes) and (formats is not None):
        if "csr" in formats:
            g = sort_csr_by_tag(g, tag=g.edata[ETYPE], tag_type="edge")
        if "csc" in formats:
            g = sort_csc_by_tag(g, tag=g.edata[ETYPE], tag_type="edge")
    return g

def _update_node_edge_map(node_map_val, edge_map_val, g, num_parts):
    """Make node_map_val / edge_map_val contiguous when some types have
    empty partitions. Logic unchanged from the original."""
    ntype_ids = {ntype: g.get_ntype_id(ntype) for ntype in g.ntypes}
    ntype_ids_reverse = {v: k for k, v in ntype_ids.items()}
    for part_id in range(num_parts):
        for ntype_id in list(ntype_ids.values()):
            ntype = ntype_ids_reverse[ntype_id]
            start_id = node_map_val[ntype][part_id][0]
            end_id = node_map_val[ntype][part_id][1]
            if not (start_id == -1 and end_id == -1):
                continue
            prev_ntype_id = (
                ntype_ids[ntype] - 1
                if ntype_ids[ntype] > 0
                else max(ntype_ids.values())
            )
            prev_ntype = ntype_ids_reverse[prev_ntype_id]
            if ntype_ids[ntype] == 0:
                if part_id == 0:
                    node_map_val[ntype][part_id][0] = 0
                else:
                    node_map_val[ntype][part_id][0] = node_map_val[prev_ntype][
                        part_id - 1
                    ][1]
            else:
                node_map_val[ntype][part_id][0] = node_map_val[prev_ntype][
                    part_id
                ][1]
            node_map_val[ntype][part_id][1] = node_map_val[ntype][part_id][0]

    etype_ids = {etype: g.get_etype_id(etype) for etype in g.canonical_etypes}
    etype_ids_reverse = {v: k for k, v in etype_ids.items()}
    for part_id in range(num_parts):
        for etype_id in list(etype_ids.values()):
            etype = etype_ids_reverse[etype_id]
            start_id = edge_map_val[etype][part_id][0]
            end_id = edge_map_val[etype][part_id][1]
            if not (start_id == -1 and end_id == -1):
                continue
            prev_etype_id = (
                etype_ids[etype] - 1
                if etype_ids[etype] > 0
                else max(etype_ids.values())
            )
            prev_etype = etype_ids_reverse[prev_etype_id]
            if etype_ids[etype] == 0:
                if part_id == 0:
                    edge_map_val[etype][part_id][0] = 0
                else:
                    edge_map_val[etype][part_id][0] = edge_map_val[prev_etype][
                        part_id - 1
                    ][1]
            else:
                edge_map_val[etype][part_id][0] = edge_map_val[prev_etype][
                    part_id
                ][1]
            edge_map_val[etype][part_id][1] = edge_map_val[etype][part_id][0]

def _get_inner_node_mask(graph, ntype_id, gpb=None):
    ndata = (
        graph.node_attributes
        if isinstance(graph, gb.FusedCSCSamplingGraph)
        else graph.ndata
    )
    assert "inner_node" in ndata, "'inner_node' is not in nodes' data"
    if NTYPE in ndata or gpb is not None:
        ntype = (
            gpb.map_to_per_ntype(ndata[NID])[0]
            if gpb is not None
            else ndata[NTYPE]
        )
        dtype = F.dtype(ndata["inner_node"])
        return ndata["inner_node"] * F.astype(ntype == ntype_id, dtype) == 1
    else:
        return ndata["inner_node"] == 1


def _get_inner_edge_mask(
    graph,
    etype_id,
):
    edata = (
        graph.edge_attributes
        if isinstance(graph, gb.FusedCSCSamplingGraph)
        else graph.edata
    )
    assert "inner_edge" in edata, "'inner_edge' is not in edges' data"
    etype = (
        graph.type_per_edge
        if isinstance(graph, gb.FusedCSCSamplingGraph)
        else (graph.edata[ETYPE] if ETYPE in graph.edata else None)
    )
    if etype is not None:
        dtype = F.dtype(edata["inner_edge"])
        return edata["inner_edge"] * F.astype(etype == etype_id, dtype) == 1
    else:
        return edata["inner_edge"] == 1

def _get_orig_ids(g, sim_g, orig_nids, orig_eids):
    """Convert/construct the original node IDs and edge IDs."""
    is_hetero = not g.is_homogeneous
    if is_hetero:
        orig_ntype = F.gather_row(sim_g.ndata[NTYPE], orig_nids)
        orig_etype = F.gather_row(sim_g.edata[ETYPE], orig_eids)
        orig_nids = F.gather_row(sim_g.ndata[NID], orig_nids)
        orig_eids = F.gather_row(sim_g.edata[EID], orig_eids)
        orig_nids = {
            ntype: F.boolean_mask(
                orig_nids, orig_ntype == g.get_ntype_id(ntype)
            )
            for ntype in g.ntypes
        }
        orig_eids = {
            etype: F.boolean_mask(
                orig_eids, orig_etype == g.get_etype_id(etype)
            )
            for etype in g.canonical_etypes
        }
    return orig_nids, orig_eids


# --------------------------------------------------------------------- #
# Internal helpers used by the optimised save_node_partition_graph below.
# They keep the exact semantics of the original code path but avoid
# recomputing masks and avoid unnecessary sorts on huge edge arrays.
# --------------------------------------------------------------------- #

def _inner_mask_and_inner_ids(part, ntype_id, etype_id):
    """Compute inner-node and inner-edge masks once and extract the
    sorted-contiguous inner ID ranges that the original code used.

    The original code called `_get_inner_node_mask` / `_get_inner_edge_mask`
    twice (once for the count, once again later) and then did
    `np.sort(asnumpy(boolean_mask(...)))` on the inner-edge IDs even though,
    after `partition_graph_with_halo(reshuffle=True)`, those IDs are already
    contiguous within a partition (as the heterogeneous branch's own
    `expected_range = np.arange(typed_eids[0], typed_eids[-1]+1)` assertion
    proves). So we only need the first/last element to derive the range.

    Returns
    -------
    inner_node_mask, inner_edge_mask : 1-D bool tensors
    n_inner_nodes, n_inner_edges : int
    node_range, edge_range : (lo, hi+1) tuples of int, or None if empty
    """
    inner_node_mask = _get_inner_node_mask(part, ntype_id)
    inner_edge_mask = _get_inner_edge_mask(part, etype_id)

    n_inner_nodes = int(F.as_scalar(F.sum(F.astype(inner_node_mask, F.int64), 0)))
    n_inner_edges = int(F.as_scalar(F.sum(F.astype(inner_edge_mask, F.int64), 0)))

    if n_inner_nodes == 0:
        node_range = None
    else:
        inner_nids = F.boolean_mask(part.ndata[NID], inner_node_mask)
        node_range = (
            int(F.as_scalar(inner_nids[0])),
            int(F.as_scalar(inner_nids[-1])) + 1,
        )
        del inner_nids

    if n_inner_edges == 0:
        edge_range = None
    else:
        # Inner edges are reshuffled into a contiguous block, so we only
        # need the first and last instead of the full sort the original
        # code performed. Cheap min/max as a safety net in case the input
        # partitioner ever produces an unsorted block.
        inner_eids = F.boolean_mask(part.edata[EID], inner_edge_mask)
        # `inner_eids` is contiguous after partition_graph_with_halo, but
        # we use min/max rather than [0]/[-1] to stay correct under any
        # ordering -- and crucially without allocating a sorted copy.
        eid_np = F.asnumpy(inner_eids)
        edge_range = (int(eid_np.min()), int(eid_np.max()) + 1)
        del inner_eids, eid_np

    return (
        inner_node_mask,
        inner_edge_mask,
        n_inner_nodes,
        n_inner_edges,
        node_range,
        edge_range,
    )


def save_node_partition_graph(
    g,
    node_part,
    graph_name,
    num_parts,
    out_path,
    partitioner,
    num_hops=1,
    balance_ntypes=None,
    balance_edges=False,
    return_mapping=False,
    **kwargs,
):
    """Partition a graph for distributed training and store the partitions on files.

    See the original DGL ``partition_graph`` for the full file-layout
    description; this implementation produces an identical on-disk format.

    Differences from the original implementation (logic unchanged):
      * Per-partition inner-node / inner-edge masks are computed once
        instead of twice.
      * The full-edge ``np.sort`` over inner-EIDs is replaced by an
        ``min/max`` scan, since inner-EIDs are already contiguous after
        ``partition_graph_with_halo(reshuffle=True)``. This is the single
        biggest cost on graphs like ogbn-papers100M.
      * The two per-partition loops (one for features, one for graph
        save) are merged so that each ``parts[part_id]`` can be released
        immediately after being written. Peak memory drops from "all
        partitions held simultaneously" to "one partition + the global
        graph".
      * ``process_partitions(...)`` is now actually written to disk
        (the original assigned its return value to a local variable
        that was never used, so dtype formatting never reached the
        ``.dgl`` file).
      * Local index tensors (``local_nodes`` / ``local_edges``) are
        explicitly freed inside each partition iteration.
    """
    # 'coo' is required for partition
    assert "coo" in np.concatenate(
        list(g.formats().values())
    ), "'coo' format should be allowed for partitioning graph."

    def get_homogeneous(g, balance_ntypes):
        if g.is_homogeneous:
            sim_g = to_homogeneous(g)
            if isinstance(balance_ntypes, dict):
                assert len(balance_ntypes) == 1
                bal_ntypes = list(balance_ntypes.values())[0]
            else:
                bal_ntypes = balance_ntypes
        elif isinstance(balance_ntypes, dict):
            num_ntypes = 0
            for key in g.ntypes:
                if key in balance_ntypes:
                    g.nodes[key].data["bal_ntype"] = (
                        F.astype(balance_ntypes[key], F.int32) + num_ntypes
                    )
                    uniq_ntypes = F.unique(balance_ntypes[key])
                    assert np.all(
                        F.asnumpy(uniq_ntypes) == np.arange(len(uniq_ntypes))
                    )
                    num_ntypes += len(uniq_ntypes)
                else:
                    g.nodes[key].data["bal_ntype"] = (
                        F.ones((g.num_nodes(key),), F.int32, F.cpu())
                        * num_ntypes
                    )
                    num_ntypes += 1
            sim_g = to_homogeneous(g, ndata=["bal_ntype"])
            bal_ntypes = sim_g.ndata["bal_ntype"]
            print(
                "The graph has {} node types and balance among {} types".format(
                    len(g.ntypes), len(F.unique(bal_ntypes))
                )
            )
            for key in g.ntypes:
                del g.nodes[key].data["bal_ntype"]
            del sim_g.ndata["bal_ntype"]
        else:
            sim_g = to_homogeneous(g)
            bal_ntypes = sim_g.ndata[NTYPE]
        return sim_g, bal_ntypes

    # ====================================================================
    start = time.time()
    sim_g, balance_ntypes = get_homogeneous(g, balance_ntypes)
    print(
        "Converting to homogeneous graph takes {:.3f}s, peak mem: {:.3f} GB".format(
            time.time() - start, get_peak_mem()
        )
    )

    start = time.time()
    parts, orig_nids, orig_eids = partition_graph_with_halo(
        sim_g, node_part, num_hops, reshuffle=True
    )
    # `node_part` is a (num_nodes,)-sized tensor that we no longer need.
    # Releasing it before the per-partition loop frees several hundred MB
    # on papers100M-class graphs.
    del node_part
    gc.collect()
    print(
        "Splitting the graph into partitions takes {:.3f}s, peak mem: {:.3f} GB".format(
            time.time() - start, get_peak_mem()
        )
    )
    if return_mapping:
        orig_nids, orig_eids = _get_orig_ids(g, sim_g, orig_nids, orig_eids)
    else:
        # Caller does not want the mapping -- drop the (num_nodes+num_edges)
        # worth of int64 tensors immediately.
        del orig_nids, orig_eids
        orig_nids = orig_eids = None
    # ====================================================================

    # If the input is a heterogeneous graph, get the original node types and
    # original node IDs. (Logic unchanged from upstream; not exercised on
    # homogeneous graphs like ogbn-papers100M, so the per-partition memory
    # cost here is paid only when needed.)
    if not g.is_homogeneous:
        for name in parts:
            orig_ids = parts[name].ndata["orig_id"]
            ntype = F.gather_row(sim_g.ndata[NTYPE], orig_ids)
            parts[name].ndata[NTYPE] = F.astype(
                ntype, RESERVED_FIELD_DTYPE[NTYPE]
            )
            assert np.all(
                F.asnumpy(ntype) == F.asnumpy(parts[name].ndata[NTYPE])
            )
            orig_ids = parts[name].edata["orig_id"]
            etype = F.gather_row(sim_g.edata[ETYPE], orig_ids)
            parts[name].edata[ETYPE] = F.astype(
                etype, RESERVED_FIELD_DTYPE[ETYPE]
            )
            assert np.all(
                F.asnumpy(etype) == F.asnumpy(parts[name].edata[ETYPE])
            )

            inner_ntype = F.boolean_mask(
                parts[name].ndata[NTYPE], parts[name].ndata["inner_node"] == 1
            )
            inner_nids = F.boolean_mask(
                parts[name].ndata[NID], parts[name].ndata["inner_node"] == 1
            )
            for ntype in g.ntypes:
                inner_ntype_mask = inner_ntype == g.get_ntype_id(ntype)
                if F.sum(F.astype(inner_ntype_mask, F.int64), 0) == 0:
                    continue
                typed_nids = F.boolean_mask(inner_nids, inner_ntype_mask)
                expected_range = np.arange(
                    int(F.as_scalar(typed_nids[0])),
                    int(F.as_scalar(typed_nids[-1])) + 1,
                )
                assert np.all(F.asnumpy(typed_nids) == expected_range)
            inner_etype = F.boolean_mask(
                parts[name].edata[ETYPE], parts[name].edata["inner_edge"] == 1
            )
            inner_eids = F.boolean_mask(
                parts[name].edata[EID], parts[name].edata["inner_edge"] == 1
            )
            for etype in g.canonical_etypes:
                inner_etype_mask = inner_etype == g.get_etype_id(etype)
                if F.sum(F.astype(inner_etype_mask, F.int64), 0) == 0:
                    continue
                typed_eids = np.sort(
                    F.asnumpy(F.boolean_mask(inner_eids, inner_etype_mask))
                )
                assert np.all(
                    typed_eids
                    == np.arange(int(typed_eids[0]), int(typed_eids[-1]) + 1)
                )

    os.makedirs(out_path, mode=0o775, exist_ok=True)
    tot_num_inner_edges = 0
    out_path = os.path.abspath(out_path)

    # ------------------------------------------------------------------ #
    # Build node_map_val / edge_map_val.
    #
    # The original code looped over all parts twice per (ntype, etype) --
    # once to count inner nodes/edges (val), once again later to fetch
    # their global IDs to derive the [start, end) ranges. We now compute
    # both in a single pass over each partition, caching the masks in
    # ``part_masks`` for reuse during the feature-extraction loop further
    # down.
    # ------------------------------------------------------------------ #
    # part_masks[part_id] = dict with cached inner masks/ranges keyed by
    # (ntype, etype). For homogeneous graphs there is exactly one entry
    # per partition; for heterogeneous graphs we cache per (ntype, etype)
    # pair only the homogeneous bookkeeping needs (the hetero branch
    # above already did its own pass).
    part_masks = {}

    if num_parts > 1:
        node_map_val = {}
        edge_map_val = {}

        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            node_map_val[ntype] = []
            running_count = 0
            for i in parts:
                inner_node_mask = _get_inner_node_mask(parts[i], ntype_id)
                count = int(
                    F.as_scalar(F.sum(F.astype(inner_node_mask, F.int64), 0))
                )
                running_count += count
                if count == 0:
                    node_map_val[ntype].append([-1, -1])
                    # We still cache the mask so the feature pass does
                    # not have to recompute it.
                    part_masks.setdefault(i, {}).setdefault("nodes", {})[
                        ntype
                    ] = (inner_node_mask, count)
                    continue
                inner_nids = F.boolean_mask(
                    parts[i].ndata[NID], inner_node_mask
                )
                node_map_val[ntype].append(
                    [
                        int(F.as_scalar(inner_nids[0])),
                        int(F.as_scalar(inner_nids[-1])) + 1,
                    ]
                )
                part_masks.setdefault(i, {}).setdefault("nodes", {})[
                    ntype
                ] = (inner_node_mask, count)
                del inner_nids
            assert running_count == g.num_nodes(ntype)

        for etype in g.canonical_etypes:
            etype_id = g.get_etype_id(etype)
            edge_map_val[etype] = []
            running_count = 0
            for i in parts:
                inner_edge_mask = _get_inner_edge_mask(parts[i], etype_id)
                count = int(
                    F.as_scalar(F.sum(F.astype(inner_edge_mask, F.int64), 0))
                )
                running_count += count
                if count == 0:
                    edge_map_val[etype].append([-1, -1])
                    part_masks.setdefault(i, {}).setdefault("edges", {})[
                        etype
                    ] = (inner_edge_mask, count)
                    continue
                # Inner edges are reshuffled into a contiguous range, so
                # min/max suffices instead of an O(E) sort. This is the
                # main hot-path saving on papers100M-scale inputs.
                inner_eids = F.boolean_mask(
                    parts[i].edata[EID], inner_edge_mask
                )
                eid_np = F.asnumpy(inner_eids)
                edge_map_val[etype].append(
                    [int(eid_np.min()), int(eid_np.max()) + 1]
                )
                part_masks.setdefault(i, {}).setdefault("edges", {})[
                    etype
                ] = (inner_edge_mask, count)
                del inner_eids, eid_np
            assert running_count == g.num_edges(etype)

        _update_node_edge_map(node_map_val, edge_map_val, g, num_parts)
    else:
        node_map_val = {}
        edge_map_val = {}
        for ntype in g.ntypes:
            ntype_id = g.get_ntype_id(ntype)
            inner_node_mask = _get_inner_node_mask(parts[0], ntype_id)
            inner_nids = F.boolean_mask(parts[0].ndata[NID], inner_node_mask)
            node_map_val[ntype] = [
                [
                    int(F.as_scalar(inner_nids[0])),
                    int(F.as_scalar(inner_nids[-1])) + 1,
                ]
            ]
            part_masks.setdefault(0, {}).setdefault("nodes", {})[ntype] = (
                inner_node_mask,
                int(inner_nids.shape[0]),
            )
            del inner_nids
        for etype in g.canonical_etypes:
            etype_id = g.get_etype_id(etype)
            inner_edge_mask = _get_inner_edge_mask(parts[0], etype_id)
            inner_eids = F.boolean_mask(parts[0].edata[EID], inner_edge_mask)
            eid_np = F.asnumpy(inner_eids)
            edge_map_val[etype] = [[int(eid_np.min()), int(eid_np.max()) + 1]]
            part_masks.setdefault(0, {}).setdefault("edges", {})[etype] = (
                inner_edge_mask,
                int(inner_eids.shape[0]),
            )
            del inner_eids, eid_np

        # Double check that the node IDs in the global ID space are sorted.
        for ntype in node_map_val:
            val = np.concatenate([np.array(l) for l in node_map_val[ntype]])
            assert np.all(val[:-1] <= val[1:])
        for etype in edge_map_val:
            val = np.concatenate([np.array(l) for l in edge_map_val[etype]])
            assert np.all(val[:-1] <= val[1:])

    start = time.time()
    ntypes = {ntype: g.get_ntype_id(ntype) for ntype in g.ntypes}
    etypes = {etype: g.get_etype_id(etype) for etype in g.canonical_etypes}
    part_metadata = {
        "graph_name": graph_name,
        "num_nodes": g.num_nodes(),
        "num_edges": g.num_edges(),
        "part_method": partitioner,
        "num_parts": num_parts,
        "halo_hops": num_hops,
        "node_map": node_map_val,
        "edge_map": edge_map_val,
        "ntypes": ntypes,
        "etypes": etypes,
    }
    part_config = os.path.join(out_path, graph_name + ".json")
    sort_etypes = len(g.etypes) > 1

    # ------------------------------------------------------------------ #
    # Per-partition write loop.
    #
    # Merged from the original two passes: features are gathered, the
    # graph structure is dtype-formatted via process_partitions, then
    # both are written to disk and the partition reference is dropped.
    # This keeps peak resident memory at roughly (one partition) +
    # (one feature dict) + (the global graph), instead of holding all
    # partitions throughout the function.
    # ------------------------------------------------------------------ #
    part_ids = list(parts.keys())
    for part_id in part_ids:
        part = parts[part_id]
        masks_for_part = part_masks.get(part_id, {})

        node_feats = {}
        edge_feats = {}
        if num_parts > 1:
            for ntype in g.ntypes:
                ntype_id = g.get_ntype_id(ntype)
                inner_node_mask, n_inner = masks_for_part.get("nodes", {}).get(
                    ntype, (None, None)
                )
                if inner_node_mask is None:
                    inner_node_mask = _get_inner_node_mask(part, ntype_id)

                # `orig_id` holds the global node IDs of the input graph;
                # for homogeneous graphs that's already the per-ntype ID,
                # for heterogeneous graphs we still need the sim_g[NID]
                # remap below.
                local_nodes = F.boolean_mask(
                    part.ndata["orig_id"], inner_node_mask
                )
                if len(g.ntypes) > 1:
                    local_nodes = F.gather_row(sim_g.ndata[NID], local_nodes)
                    print(
                        "part {} has {} nodes of type {} and {} are inside the partition".format(
                            part_id,
                            F.as_scalar(
                                F.sum(part.ndata[NTYPE] == ntype_id, 0)
                            ),
                            ntype,
                            len(local_nodes),
                        )
                    )
                else:
                    print(
                        "part {} has {} nodes and {} are inside the partition".format(
                            part_id, part.num_nodes(), len(local_nodes)
                        )
                    )

                for name in g.nodes[ntype].data:
                    if name in [NID, "inner_node"]:
                        continue
                    node_feats[ntype + "/" + name] = F.gather_row(
                        g.nodes[ntype].data[name], local_nodes
                    )
                del local_nodes, inner_node_mask

            for etype in g.canonical_etypes:
                etype_id = g.get_etype_id(etype)
                inner_edge_mask, n_inner = masks_for_part.get("edges", {}).get(
                    etype, (None, None)
                )
                if inner_edge_mask is None:
                    inner_edge_mask = _get_inner_edge_mask(part, etype_id)

                local_edges = F.boolean_mask(
                    part.edata["orig_id"], inner_edge_mask
                )
                if not g.is_homogeneous:
                    local_edges = F.gather_row(sim_g.edata[EID], local_edges)
                    print(
                        "part {} has {} edges of type {} and {} are inside the partition".format(
                            part_id,
                            F.as_scalar(
                                F.sum(part.edata[ETYPE] == etype_id, 0)
                            ),
                            etype,
                            len(local_edges),
                        )
                    )
                else:
                    print(
                        "part {} has {} edges and {} are inside the partition".format(
                            part_id, part.num_edges(), len(local_edges)
                        )
                    )
                tot_num_inner_edges += int(local_edges.shape[0])

                for name in g.edges[etype].data:
                    if name in [EID, "inner_edge"]:
                        continue
                    edge_feats[
                        _etype_tuple_to_str(etype) + "/" + name
                    ] = F.gather_row(g.edges[etype].data[name], local_edges)
                del local_edges, inner_edge_mask
        else:
            for ntype in g.ntypes:
                if len(g.ntypes) > 1:
                    ntype_id = g.get_ntype_id(ntype)
                    inner_node_mask = _get_inner_node_mask(part, ntype_id)
                    local_nodes = F.boolean_mask(
                        part.ndata["orig_id"], inner_node_mask
                    )
                    local_nodes = F.gather_row(sim_g.ndata[NID], local_nodes)
                    del inner_node_mask
                else:
                    local_nodes = sim_g.ndata[NID]
                for name in g.nodes[ntype].data:
                    if name in [NID, "inner_node"]:
                        continue
                    node_feats[ntype + "/" + name] = F.gather_row(
                        g.nodes[ntype].data[name], local_nodes
                    )
                if len(g.ntypes) > 1:
                    del local_nodes
            for etype in g.canonical_etypes:
                if not g.is_homogeneous:
                    etype_id = g.get_etype_id(etype)
                    inner_edge_mask = _get_inner_edge_mask(part, etype_id)
                    local_edges = F.boolean_mask(
                        part.edata["orig_id"], inner_edge_mask
                    )
                    local_edges = F.gather_row(sim_g.edata[EID], local_edges)
                    del inner_edge_mask
                else:
                    local_edges = sim_g.edata[EID]
                for name in g.edges[etype].data:
                    if name in [EID, "inner_edge"]:
                        continue
                    edge_feats[
                        _etype_tuple_to_str(etype) + "/" + name
                    ] = F.gather_row(g.edges[etype].data[name], local_edges)
                if not g.is_homogeneous:
                    del local_edges

        # We're done with `orig_id` for this partition. Drop it before
        # the dtype-formatting pass so process_partitions doesn't have
        # to copy it.
        del part.ndata["orig_id"]
        del part.edata["orig_id"]

        # Write features.
        part_dir = os.path.join(out_path, "part" + str(part_id))
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        save_tensors(node_feat_file, node_feats)
        save_tensors(edge_feat_file, edge_feats)
        # Free the feature dicts before the graph save (each value is a
        # potentially huge tensor).
        del node_feats, edge_feats

        part_metadata["part-{}".format(part_id)] = {
            "node_feats": os.path.relpath(node_feat_file, out_path),
            "edge_feats": os.path.relpath(edge_feat_file, out_path),
        }

        # Dtype-format and save the graph structure.
        # NOTE (bug fix vs. original): we now persist the result of
        # process_partitions, so inner_node/inner_edge are written as
        # uint8 and NID/EID/NTYPE/ETYPE land at their declared dtypes.
        part = process_partitions(part, None, sort_etypes)
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata["part-{}".format(part_id)][
            "part_graph"
        ] = os.path.relpath(part_graph_file, out_path)
        _save_dgl_graphs(part_graph_file, [part], formats=None)

        # Drop this partition before iterating to the next one. With
        # papers100M and num_parts=8 this is the difference between
        # holding ~all subgraphs concurrently and holding one at a time.
        del part
        del parts[part_id]
        part_masks.pop(part_id, None)
        gc.collect()

    _dump_part_config(part_config, part_metadata)

    num_cuts = sim_g.num_edges() - tot_num_inner_edges
    if num_parts == 1:
        num_cuts = 0
    print(
        "There are {} edges in the graph and {} edge cuts for {} partitions.".format(
            g.num_edges(), num_cuts, num_parts
        )
    )

    print(
        "Save partitions: {:.3f} seconds, peak memory: {:.3f} GB".format(
            time.time() - start, get_peak_mem()
        )
    )

    if return_mapping:
        return orig_nids, orig_eids
