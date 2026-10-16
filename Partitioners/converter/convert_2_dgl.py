#================================================================================
# Libraries
#================================================================================
import argparse
import re
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collections import defaultdict

import dgl
from dgl.data.utils import load_graphs

from tqdm import tqdm

from loader_functions import *
from node_partition_dgl_format_saver import *
from edge_partition_dgl_format_saver import *


#================================================================================
# Entry Point
#================================================================================
def main(args):
    print("--- Loading Global Graph ---")
    g_list, _ = load_graphs(args.graph_path)
    graph = g_list[0]

    input_path = Path(args.input_path)
    output_path = Path(args.output_dir)

    if args.partition_type == "node":
        if args.partition_input_format == "node_map_flat":
            node_parts = load_node_map_flat(input_path)
        elif args.partition_input_format == "sdtgnn_format":
            node_parts = load_sdtgnn_node_partition(
                input_path,
                num_nodes=graph.num_nodes()
            )
        else:
            raise Exception(f"Unknown partition_input_format for node partition: {args.partition_input_format}")

        if graph.num_nodes() != len(node_parts):
            raise Exception(f"Node Parts length({len(node_parts)}) is not equal Graph Node length({graph.num_nodes()}).")

        save_node_partition_graph(
            g=graph,
            node_part=node_parts,
            graph_name=args.graph_name,
            num_parts=int(node_parts.max()) + 1,
            out_path=output_path.as_posix(),
            partitioner=args.partitioner,
            num_hops=1,
            balance_ntypes=None,
            blance_edges=False,
            return_mapping=False
        )


    elif args.partition_type == "edge":
        if args.partition_input_format == "edge_list_paired":
            edge_parts = load_edge_list_paired(graph, input_path)
        elif args.partition_input_format == "edge_map_flat":
            edge_parts = load_edge_map_flat(graph, input_path)
        elif args.partition_input_format == "edge_list_labeled":
            edge_parts = load_edge_list_labeled(graph, input_path)
        elif args.partition_input_format == "dgl_style_multi":
            edge_parts = load_dgl_style_multi(graph, input_path)
        else:
            raise Exception(f"Unknown partition_input_format for edge partition: {args.partition_input_format}")

        if graph.num_edges() != len(edge_parts):
            raise Exception(
                f"Edge Parts length({len(edge_parts)}) is not equal "
                f"Graph Edge length({graph.num_edges()})."
            )

        save_edge_partition_graph(
            g=graph,
            edge_part=edge_parts,
            graph_name=args.graph_name,
            num_parts=int(edge_parts.max()) + 1,
            out_path=output_path.as_posix(),
            partitioner=args.partitioner,
            master_rule="degree",       # or "first" / "hash" / "degree"
            return_mapping=False,
            write_drpa_meta=True
        )
    else:
        raise Exception(f"Unknown partition type: {args.partition_type}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert graph partitioner output into DGL-compatible partition format. "
            "Edge-cut output matches dgl.distributed.partition_graph: subgraph "
            "contains edges with at least one inner endpoint (either-inner), "
            "inner_edge uses DGL's dst-owned convention. Vertex-cut output uses "
            "the same file layout with an extra _partition_type marker."
        )
    )

    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--graph_name", type=str, required=True)
    parser.add_argument("--partition_type", type=str, required=True,
                        choices=["edge", "node"])
    parser.add_argument("--partition_input_format", type=str, required=True,
                        choices=["dgl_style_multi", "node_map_flat", "sdtgnn_format", "edge_map_flat",
                                 "edge_list_labeled", "edge_list_paired"])
    parser.add_argument("--partitioner", type=str, required=True,
                        help="Name of the partitioning algorithm (stored in metadata).")

    args = parser.parse_args()
    main(args)
