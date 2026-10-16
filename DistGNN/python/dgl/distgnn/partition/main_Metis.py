#!/usr/bin/env python3

"""
Script for downloading various datasets, partitioning them with METIS
and saving them in DGL format.

Requirements:
- dgl
- torch
- numpy
- metis (Python wrapper for METIS)
- ogb (for OGB datasets)

Install dependencies:
pip install dgl torch numpy metis ogb
"""

import os
import sys
import dgl
import torch as th
import numpy as np
import json
from pathlib import Path
import dgl
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors
from scipy.io import mmread
import requests
import random


def download_proteins():
    """Download the Proteins dataset."""
    print("Downloading dataset...")
    print("This might a take while..")
    url = "https://portal.nersc.gov/project/m1982/GNN/subgraph3_iso_vs_iso_30_70length_ALL.m100.propermm.mtx"
    r = requests.get(url)
    with open("proteins.mtx", "wb") as handle:
        handle.write(r.content)


def proteins_mtx2dgl():
    """Convert Proteins MTX format to DGL."""
    print("Converting mtx2dgl..")
    print("This might a take while..")
    a = mmread("proteins.mtx")
    coo = a.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)

    g = dgl.graph((u, v))

    n = g.number_of_nodes()
    feat_size = 128
    feats = th.randn([n, feat_size], dtype=th.float32)

    train_size = min(1000000, n)
    test_size = min(500000, n - train_size)
    val_size = min(5000, n - train_size - test_size)
    nlabels = 256

    train_mask = th.zeros(n, dtype=th.bool)
    test_mask = th.zeros(n, dtype=th.bool)
    val_mask = th.zeros(n, dtype=th.bool)
    label = th.zeros(n, dtype=th.int64)

    for i in range(train_size):
        train_mask[i] = True

    for i in range(test_size):
        test_mask[train_size + i] = True

    for i in range(val_size):
        val_mask[train_size + test_size + i] = True

    for i in range(n):
        label[i] = random.choice(range(nlabels))

    g.ndata["feat"] = feats
    g.ndata["train_mask"] = train_mask
    g.ndata["test_mask"] = test_mask
    g.ndata["val_mask"] = val_mask
    g.ndata["label"] = label

    print(g)
    return g


def save_dataset(g, dataset):
    """Save the dataset."""
    print("Saving dataset..")
    part_dir = os.path.join("./" + dataset)
    node_feat_file = os.path.join(part_dir, "node_feat.dgl")
    part_graph_file = os.path.join(part_dir, "graph.dgl")
    os.makedirs(part_dir, mode=0o775, exist_ok=True)
    save_tensors(node_feat_file, g.ndata)
    save_graphs(part_graph_file, [g])


def load_proteins(dataset):
    """Load the Proteins dataset."""
    part_dir = dataset
    graph_file = os.path.join(part_dir + "/graph.dgl")

    if not os.path.exists("proteins.mtx"):
        download_proteins()
    if not os.path.exists(graph_file):
        g = proteins_mtx2dgl()
        save_dataset(g, dataset)

    graph = load_graphs(graph_file)[0][0]
    return graph


def load_ogb(dataset_name):
    """Load OGB datasets."""
    try:
        from ogb.nodeproppred import DglNodePropPredDataset

        dataset = DglNodePropPredDataset(name=dataset_name, root="dataset/")
        split_idx = dataset.get_idx_split()
        graph, labels = dataset[0]

        # Set masks
        train_mask = th.zeros(graph.number_of_nodes(), dtype=th.bool)
        val_mask = th.zeros(graph.number_of_nodes(), dtype=th.bool)
        test_mask = th.zeros(graph.number_of_nodes(), dtype=th.bool)

        train_mask[split_idx["train"]] = True
        val_mask[split_idx["valid"]] = True
        test_mask[split_idx["test"]] = True

        graph.ndata["train_mask"] = train_mask
        graph.ndata["val_mask"] = val_mask
        graph.ndata["test_mask"] = test_mask
        graph.ndata["label"] = labels.squeeze()

        return graph, labels
    except ImportError:
        print("OGB is not installed. Install it with: pip install ogb")
        sys.exit(1)


class args_:
    """Argument class for compatibility."""

    def __init__(self, dataset):
        self.dataset = dataset


def metis_partition(num_community, dataset, output_dir):
    """
    Perform METIS partitioning (equivalent to vertexCut_v2 in Libra).

    Args:
        num_community (int): Number of partitions
        dataset (str): Dataset name
        output_dir (str): Output directory

    Returns:
        int: Maximum partition size
    """
    args = args_(dataset)
    print(f"Input dataset: {args.dataset}")

    # Load the dataset specified by its name
    if args.dataset == "ogbn-products":
        print("Loading ogbn-products")
        G, _ = load_ogb("ogbn-products")
    elif args.dataset == "ogbn-papers100M":
        print("Loading ogbn-papers100M")
        G, _ = load_ogb("ogbn-papers100M")
    elif args.dataset == "proteins":
        G = load_proteins("proteins")
    elif args.dataset == "ogbn-arxiv":
        print("Loading ogbn-arxiv")
        G, _ = load_ogb("ogbn-arxiv")
    else:
        try:
            # For standard DGL datasets
            if args.dataset == "cora":
                from dgl.data import CoraGraphDataset

                dataset_obj = CoraGraphDataset()
                G = dataset_obj[0]
            elif args.dataset == "pubmed":
                from dgl.data import PubmedGraphDataset

                dataset_obj = PubmedGraphDataset()
                G = dataset_obj[0]
            elif args.dataset == "citeseer":
                from dgl.data import CiteseerGraphDataset

                dataset_obj = CiteseerGraphDataset()
                G = dataset_obj[0]
            elif args.dataset == "reddit":
                from dgl.data import RedditDataset

                dataset_obj = RedditDataset()
                G = dataset_obj[0]
            else:
                raise ValueError(f"Dataset {dataset} not supported")
        except Exception as e:
            print(f"Dataset {dataset} not found: {e}")
            sys.exit(1)

    print("Done loading", flush=True)

    N_n = G.number_of_nodes()
    N_e = G.number_of_edges()
    print(f"Graph: {N_n} nodes, {N_e} edges")

    print(f"Partitioning graph into {num_community} parts with METIS...")

    try:
        # METIS partitioning
        print(G)
        resultDictGraph = dgl.metis_partition(G, num_community)

        # edge_cuts, partition_assignment = metis.part_graph(
        #    adjacency_list, nparts=num_community, recursive=True
        # )

        print(f"Partitionierung abgeschlossen:")
        # print(f"  - Edge cuts: {edge_cuts}")

        # Create partitioned graphs
        partition_graphs = create_partitioned_graphs(G, resultDictGraph, num_community)

        # Save partitions
        save_metis_partitions(
            partition_graphs, resultDictGraph, output_dir, num_community, args.dataset
        )

        # Compute the maximum partition size
        partition_sizes = [g.number_of_nodes() for g in partition_graphs.values()]
        max_partition_size = max(partition_sizes)

        print(f"Partitionsgrssen: {partition_sizes}")
        print(f"Maximale Partitionsgrsse: {max_partition_size}")

        return max_partition_size

    except Exception as e:
        print(f"Error during METIS partitioning: {e}")
        raise


def create_partitioned_graphs(original_graph, partition_assignment, num_partitions):
    """
    Create separate DGL graphs for each partition.

    Args:
        original_graph: Original DGL graph
        partition_assignment: METIS partition assignment
        num_partitions: Number of partitions

    Returns:
        dict: Dictionary with partition IDs as keys and DGL graphs as values
    """
    print("Creating partitioned graphs...")

    partition_graphs = {}

    for part_id in range(num_partitions):
        # Find nodes in this partition
        nodes_in_partition = partition_assignment[part_id].nodes()

        if len(nodes_in_partition) == 0:
            print(f"Warning: partition {part_id} is empty")
            continue

        # Create subgraph
        subgraph = dgl.node_subgraph(original_graph, nodes_in_partition)

        # Copy all node data
        for key, value in original_graph.ndata.items():
            subgraph.ndata[key] = value[nodes_in_partition]

        # Add original node IDs
        subgraph.ndata["original_id"] = nodes_in_partition

        partition_graphs[part_id] = subgraph

        print(
            f"Partition {part_id}: {subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges"
        )

    return partition_graphs


def save_metis_partitions(
    partition_graphs, partition_assignment, output_dir, num_partitions, dataset
):
    """
    Save the METIS partitions (equivalent to libra2dgl.run).

    Args:
        partition_graphs: Dictionary of partitioned graphs
        partition_assignment: METIS partition assignment
        output_dir: Output directory
        num_partitions: Number of partitions
    """
    print("Saving METIS partitions...")

    print("Create Folder: ", output_dir)
    # Create output directory
    os.makedirs(output_dir, mode=0o775, exist_ok=True)

    dgl_paths = {}
    node_counter = []
    edge_counter = []
    halo_nodes_counter = []

    # Save each partitioned graph
    for part_id, graph in partition_graphs.items():
        tmpGraph = graph.ndata["feat"].type(th.int32)
        halo_nodes_counter.append(
            (~tmpGraph).nonzero(as_tuple=True)[0].unique().size()[0]
        )
        # halo_nodes_counter.append()

        # Save graph
        folder_path = output_dir + f"/part{part_id}"
        if not os.path.exists(folder_path):
            print("Create Folder: ", folder_path)
            os.makedirs(folder_path, mode=0o775, exist_ok=True)

        graph_file = os.path.join(folder_path, "graph.dgl")
        save_graphs(graph_file, [graph])

        # Save node features separately (as in Libra)
        node_feat_file = os.path.join(folder_path, "node_feat.dgl")
        save_tensors(node_feat_file, graph.ndata)

        edge_feat_file = os.path.join(folder_path, "edge_feat.dgl")
        save_tensors(edge_feat_file, graph.edata)

        dgl_paths[f"part-{part_id}"] = {
            "edge_feats": edge_feat_file,
            "node_feats": node_feat_file,
            "part_graph": graph_file,
        }

        node_counter.append(graph.number_of_nodes())
        edge_counter.append(graph.number_of_edges())

        print(f"Partition {part_id} saved: {graph_file}")

    # Save partition assignment
    # partition_file = os.path.join(output_dir, "partition_assignment.json")
    # with open(partition_file, "w", encoding="utf-8") as f:
    # json.dump(partition_assignment, f, indent=4,ensure_ascii=False)

    # Save metadata
    metadata = {
        "edge_map": edge_counter,
        "graph_name": dataset,
        "halo_hops": halo_nodes_counter,
        "node_map": node_counter,
        "num_edges": sum(g.number_of_edges() for g in partition_graphs.values()),
        "num_nodes": sum(g.number_of_nodes() for g in partition_graphs.values()),
        "num_parts": len(partition_graphs),
    }

    metadata.update(dgl_paths)
    metadata["part_method"] = "METIS"

    metadata_file = os.path.join(output_dir, dataset + ".json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"Metadata saved in {output_dir}")


def load_metis_partitions(input_dir):
    """
    Load the saved METIS partitions.

    Args:
        input_dir: Input directory

    Returns:
        tuple: (partition_graphs, partition_assignment, metadata)
    """
    print(f"Loading METIS partitions from '{input_dir}'...")

    # Load metadata
    metadata_file = os.path.join(input_dir, "metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Load partition assignment
    partition_file = os.path.join(input_dir, "partition_assignment.json")
    with open(partition_file, "r") as f:
        partition_assignment = json.load(f)

    # Load all partition graphs
    partition_graphs = {}
    for part_id in range(metadata["num_partitions"]):
        graph_file = os.path.join(input_dir, f"part_{part_id}_graph.dgl")
        if os.path.exists(graph_file):
            graphs, _ = dgl.load_graphs(graph_file)
            partition_graphs[part_id] = graphs[0]

    print(f"Successfully loaded {len(partition_graphs)} partitions")

    return partition_graphs, partition_assignment, metadata


def main():
    """Main function of the script."""
    if len(sys.argv) != 2:
        print("Error: Input_dataset required !!")
        print("Usage: python script.py <dataset_name>")
        print(
            "Supported datasets: cora, pubmed, citeseer, reddit, ogbn-products, ogbn-arxiv, proteins, ogbn-papers100M"
        )
        sys.exit(1)

    dataset = sys.argv[1]
    print(f"Dataset: {dataset}")

    # Determine output directory
    prefix = ""
    if dataset == "cora":
        resultdir = os.path.join(prefix, "METIS_result_cora")
        index = 0
    elif dataset == "pubmed":
        resultdir = os.path.join(prefix, "METIS_result_pubmed")
        index = 0
    elif dataset == "citeseer":
        resultdir = os.path.join(prefix, "METIS_result_citeseer")
        index = 0
    elif dataset == "reddit":
        resultdir = os.path.join(prefix, "METIS_result_reddit")
        index = 0
    elif dataset == "ogbn-products":
        resultdir = os.path.join(prefix, "METIS_result_ogbn-products")
        index = 1
    elif dataset == "ogbn-papers100M":
        resultdir = os.path.join(prefix, "METIS_result_ogbn-papers100M")
        index = 3
    elif dataset == "proteins":
        resultdir = os.path.join(prefix, "METIS_result_proteins")
        index = 2
    elif dataset == "ogbn-arxiv":
        resultdir = os.path.join(prefix, "METIS_result_ogbn-arxiv")
        index = 0
    else:
        print(f"Error: Dataset {dataset} not supported!")
        sys.exit(1)

    # Create output directory
    try:
        os.makedirs(resultdir, mode=0o775, exist_ok=True)
    except:
        print(f"Error: Could not create directory: {resultdir}")
        sys.exit(1)

    # Partition ranges (as in Libra)
    partition_ranges = [
        [2, 4, 8, 16],  # index 0: small datasets
        [2, 4, 8, 16, 32, 64],  # index 1: medium-sized datasets
        [2, 4, 8, 16, 32, 64],  # index 2: proteins
        [32, 64, 128],  # index 3: very large datasets
    ]

    print(f"Output is stored in {resultdir}", flush=True)
    print(f"Partition range: {partition_ranges[index]}")

    # Run partitioning for different partition counts
    for num_community in partition_ranges[index]:
        print(f"Executing partitions: {num_community}")

        try:
            resultdir_ = os.path.join(resultdir, f"{num_community}Communities")
            os.makedirs(resultdir_, mode=0o775, exist_ok=True)
        except:
            print(f"Error: Could not create sub-directory: {resultdir_}")
            continue

        # Run METIS partitioning
        max_partition_size = metis_partition(num_community, dataset, resultdir_)

        print(
            f"\nThe max partition size for {num_community} communities: {max_partition_size}"
        )
        print("METIS partitioning completed !!!")
        print()

    print("Partitioning completed successfully !!!")


if __name__ == "__main__":
    main()
