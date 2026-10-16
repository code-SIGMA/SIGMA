import argparse
import json
from pathlib import Path


def normalize_keys(graph, labels):
    # feat -> features
    if "feat" in graph.ndata and "features" not in graph.ndata:
        graph.ndata["features"] = graph.ndata.pop("feat")

    if labels.dim() > 1 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    graph.ndata["labels"] = labels

    return graph, labels


def add_masks_from_split(graph, split_idx):
    import torch as th

    num_nodes = graph.num_nodes()
    train_mask = th.zeros(num_nodes, dtype=th.bool)
    val_mask = th.zeros(num_nodes, dtype=th.bool)
    test_mask = th.zeros(num_nodes, dtype=th.bool)

    train_mask[split_idx["train"]] = True
    val_mask[split_idx["valid"]] = True
    test_mask[split_idx["test"]] = True

    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    return graph


def main(args):
    print("Import dgl...")
    import dgl

    print("Import ogb...")
    from ogb.nodeproppred import DglNodePropPredDataset

    print("Import torch...")
    import torch as th

    print("Load Dataset...")
    dataset = DglNodePropPredDataset(name=args.dataset, root=args.raw_dir)

    print("Load Graph...")
    graph, labels = dataset[0]
    graph = graph.cpu()

    if "feat" not in graph.ndata and "features" not in graph.ndata:
        raise ValueError(
            "Node features are missing: neither 'feat' nor 'features' in graph.ndata"
        )

    print("Normalize ndata keys (feat->features) and attach labels...")
    graph, labels = normalize_keys(graph, labels)

    print("Create train/val/test masks from OGB split...")
    split_idx = dataset.get_idx_split()
    graph = add_masks_from_split(graph, split_idx)

    required = ["features", "labels", "train_mask", "val_mask", "test_mask"]
    missing = [k for k in required if k not in graph.ndata]
    if missing:
        raise RuntimeError(f"Still missing after normalization: {missing}")

    print("Make Graph Bidirected and remove duplicates...")
    graph = dgl.to_simple(
        dgl.to_bidirected(graph, copy_ndata=True),
        copy_ndata=True,
        copy_edata=True,
    )

    print("Remove Self-Loops...")
    graph = dgl.remove_self_loop(graph)

    print("Save Graph...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dgl_graph_save_path = save_dir / f"{args.dataset}.dgl"
    dgl_info_save_path = save_dir / f"{args.dataset}_info.json"

    dgl.save_graphs(dgl_graph_save_path.as_posix(), [graph], {"labels": labels})

    print("Save Graph Information...")
    tmp_meta = dataset.meta_info.reset_index()
    tmp_meta.loc[len(tmp_meta)] = ["node_count", graph.num_nodes()]
    tmp_meta.loc[len(tmp_meta)] = ["edge_count", graph.num_edges()]
    tmp_meta.loc[len(tmp_meta)] = ["feature_dim", int(graph.ndata["features"].shape[1])]
    tmp_meta.loc[len(tmp_meta)] = ["train_count", int(graph.ndata["train_mask"].sum())]
    tmp_meta.loc[len(tmp_meta)] = ["val_count", int(graph.ndata["val_mask"].sum())]
    tmp_meta.loc[len(tmp_meta)] = ["test_count", int(graph.ndata["test_mask"].sum())]

    json_data = json.loads(tmp_meta.set_index("index").to_json())
    json_str = json.dumps(json_data[args.dataset], indent=4)
    with open(dgl_info_save_path.as_posix(), "w") as f:
        f.write(json_str)

    print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        choices=["ogbn-products", "ogbn-arxiv", "ogbn-papers100M"],
        help="Dataset to download from ogb.",
        required=True,
    )
    parser.add_argument(
        "-r", "--raw-dir",
        type=str,
        help="Path to save the raw data from ogb.",
        default="/root/.ogb/",
        required=False,
    )
    parser.add_argument(
        "-s", "--save-dir",
        type=str,
        help="Path to save the graph in DGL format and information in json format.",
        required=True,
    )
    args = parser.parse_args()
    main(args)
