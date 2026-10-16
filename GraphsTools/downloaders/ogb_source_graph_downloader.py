import argparse
import json
from pathlib import Path

# Shared prep module: canonical feature names (feat/label/*_mask),
# float64 -> float32 downcast, (N,1) label squeeze, int64 labels and
# CSC materialisation -- the work the training script used to redo at
# every start-up.
from graph_prep import prepare_graph_for_training, to_csc


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

    print("Attach labels...")
    graph.ndata["label"] = labels

    print("Create train/val/test masks from OGB split...")
    split_idx = dataset.get_idx_split()
    graph = add_masks_from_split(graph, split_idx)

    # Canonical names (feat / label / *_mask -- what the trainer reads
    # without its alias fallback), (N,1) label squeeze, int64 labels
    # (same cast the trainer applied at every start-up), optional fp32
    # downcast. CSC is materialised later, after the structural changes.
    print("Canonicalize ndata keys and dtypes (graph_prep)...")
    graph = prepare_graph_for_training(
        graph, fp32=not args.no_f32_features, csc=False, log=print
    )
    labels = graph.ndata["label"]

    required = ["feat", "label", "train_mask", "val_mask", "test_mask"]
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

    # Materialise CSC *keeping COO*: the vertex-cut converter needs COO
    # for partitioning; CSC saves the trainer's full-graph eval from
    # rebuilding it at every start-up (a cheap fallback stays there).
    print("Materialize CSC (keeping COO for the converter)...")
    graph = to_csc(graph, restrict=False, log=print)

    print("Save Graph...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dgl_graph_save_path = save_dir / f"{args.dataset}.dgl"
    dgl_info_save_path = save_dir / f"{args.dataset}_info.json"

    dgl.save_graphs(dgl_graph_save_path.as_posix(), [graph],
                    {"labels": graph.ndata["label"]})

    print("Save Graph Information...")
    tmp_meta = dataset.meta_info.reset_index()
    tmp_meta.loc[len(tmp_meta)] = ["node_count", graph.num_nodes()]
    tmp_meta.loc[len(tmp_meta)] = ["edge_count", graph.num_edges()]
    tmp_meta.loc[len(tmp_meta)] = ["feature_dim", int(graph.ndata["feat"].shape[1])]
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
    parser.add_argument(
        "--no-f32-features",
        action="store_true",
        help="Keep float64 features as-is instead of downcasting to "
             "float32 (downcast is the default; halves feature RAM).",
    )
    args = parser.parse_args()
    main(args)
