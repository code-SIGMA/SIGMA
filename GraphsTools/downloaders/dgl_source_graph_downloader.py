import argparse
import json
from pathlib import Path

import torch as th

# Shared prep module: canonical feature names (feat/label/*_mask),
# float64 -> float32 downcast and CSC materialisation -- the work the
# training script used to redo at every start-up.
from graph_prep import prepare_graph_for_training, to_csc


def ensure_masks(graph, num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    if all(k in graph.ndata for k in ("train_mask", "val_mask", "test_mask")):
        return graph

    print("  -> No masks found, generating a random split...")
    g = th.Generator().manual_seed(seed)
    perm = th.randperm(num_nodes, generator=g)

    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_mask = th.zeros(num_nodes, dtype=th.bool)
    val_mask = th.zeros(num_nodes, dtype=th.bool)
    test_mask = th.zeros(num_nodes, dtype=th.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    return graph


def main(args):
    print("Import dgl...")
    import dgl

    print("Import Dataset...")
    match args.dataset:
        case "cora":
            from dgl.data import CoraGraphDataset as Dataset
        case "reddit":
            from dgl.data import RedditDataset as Dataset
        case "yelp":
            from dgl.data import YelpDataset as Dataset
        case "fraudyelp":
            from dgl.data import FraudYelpDataset as Dataset
        case "flickr":
            from dgl.data import FlickrDataset as Dataset
        case "coauthorphysics":
            from dgl.data import CoauthorPhysicsDataset as Dataset
        case "amazoncomputers":
            from dgl.data import AmazonCoBuyComputerDataset as Dataset
        case _:
            raise KeyError(f"Dataset '{args.dataset}' not supported.")

    print("Load Dataset...")
    dataset = Dataset()

    print("Load Graph...")
    graph = dataset[0].cpu()

    print("Ensure train/val/test masks...")
    graph = ensure_masks(graph, graph.num_nodes())

    # Canonical names (feat / label / *_mask -- what the trainer reads
    # without its alias fallback), bool masks, int64 labels, (N,1)
    # label squeeze, optional fp32 downcast. CSC is materialised later,
    # after the structural changes below.
    print("Canonicalize ndata keys and dtypes (graph_prep)...")
    graph = prepare_graph_for_training(
        graph, fp32=not args.no_f32_features, csc=False, log=print
    )

    # Sanity-Checks
    required = ["feat", "label", "train_mask", "val_mask", "test_mask"]
    missing = [k for k in required if k not in graph.ndata]
    if missing:
        raise RuntimeError(f"Still missing after normalization: {missing}")

    information = {
        "name": dataset.name,
        "node_count": graph.num_nodes(),
        "edge_count": graph.num_edges(),
        "num_classes": dataset.num_classes,
        "hash": getattr(dataset, "hash", None),
        "url": getattr(dataset, "url", None),
        "feature_dim": int(graph.ndata["feat"].shape[1]),
        "train_count": int(graph.ndata["train_mask"].sum()),
        "val_count": int(graph.ndata["val_mask"].sum()),
        "test_count": int(graph.ndata["test_mask"].sum()),
    }

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

    dgl.save_graphs(dgl_graph_save_path.as_posix(), [graph])

    print("Save Graph Information...")
    with open(dgl_info_save_path.as_posix(), "w") as f:
        json.dump(information, f, indent=4)

    print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset",
        choices=["cora", "reddit", "yelp", "fraudyelp", "flickr",
                 "coauthorphysics", "amazoncomputers"],
        help="Dataset to download from dgl.data.",
        required=True,
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
