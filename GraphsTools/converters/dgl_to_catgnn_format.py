import argparse
import os
import json
import numpy as np
from pathlib import Path


def load_info_json(graph_path: str) -> dict:
    """Loads the optional <dataset>_info.json written by the download script,
    if it sits next to the .dgl file (same stem, '_info.json' suffix)."""
    stem = os.path.splitext(graph_path)[0]
    info_path = f"{stem}_info.json"
    if not os.path.exists(info_path):
        return {}
    print(f"Found accompanying info file ({info_path}), loading for sanity checks...")
    with open(info_path, "r") as f:
        return json.load(f)


def main(args):
    print("Import dgl...")
    import dgl
    from dgl.data.utils import load_graphs

    info = load_info_json(args.graph_path)

    print("Load Graph...")
    graphs, _ = load_graphs(args.graph_path)
    graph = graphs[0]

    print("Ensure graph is on CPU...")
    graph = graph.cpu()

    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    print(f"Nodes: {num_nodes}  Edges: {num_edges}")

    if info:
        if "node_count" in info and info["node_count"] != num_nodes:
            print(f"  WARNING: node_count in info.json ({info['node_count']}) "
                  f"!= graph.num_nodes() ({num_nodes})")
        if "edge_count" in info and info["edge_count"] != num_edges:
            print(f"  WARNING: edge_count in info.json ({info['edge_count']}) "
                  f"!= graph.num_edges() ({num_edges})")

    dataset_dir = (Path(args.output_path) / f"datasets/{args.dataset_name}/").as_posix()
    print(f"Create output directory ({dataset_dir})...")
    os.makedirs(dataset_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # edge_list.csv
    # -----------------------------------------------------------------
    print("Convert DGL graph to edge_list.csv...")
    src, dst = graph.edges()
    edge_list_path = os.path.join(dataset_dir, "edge_list.csv")
    print(f"Write edge list file ({edge_list_path})...")
    with open(edge_list_path, "w") as f:
        for u, v in zip(src.tolist(), dst.tolist()):
            f.write(f"{u},{v}\n")

    # -----------------------------------------------------------------
    # feats.npy
    # -----------------------------------------------------------------
    if args.feat_key not in graph.ndata:
        raise KeyError(
            f"ndata key '{args.feat_key}' not found on graph. "
            f"Available ndata keys: {list(graph.ndata.keys())}. "
            f"Use --feat-key to point to the correct one."
        )
    print(f"Extract node features (ndata['{args.feat_key}'])...")
    feats = graph.ndata[args.feat_key].numpy()
    if info and "feature_dim" in info and info["feature_dim"] != feats.shape[1]:
        print(f"  WARNING: feature_dim in info.json ({info['feature_dim']}) "
              f"!= feats.shape[1] seen in graph ({feats.shape[1]})")
    feats_path = os.path.join(dataset_dir, "feats.npy")
    print(f"Write features file ({feats_path})...")
    np.save(feats_path, feats)

    # -----------------------------------------------------------------
    # class_map.json
    # -----------------------------------------------------------------
    if args.label_key not in graph.ndata:
        raise KeyError(
            f"ndata key '{args.label_key}' not found on graph. "
            f"Available ndata keys: {list(graph.ndata.keys())}. "
            f"Use --label-key to point to the correct one."
        )
    print(f"Extract labels (ndata['{args.label_key}'])...")
    labels = graph.ndata[args.label_key]

    # Auto-detect multilabel from tensor shape (labels.dim() > 1), unless
    # explicitly overridden via --multilabel / --no-multilabel. The graph_prep
    # step in the download script only squeezes (N,1) -> (N,); true multilabel
    # tensors (N, num_classes) stay 2D, so shape is a reliable signal here.
    detected_multilabel = labels.dim() > 1
    if args.multilabel is None:
        is_multilabel = detected_multilabel
        print(f"  Auto-detected multilabel={is_multilabel} from label shape {tuple(labels.shape)}")
    else:
        is_multilabel = args.multilabel
        if is_multilabel != detected_multilabel:
            print(f"  WARNING: --multilabel={is_multilabel} overrides "
                  f"auto-detection ({detected_multilabel}) from label shape {tuple(labels.shape)}")

    if info and "num_classes" in info and not is_multilabel:
        n_classes_seen = int(labels.long().max().item()) + 1
        if info["num_classes"] != n_classes_seen:
            print(f"  WARNING: num_classes in info.json ({info['num_classes']}) "
                  f"!= max(label)+1 seen in graph ({n_classes_seen})")

    if is_multilabel:
        label_list = labels.numpy().tolist()
        class_map = {str(i): label_list[i] for i in range(num_nodes)}
    else:
        label_list = labels.long().tolist()
        class_map = {str(i): int(label_list[i]) for i in range(num_nodes)}

    class_map_path = os.path.join(dataset_dir, "class_map.json")
    print(f"Write class map file ({class_map_path})...")
    with open(class_map_path, "w") as f:
        json.dump(class_map, f)

    # -----------------------------------------------------------------
    # role.json
    # -----------------------------------------------------------------
    print("Build train/val/test split...")
    has_masks = all(
        k in graph.ndata for k in (args.train_mask_key, args.val_mask_key, args.test_mask_key)
    )

    if has_masks:
        print("Using existing split masks from ndata...")
        train_ids = graph.ndata[args.train_mask_key].nonzero(as_tuple=True)[0].tolist()
        val_ids = graph.ndata[args.val_mask_key].nonzero(as_tuple=True)[0].tolist()
        test_ids = graph.ndata[args.test_mask_key].nonzero(as_tuple=True)[0].tolist()
    else:
        print(
            f"No split masks found (looked for '{args.train_mask_key}', "
            f"'{args.val_mask_key}', '{args.test_mask_key}') — "
            f"generating random split with seed {args.split_seed}..."
        )
        rng = np.random.default_rng(args.split_seed)
        perm = rng.permutation(num_nodes)
        n_train = int(num_nodes * args.train_ratio)
        n_val = int(num_nodes * args.val_ratio)
        train_ids = perm[:n_train].tolist()
        val_ids = perm[n_train:n_train + n_val].tolist()
        test_ids = perm[n_train + n_val:].tolist()

    print(f"train: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")
    if info:
        for key, ids in (("train_count", train_ids), ("val_count", val_ids), ("test_count", test_ids)):
            if key in info and info[key] != len(ids):
                print(f"  WARNING: {key} in info.json ({info[key]}) != count here ({len(ids)})")

    role = {"tr": train_ids, "va": val_ids, "te": test_ids}
    role_path = os.path.join(dataset_dir, "role.json")
    print(f"Write role file ({role_path})...")
    with open(role_path, "w") as f:
        json.dump(role, f)

    print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a DGL graph file into the SDT_GNN dataset format "
                     "(edge_list.csv, feats.npy, class_map.json, role.json)."
    )
    parser.add_argument("-g", "--graph-path", type=str, required=True,
                        help="Path to the dgl graph file.")
    parser.add_argument("-o", "--output-path", type=str, required=True,
                        help="Base 'datasets' directory. A subfolder named "
                             "--dataset-name will be created inside it, "
                             "matching what SDT_GNN.Partitioning expects as 'path'.")
    parser.add_argument("-n", "--dataset-name", type=str, required=True,
                        help="Dataset name (used as the output subfolder name, "
                             "and later passed to SDT_GNN.Partitioning as 'dataset').")
    parser.add_argument("--feat-key", type=str, default="feat",
                        help="ndata key holding node features (default: 'feat').")
    parser.add_argument("--label-key", type=str, default="label",
                        help="ndata key holding node labels (default: 'label').")
    multilabel_group = parser.add_mutually_exclusive_group()
    multilabel_group.add_argument("--multilabel", dest="multilabel", action="store_true",
                        default=None,
                        help="Force multilabel handling (label vector per node). "
                             "Default: auto-detected from label tensor shape.")
    multilabel_group.add_argument("--no-multilabel", dest="multilabel", action="store_false",
                        help="Force single-label handling (one class index per node). "
                             "Default: auto-detected from label tensor shape.")
    parser.add_argument("--train-mask-key", type=str, default="train_mask",
                        help="ndata key holding the train mask (default: 'train_mask').")
    parser.add_argument("--val-mask-key", type=str, default="val_mask",
                        help="ndata key holding the val mask (default: 'val_mask').")
    parser.add_argument("--test-mask-key", type=str, default="test_mask",
                        help="ndata key holding the test mask (default: 'test_mask').")
    parser.add_argument("--train-ratio", type=float, default=0.6,
                        help="Fallback random split: fraction of nodes for training "
                             "if no masks are found (default: 0.6).")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fallback random split: fraction of nodes for validation "
                             "if no masks are found (default: 0.2). "
                             "Remainder goes to test.")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Random seed for the fallback split (default: 42).")
    args = parser.parse_args()
    main(args)
