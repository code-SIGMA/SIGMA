"""
download_twitch_graph.py
Downloads the Twitch-Gamers dataset from SNAP, converts it to a DGL graph,
and saves it together with a JSON metadata file.

Usage:
    python download_twitch_graph.py -d twitch -s ./output
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    """Stream-download *url* into *dest*, showing a simple progress indicator."""
    response = requests.get(url, stream=True, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Cannot download file (HTTP {response.status_code}): {url}")

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1 << 20  # 1 MiB

    with dest.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded >> 20} / {total >> 20} MiB  ({pct:.1f}%)",
                          end="", flush=True)
    print()  # newline after progress


def _edge_hash(edges) -> str:
    """Return a short SHA-256 digest of the edge list as a dataset fingerprint."""
    raw = edges.to_csv(index=False).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def make_split_masks(num_nodes: int, train_ratio: float = 0.6,
                     val_ratio: float = 0.2, seed: int = 42):
    """Creates random train/val/test masks when not provided by the graph."""
    import torch as th

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))

    train_mask = th.zeros(num_nodes, dtype=th.bool)
    val_mask = th.zeros(num_nodes, dtype=th.bool)
    test_mask = th.zeros(num_nodes, dtype=th.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Defer heavy imports so --help is instant
    print("Importing dgl  ")
    import dgl
    print("Importing pandas  ")
    import pandas as pd
    print("Importing torch  ")
    import torch as th

    url = "https://snap.stanford.edu/data/twitch_gamers.zip"

    # ---- Download & extract ------------------------------------------------
    print("Downloading dataset  ")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "twitch_gamers.zip"

        _download_with_progress(url, zip_path)

        print("Extracting  ")
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(path=tmp)

        print("Reading CSV files  ")
        features = pd.read_csv(tmp / "large_twitch_features.csv")
        edges = pd.read_csv(tmp / "large_twitch_edges.csv")

    # ---- Build node-ID mapping ---------------------------------------------
    print("Converting dataset to DGL graph  ")
    node_ids = features["numeric_id"].values
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    src = edges["numeric_id_1"].map(id_map)
    dst = edges["numeric_id_2"].map(id_map)

    # Drop edges whose endpoints are not present in the feature table
    valid_mask = src.notna() & dst.notna()
    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        print(f"  Warning: dropping {n_dropped} edge(s) with unmapped node IDs.")
    src = src[valid_mask]
    dst = dst[valid_mask]

    src_t = th.tensor(src.values, dtype=th.long)
    dst_t = th.tensor(dst.values, dtype=th.long)

    # ---- Create graph & attach features/labels -----------------------------
    num_nodes = len(node_ids)
    graph = dgl.graph((src_t, dst_t), num_nodes=num_nodes)

    # "mature" is the label -- exclude it from input features to avoid leakage
    feature_cols = ["views", "life_time", "dead_account", "affiliate"]
    node_features = th.tensor(features[feature_cols].values, dtype=th.float32)

    graph.ndata["features"] = node_features

    labels = th.tensor(features["mature"].values, dtype=th.long)
    graph.ndata["labels"] = labels

    num_classes = int(features["mature"].nunique())

    # ---- Generate train/val/test masks -------------------------------------
    print("Generating train/val/test split masks  ")
    train_mask, val_mask, test_mask = make_split_masks(graph.num_nodes())
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    print(f"  Train nodes : {train_mask.sum().item():,}")
    print(f"  Val nodes   : {val_mask.sum().item():,}")
    print(f"  Test nodes  : {test_mask.sum().item():,}")

    # Sanity-Check
    required = ["features", "labels", "train_mask", "val_mask", "test_mask"]
    missing = [k for k in required if k not in graph.ndata]
    if missing:
        raise RuntimeError(f"Still missing after normalization: {missing}")

    # ---- Post-processing (Metis requirements) ------------------------------
    print("Making graph bidirected and removing duplicate edges  ")
    graph = dgl.to_simple(
        dgl.to_bidirected(graph, copy_ndata=True),
        copy_ndata=True,
        copy_edata=True,
    )

    print("Removing self-loops  ")
    graph = dgl.remove_self_loop(graph)

    print(f"  Nodes : {graph.num_nodes():,}")
    print(f"  Edges : {graph.num_edges():,}")

    # ---- Save --------------------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dgl_path = save_dir / f"{args.dataset}.dgl"
    info_path = save_dir / f"{args.dataset}_info.json"

    print(f"Saving graph to {dgl_path}  ")
    dgl.save_graphs(str(dgl_path), [graph])

    information = {
        "name":         args.dataset,
        "node_count":   graph.num_nodes(),
        "edge_count":   graph.num_edges(),
        "num_classes":  num_classes,
        "feature_dim":  int(graph.ndata["features"].shape[1]),
        "feature_cols": feature_cols,
        "hash":         _edge_hash(edges),
        "url":          url,
        "split": {
            "train": int(graph.ndata["train_mask"].sum()),
            "val":   int(graph.ndata["val_mask"].sum()),
            "test":  int(graph.ndata["test_mask"].sum()),
        },
    }

    print(f"Saving metadata to {info_path}  ")
    with info_path.open("w") as fh:
        json.dump(information, fh, indent=4)

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and convert the Twitch-Gamers graph to DGL format."
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=["twitch"],
        required=True,
        help="Dataset to download.",
    )
    parser.add_argument(
        "-s", "--save-dir",
        required=True,
        help="Directory in which to save the .dgl graph and _info.json file.",
    )
    main(parser.parse_args())
