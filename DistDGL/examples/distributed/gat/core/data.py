"""Split nodes among trainers and determine the number of classes."""
import dgl
import dgl.distributed
import numpy as np
import torch as th


def split_nodes(g):
    """Train/val/test IDs for this trainer. Returns (train_nid, val_nid, test_nid)."""
    pb = g.get_partition_book()
    kw = {"node_trainer_ids": g.ndata["trainer_id"]} if "trainer_id" in g.ndata else {}

    train_nid, val_nid, test_nid = (
        dgl.distributed.node_split(g.ndata[f"{split}_mask"], pb, force_even=True, **kw)
        for split in ("train", "val", "test")
    )

    local_nid = pb.partid2nids(pb.partid).detach().numpy()

    def n_local(nid):
        return len(np.intersect1d(nid.numpy(), local_nid))

    print(
        f"part {g.rank()}, train: {len(train_nid)} (local: {n_local(train_nid)}), "
        f"val: {len(val_nid)} (local: {n_local(val_nid)}), "
        f"test: {len(test_nid)} (local: {n_local(test_nid)})"
    )
    return train_nid, val_nid, test_nid


def infer_n_classes(g, n_classes=0):
    """Use n_classes from the arguments, or count the labels otherwise."""
    if n_classes == 0:
        labels = g.ndata["label"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print(f"Number of classes: {n_classes}")
    return n_classes
