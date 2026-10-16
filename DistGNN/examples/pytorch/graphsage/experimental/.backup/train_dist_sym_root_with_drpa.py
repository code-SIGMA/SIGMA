#================================================================================
# Libraries
#================================================================================

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

sys.path.append(str(BASE_DIR.parent.parent))
sys.path.append(str(BASE_DIR / "utils"))

import argparse
import json
import time

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from dgl.data import register_data_args
from dgl.data.utils import load_graphs, load_tensors
from dgl.distgnn.drpa_sym import drpa
from dgl.nn.pytorch.conv import SAGEConv
from torch import nn

try:
    import torch_ccl
except ImportError as e:
    print(f"[startup] torch_ccl not available: {e}", flush=True)

from drpa_meta import (
    attach_drpa_meta_to_graph,
    validate_leaders_consistent_dist,
)

#================================================================================
# Main Script
#================================================================================

class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        dropout,
        aggregator_type,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        # remote_ = False

        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))

    def forward(self, graph, inputs, inf=False):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def evaluate_all(model, graph, features, labels, nid_dict):
    model.eval()
    with th.no_grad():
        logits = model(graph, features)
        _, pred = th.max(logits, dim=1)
        return {k: (int((pred[nid] == labels[nid]).sum()), len(nid))
                for k, nid in nid_dict.items()}

def run(g, data):
    n_classes, _, _, rank, _ = data

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    in_feats = features.shape[1]

    if args.gpu >= 0:
        raise ValueError("This DistGNN training script is CPU-only. Use --gpu -1.")

    train_nid = train_mask.nonzero(as_tuple=False).squeeze()

    # create GraphSAGE model
    model = GraphSAGE(
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers,
        F.relu,
        args.dropout,
        args.aggregator_type,
    )

    if args.distributed:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    # use optimizer
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    graph_path = Path(args.graph_path)
    gr = load_graphs(graph_path.as_posix())[0][0]
    gr = gr.int()

    gr_features = gr.ndata["feat"]
    gr_labels = gr.ndata["label"]

    gr_train_mask = gr.ndata["train_mask"]
    gr_train_nid = gr_train_mask.nonzero(as_tuple=False).squeeze()
    gr_val_mask = gr.ndata["val_mask"]
    gr_val_nid = gr_val_mask.nonzero(as_tuple=False).squeeze()
    gr_test_mask = gr.ndata["test_mask"]
    gr_test_nid = gr_test_mask.nonzero(as_tuple=False).squeeze()

    best_val_acc = float("-inf")
    best_model_state = None
    best_epoch = -1

    # initialize graph
    total_time_sum = 0
    max_time_sum = 0
    max_time = 0
    for epoch in range(args.n_epochs):
        is_last_epoch = epoch == args.n_epochs-1
        start_time = time.time()
        model.train()

        # forward
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])

        forward_end = time.time()
        optimizer.zero_grad()
        loss.backward()
        compute_end = time.time()

        if args.distributed:
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= args.world_size

        optimizer.step()
        update_time = time.time() - compute_end

        forward_time = forward_end - start_time
        backward_time = compute_end - forward_end
        total_time = forward_time + backward_time + update_time

        total_time_sum += total_time

        max_time = total_time
        if args.distributed:
            t = th.tensor(total_time, dtype=th.float64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            max_time = float(t.item())

        max_time_sum += max_time

        if rank == 0 and (epoch % args.eval_each_n_epoch == 0 or is_last_epoch):

            nid_dict = {"train": gr_train_nid, "val":   gr_val_nid}

            if is_last_epoch:
                nid_dict["test"] = gr_test_nid

            res = evaluate_all(model, gr, gr_features, gr_labels, nid_dict)

            train_cum_acc1 = res["train"][0] / res["train"][1]
            val_cum_acc1   = res["val"][0]   / res["val"][1]
            if is_last_epoch:
                test_cum_acc1   = res["test"][0]   / res["test"][1]

            if val_cum_acc1 > best_val_acc:
                best_val_acc = val_cum_acc1
                best_model_state = {k: v.detach().clone()
                                    for k, v in model.state_dict().items()}
                best_epoch = epoch

            msg = (f"Epoch {epoch:05d} | Max Time(s) {max_time:.4f} | Total Time(s) {total_time:.4f} | "
                   f"Forward Time(s) {forward_time:.4f} | Backward Time(s) {backward_time:.4f} | "
                   f"Update Time(s) {update_time:.4f} | Train Loss {loss.item():.4f} | "
                   f"Train Accuracy {train_cum_acc1*100:.4f} | Val Accuracy {val_cum_acc1*100:.4f} | ")
            if is_last_epoch:
                msg += f"Test Accuracy {test_cum_acc1*100:.4f} | "
            msg += f"Best Epoch {best_epoch:05d}"

            print(msg, flush=True)
        elif rank == 0:
            print(f"Epoch: {epoch} time: {total_time:0.4} sec", flush=True)
            print()

        if args.distributed:
            dist.barrier()


    if rank == 0 and best_model_state is not None:
        model.load_state_dict(best_model_state)
        res = evaluate_all(model, gr, gr_features, gr_labels,
                            {"train": gr_train_nid,
                             "val":   gr_val_nid,
                             "test": gr_test_nid})

        best_train_acc = res["train"][0] / res["train"][1]
        best_val_acc   = res["val"][0]   / res["val"][1]
        best_test_acc   = res["test"][0]   / res["test"][1]

        print(
            "#############################################################", flush=True
        )
        print(
            "Best Model Single Node Evaluation:\n" \
            f"\tTrain Accuracy: {best_train_acc * 100:.4f}\n" \
            f"\tVal Accuracy: {best_val_acc * 100:.4f}\n" \
            f"\tTest Accuracy: {best_test_acc * 100:.4f}\n" \
            f"\tFrom which Epoch: {best_epoch:05d}\n" \
            f"General Epochs/s for rank == 0: {args.n_epochs / total_time_sum:.4f}",
            f"General Epochs/s: {args.n_epochs / max_time_sum:.4f}",
            flush=True)
        print(
            "#############################################################", flush=True
        )
    else:
        print(f"No Checkpoint. n_epochs({args.n_epochs}) to small for eval_each_n_epoch({args.eval_each_n_epoch})")

#----------------------------------------------------------------------------------
# DRPA metadata
#----------------------------------------------------------------------------------

def attach_drpa_metadata(graph, part_metadata, part_config_parent,
                         rank, num_parts, world_size, args):
    if args.nr == -1:
        return None

    node_map = attach_drpa_meta_to_graph(
        graph, part_metadata, part_config_parent, rank, num_parts,
        require_orig_nid=True,
        trust_meta=False,
        log=lambda m: print(f"Rank: {rank}, Message: {m}", flush=True),
    )

    assert "_ID" in graph.ndata, "_ID doesnt exist in graph.ndata"
    gids_key = "_ID"
    if gids_key in graph.ndata:
        validate_leaders_consistent_dist(
            graph.dstdata["lf"], graph.ndata[gids_key],
            graph.ndata["inner_node"], rank, world_size, dist)
        if rank == 0 and world_size > 1:
            print()
            print(f"Rank: {rank}, Message: DRPA global-consistency check passed", flush=True),
    else:
        print(f"Rank: {rank}, Message: no reshuffled NID; skip global-consistency check", flush=True),
    return node_map

#----------------------------------------------------------------------------------
def main(graph, data):
    print("Starting a run: ")
    run(graph, data)
    print("Run completed !!!")

if __name__ == "__main__":
    #----------------------------------------------------------------------------------
    # Parser Setup
    #----------------------------------------------------------------------------------
    th.set_printoptions(threshold=10)

    parser = argparse.ArgumentParser(description="GraphSAGE")
    register_data_args(parser)
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--nr", type=int, default=1, help="#delay in delayed updates")
    parser.add_argument("--val", default=False, action="store_true")
    parser.add_argument("--lr", type=float, default=0.032, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="number of hidden gcn layers"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    parser.add_argument(
        "--aggregator-type",
        type=str,
        default="gcn",
        choices=["gcn", "mean", "pool", "lstm"],
        help="Aggregator type: mean/gcn/pool/lstm",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="ccl", type=str, help="distributed backend"
    )

    parser.add_argument(
        "--graph-path", type=str, help="Path to dgl graph.", required=True
    )

    parser.add_argument(
        "--partition-config-path", type=str, help="Path to partition config.", required=True
    )

    parser.add_argument(
        "--eval-each-n-epoch", default=1, type=int, help="Epoch interval for evaluation"
    )

    args = parser.parse_args()
    print(args)
    #----------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("PMI_SIZE", "-1"))
        if args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1
    if args.distributed:
        args.rank = int(os.environ.get("PMI_RANK", "-1"))
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    print("Rank: ", args.rank, " World_size: ", args.world_size)
    nc = args.world_size
    #----------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------
    # Load Partition config
    #----------------------------------------------------------------------------------

    part_config_path = Path(args.partition_config_path)

    if not part_config_path.exists():
        raise FileNotFoundError(f"The Partition Config Path '{part_config_path.as_posix()}' doesn't exists.")

    with open(part_config_path.as_posix()) as conf_f:
        part_metadata = json.load(conf_f)
    #----------------------------------------------------------------------------------


    #----------------------------------------------------------------------------------
    # Validate Partitions
    #----------------------------------------------------------------------------------
    part_files = part_metadata[f"part-{args.rank}"]
    assert "node_feats" in part_files, "the partition does not contain node features."
    assert "edge_feats" in part_files, "the partition does not contain edge feature."
    assert "part_graph" in part_files, "the partition does not contain graph structure."
    #----------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------
    # Load from Partition
    #----------------------------------------------------------------------------------
    node_feats = load_tensors((part_config_path.parent / part_files["node_feats"]).as_posix())

    graph = load_graphs((part_config_path.parent / part_files["part_graph"]).as_posix())[0][0]
    #----------------------------------------------------------------------------------

    num_parts = part_metadata["num_parts"]
    node_map = part_metadata["node_map"]


    #--------------------------------
    inner = graph.ndata["inner_node"].bool()
    n_inner = int(inner.sum())
    assert n_inner == node_feats["_N/train_mask"].shape[0], \
        f"inner_node={n_inner} != masks={node_feats['_N/train_mask'].shape[0]}"
    contiguous = bool(th.all(inner[:n_inner]))

    def _place(src):
        full = th.zeros((graph.num_nodes(), *src.shape[1:]), dtype=src.dtype)
        if contiguous:
            full[:n_inner] = src
        else:
            full[inner] = src
        return full

    for name in ("train_mask", "val_mask", "test_mask"):
        graph.ndata[name] = _place(node_feats[f"_N/{name}"])

    orig_nid = graph.ndata["orig_id"].long()
    assert orig_nid.shape[0] == graph.num_nodes()

    graph_path = Path(args.graph_path)
    g_orig = load_graphs(graph_path.as_posix())[0][0]
    for name in ("feat", "label"):
        graph.ndata[name] = g_orig.ndata[name][orig_nid]

    sel = graph.ndata["feat"][:n_inner] if contiguous else graph.ndata["feat"][inner]
    assert th.allclose(sel, node_feats["_N/feat"]), \
        "_ID mapping does not match the partition features"

    labels_all = g_orig.ndata["label"]
    n_classes = len(th.unique(labels_all[th.logical_not(th.isnan(labels_all))]))
    del g_orig, labels_all
    print("n_classes: ", n_classes, flush=True)
    #--------------------------------

    #----------------------------------------------------------------------------------
    # Load DRPA Node Map
    #----------------------------------------------------------------------------------
    drpa_node_map = attach_drpa_metadata(graph, part_metadata, part_config_path.parent,
                                         args.rank, num_parts, args.world_size, args)
    #----------------------------------------------------------------------------------

    drpa_enabled = args.nr != -1
    if drpa_enabled:
        if drpa_node_map is None:
            raise RuntimeError("attach_drpa_metadata lieferte keine node_map")
        gobj = drpa(graph, args.rank, num_parts, drpa_node_map, args.nr, dist, args.n_layers)
    else:
        gobj = graph

    data = n_classes, node_map, num_parts, args.rank, args.world_size
    main(gobj, data)

    if drpa_enabled:
        gobj.drpa_finalize()

    if args.rank == 0:
        print("Run details:")
        print("Dataset:", args.dataset)
        print("Delay: ", args.nr)
        print("lr: ", args.lr)
        print("Aggregator: ", args.aggregator_type)
        print()
        print()

    if args.distributed:
        dist.barrier()

#================================================================================
