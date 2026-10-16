#================================================================================
# Libraries
#================================================================================

import os
import sys

sys.path.append("../")

import argparse
import copy
import json
import time
from pathlib import Path

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


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with th.no_grad():
        logits = model(graph, features, True)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), correct.item(), len(labels)


def reduce_mean(value):
    value_t = th.tensor(value, dtype=th.float32)
    if args.distributed:
        dist.all_reduce(value_t, op=dist.ReduceOp.SUM)
        value_t /= args.world_size
    return float(value_t.item())


def reduce_accuracy(correct, count):
    correct_t = th.tensor(correct, dtype=th.float32)
    count_t = th.tensor(count, dtype=th.float32)
    if args.distributed:
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    return float((correct_t / count_t).item())


def run(g, data):
    n_classes, _, _, rank, _ = data

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    in_feats = features.shape[1]

    if args.gpu >= 0:
        raise ValueError("This DistGNN training script is CPU-only. Use --gpu -1.")

    train_nid = train_mask.nonzero(as_tuple=False).squeeze()

    g = g.int()

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

    # use optimizer
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    graph_path = Path(args.graph_path)
    gr = load_graphs(graph_path.as_posix())[0][0]

    gr_features = gr.ndata["feat"]
    gr_labels = gr.ndata["label"]

    gr_train_mask = gr.ndata["train_mask"]
    gr_train_nid = gr_train_mask.nonzero(as_tuple=False).squeeze().squeeze()
    gr_val_mask = gr.ndata["val_mask"]
    gr_val_nid = gr_val_mask.nonzero(as_tuple=False).squeeze().squeeze()
    gr_test_mask = gr.ndata["test_mask"]
    gr_test_nid = gr_test_mask.nonzero(as_tuple=False).squeeze().squeeze()

    gr = gr.int()

    best_val_acc = float("-inf")
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = -1

    # initialize graph
    total_time_sum = 0
    for epoch in range(args.n_epochs):
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

        # Select the checkpoint on validation only; test is evaluated once at the end.
        _, train_correct, train_count = evaluate(model, gr, gr_features, gr_labels, gr_train_nid)
        train_cum_acc1 = reduce_accuracy(train_correct, train_count)
        _, val_correct, val_count = evaluate(model, gr, gr_features, gr_labels, gr_val_nid)
        val_cum_acc1 = reduce_accuracy(val_correct, val_count)
        _, test_correct, test_count = evaluate(model, gr, gr_features, gr_labels, gr_test_nid)
        test_cum_acc1 = reduce_accuracy(test_correct, test_count)

        current_val_acc = val_cum_acc1
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch


        if rank == 0 and args.val:
            format_str = "Epoch {:05d} | Total Time(s) {:.4f} | Forward Time(s) {:.4f} | Backward Time(s) {:.4f} | Update Time(s) {:.4f} | Train Loss {:.4f} | Train Accuracy {:.4f} | Val Accuracy {:.4f} | Test Accuracy {:.4f} | Best Val Epoch {:05d}"

            print(format_str.format(
                epoch,
                total_time,
                forward_time,
                backward_time,
                update_time,
                loss.item(),
                train_cum_acc1 * 100,
                val_cum_acc1 * 100,
                test_cum_acc1 * 100,
                best_epoch,
            ),
                  flush=True,
                  )

        if  not args.val and args.rank == 0:
            print(f"Epoch: {epoch} time: {total_time:0.4} sec", flush=True)
            print()

    model.load_state_dict(best_model_state)

    gr = load_graphs(graph_path.as_posix())[0][0]

    gr_test_mask = gr.ndata["test_mask"]
    gr_test_nid = gr_test_mask.nonzero(as_tuple=False).squeeze().squeeze()

    gr = gr.int()

    # Train
    _, train_correct, train_count = evaluate(model, gr, gr_features, gr_labels, gr_train_nid)
    train_cum_acc1 = reduce_accuracy(train_correct, train_count)
    # Test
    _, test_correct, test_count = evaluate(model, gr, gr_features, gr_labels, gr_test_nid)
    test_cum_acc1 = reduce_accuracy(test_correct, test_count)

    if args.rank == 0:
        print(
            "#############################################################", flush=True
        )
        print(
            "Single node accuracy: " \
            f"Train Accuracy {train_cum_acc1 * 100:.4f} | " \
            f"Best Val Accuracy {best_val_acc * 100:.4f} | " \
            f"Test Accuracy {test_cum_acc1 * 100:.4f} | " \
            f"Best Val Epoch {best_epoch:05d} | " \
            f"Epochs/s: {args.n_epochs / total_time_sum:.4f}",
            flush=True)
        print(
            "#############################################################", flush=True
        )


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
        "--n-layers", type=int, default=1, help="number of hidden gcn layers"
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
    assert n_inner == node_feats["_N/feat"].shape[0], \
        f"inner_node={n_inner} != feats={node_feats['_N/feat'].shape[0]}"
    contiguous = bool(th.all(inner[:n_inner]))

    def _place(src):
        full = th.zeros((graph.num_nodes(), *src.shape[1:]), dtype=src.dtype)
        if contiguous:
            full[:n_inner] = src
        else:
            full[inner] = src
        return full

    for name in ("feat", "label", "train_mask", "val_mask", "test_mask"):
        graph.ndata[name] = _place(node_feats[f"_N/{name}"])
    #--------------------------------

    #----------------------------------------------------------------------------------
    # Load Original Graph
    #----------------------------------------------------------------------------------
    graph_path = Path(args.graph_path)
    g_orig = load_graphs(graph_path.as_posix())[0][0]
    #----------------------------------------------------------------------------------

    labels = g_orig.ndata["label"][np.arange(g_orig.number_of_nodes())]
    #--------------------------------

    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    print("n_classes: ", n_classes, flush=True)

    # Without drpa args.nr == -1
    gobj = drpa(graph, args.rank, num_parts, node_map, -1, dist, args.n_layers)
    data = n_classes, node_map, num_parts, args.rank, args.world_size
    main(gobj, data)

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
