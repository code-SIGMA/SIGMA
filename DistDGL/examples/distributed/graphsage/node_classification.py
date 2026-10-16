#================================================================================
# Library
#================================================================================
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "libs"))
sys.path.append(str(BASE_DIR / "core"))

import numpy as np

import dgl
import dgl.distributed
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch._dynamo  # noqa: F401
import torch.nested._internal.nested_tensor  # noqa: F401

from args import build_parser
from bootstrap import load_environment, init_distributed, pick_device, prepare_metrics_dir
from checks import run_startup_checks
from data import split_nodes, infer_n_classes
from evaluation import compute_acc, evaluate
from model import DistSAGE
from run_logging import RunLogger


#================================================================================
# Training
#================================================================================

def build_model(args, in_feats, n_classes, device):
    model = DistSAGE(in_feats, args.num_hidden, n_classes,
                     args.num_layers, F.relu, args.dropout).to(device)
    if args.num_gpus == 0:
        return th.nn.parallel.DistributedDataParallel(model)
    return th.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )


def run(args, device, g, train_nid, val_nid, test_nid, n_classes, logger):
    """Train and evaluate DistSAGE. Returns (mean epoch time, test accuracy)."""
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.distributed.DistNodeDataLoader(
        g, train_nid, sampler,
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )
    model = build_model(args, g.ndata["feat"].shape[1], n_classes, device)
    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    rank = g.rank()
    best_val_acc = float("-inf")
    best_model_state = None
    best_epoch = -1
    epoch_time = []
    test_acc = -1.0

    def sync():
        if device.type == "cuda":
            th.cuda.synchronize(device)

    try:
        for epoch in range(args.num_epochs):
            is_last_epoch = epoch == args.num_epochs - 1

            th.distributed.barrier()

            tic = time.time()
            sample_time = data_time = forward_time = backward_time = update_time = 0.0
            loss_sum = 0.0
            num_seeds = 0
            start = time.time()

            with model.join():
                for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                    t0 = time.time()
                    sample_time += t0 - start                 # Neighbor-Sampling

                    batch_inputs = g.ndata["feat"][input_nodes]   # Remote-Fetch
                    batch_labels = g.ndata["label"][seeds].long()
                    blocks = [b.to(device) for b in blocks]
                    batch_inputs = batch_inputs.to(device)
                    batch_labels = batch_labels.to(device)
                    sync()
                    t1 = time.time()
                    data_time += t1 - t0                      # Fetch + H2D

                    batch_pred = model(blocks, batch_inputs)
                    loss = loss_fcn(batch_pred, batch_labels)
                    sync()
                    t2 = time.time()

                    optimizer.zero_grad()
                    loss.backward()                           # inkl. DDP-Allreduce
                    sync()
                    t3 = time.time()

                    optimizer.step()
                    sync()
                    t4 = time.time()

                    forward_time += t2 - t1
                    backward_time += t3 - t2
                    update_time += t4 - t3

                    num_seeds += len(seeds)
                    loss_sum += loss.item() * len(batch_labels)

                    if (step + 1) % args.log_every == 0 and rank == 0:
                        acc = compute_acc(batch_pred, batch_labels)
                        print(f"\nPart {rank} | Epoch {epoch:05d} | Step {step:05d}"
                              f" | Loss {loss.item():.4f} | Train Acc {acc.item():.4f}\n",
                              flush=True)
                    start = time.time()

            total_time = time.time() - tic
            epoch_time.append(total_time)

            # ---------------- outside the timed section from here on ----------------
            # Global loss (weighted by seeds) and maximum epoch time
            s = th.tensor([loss_sum, float(num_seeds)], dtype=th.float64)
            th.distributed.all_reduce(s, op=th.distributed.ReduceOp.SUM)
            loss_val = s[0].item() / max(s[1].item(), 1.0)

            t = th.tensor([total_time], dtype=th.float64)
            th.distributed.all_reduce(t, op=th.distributed.ReduceOp.MAX)
            max_time = t.item()

            # ---------------- Evaluation (ALL ranks, collective) ----------------
            train_acc = val_acc = test_acc_epoch = -1.0
            do_eval = epoch % args.eval_every == 0 or is_last_epoch
            if do_eval:
                nid_dict = {"train": train_nid, "val": val_nid}
                if is_last_epoch:
                    nid_dict["test"] = test_nid
                res = evaluate(model.module, g, g.ndata["feat"], g.ndata["label"],
                               nid_dict, args.batch_size_eval, device)
                train_acc, val_acc = res["train"], res["val"]
                if is_last_epoch:
                    test_acc_epoch = test_acc = res["test"]

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_model_state = {k: v.detach().clone()
                                        for k, v in model.module.state_dict().items()}

            logger.log_epoch(
                {
                    "epoch": epoch,
                    "train_loss": loss_val,
                    "total_time": total_time,
                    "forward_time": forward_time,
                    "backward_time": backward_time,
                    "wait_time": sample_time + data_time,
                    "update_time": update_time,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc_epoch,
                },
                do_eval=do_eval, max_time=max_time,
                sample_time=sample_time, data_time=data_time,
            )
    finally:
        logger.close()

    # ---------------- Best Model Evaluation ----------------
    if best_model_state is not None:
        model.module.load_state_dict(best_model_state)
        res = evaluate(model.module, g, g.ndata["feat"], g.ndata["label"],
                       {"train": train_nid, "val": val_nid, "test": test_nid},
                       args.batch_size_eval, device)
        if rank == 0:
            logger.log_best(res, best_epoch)
        test_acc = res["test"]

    return np.mean(epoch_time[-max(1, int(args.num_epochs * 0.8)):]), test_acc


#================================================================================
# Main
#================================================================================

def main(args):
    run_id, database = load_environment("/root/env.json")
    g = init_distributed(args)
    run_startup_checks(g, args, database_loaded=bool(database))

    data_dirpath = prepare_metrics_dir(args, g.rank())
    train_nid, val_nid, test_nid = split_nodes(g)
    device = pick_device(g.rank(), args.num_gpus)
    n_classes = infer_n_classes(g, args.n_classes)

    logger = RunLogger(args, g.rank(), device, data_dirpath, run_id, database)
    epoch_time, test_acc = run(args, device, g, train_nid, val_nid, test_nid,
                               n_classes, logger)
    print(
        f"Summary of node classification(GraphSAGE): GraphName "
        f"{args.graph_name} | TrainEpochTime(mean) {epoch_time:.4f} "
        f"| TestAccuracy {test_acc:.4f}"
    )


if __name__ == "__main__":
    args = build_parser()
    print(f"Arguments: {args}")
    main(args)
