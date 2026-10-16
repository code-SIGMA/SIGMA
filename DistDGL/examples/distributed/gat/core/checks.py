"""Collective startup checks. All ranks must call run_startup_checks."""
import dgl
import dgl.distributed
import torch as th
import torch.distributed as dist

from startup_checks import (
    gpu_report, check_gpu_isolation, format_gpu_report,
    shared_gpus_across_ranks, feature_stats_dist, check_features,
)


def abort_if_any_failed(ok, message, rank, distributed=True):
    """Turn a per-rank condition into a collective one.

    Every rank reports whether its local check passed; the failure count is summed
    across the group. If any rank failed, all ranks raise together. Without this,
    a single failing rank leaves the others blocked in the next collective until
    the timeout fires, which surfaces as "Connection reset by peer".
    """
    if not ok:
        print(f"[rank {rank}] {message}", flush=True)

    n_failed = 0 if ok else 1
    if distributed:
        counter = th.tensor([n_failed], dtype=th.int32)
        dist.all_reduce(counter, op=dist.ReduceOp.SUM)
        n_failed = int(counter.item())

    if n_failed:
        raise RuntimeError(
            f"Aborting: check failed on {n_failed} rank(s) -- see the [rank N] lines above"
        )


def _print_findings(rank, warnings, errors):
    for w in warnings:
        print(f"[rank {rank}] warning: {w}", flush=True)
    for e in errors:
        print(f"[rank {rank}] ERROR: {e}", flush=True)


def check_database(args, database_loaded, rank):
    abort_if_any_failed(
        not (args.send_data_to_database and not database_loaded),
        "--send-data-to-database used but database is not loaded",
        rank,
    )


def check_gpus(rank):
    """Every rank must own exactly one GPU, and no two ranks may land on the
    same physical device -- a shared GPU halves the throughput of both runs
    and makes the epoch times meaningless."""
    grep = gpu_report()
    print(format_gpu_report(grep, rank), flush=True)

    ok, errors, warnings = check_gpu_isolation(grep)
    _print_findings(rank, warnings, errors)
    reason = "gpu isolation: " + "; ".join(errors) if errors else "gpu isolation check failed"
    abort_if_any_failed(ok, reason, rank)

    # UUIDs are globally unique, so an equal UUID on two ranks means the same
    # physical device even across containers. Every rank gets the same list.
    shared = shared_gpus_across_ranks(grep, True)
    abort_if_any_failed(not shared, "ranks share a GPU: " + "; ".join(shared), rank)


def check_node_features(g):
    rank = g.rank()
    # Disjoint node set per rank (also correct with multiple trainers per
    # machine, unlike partid2nids, which would count nodes twice there).
    all_nodes = dgl.distributed.node_split(
        th.ones(g.num_nodes(), dtype=th.bool), g.get_partition_book(), force_even=True
    )
    st = feature_stats_dist(g.ndata["feat"], all_nodes)
    ok, errors, warnings = check_features(st)   # identical on all ranks

    if rank == 0:
        print(
            f"feat n/dim: {st['n_nodes']}/{st['n_dims']}  "
            f"min/max/mean/std: {st['min']:.3f} {st['max']:.3f} {st['mean']:.3f} {st['std']:.3f}  "
            f"max|col_mean|: {st['max_abs_col_mean']:.3f}  max col_std: {st['max_col_std']:.3f}  "
            f"dead_dims: {st['n_dead_dims']}  nonfinite: {st['n_nonfinite']}",
            flush=True,
        )
        _print_findings(rank, warnings, errors)

    abort_if_any_failed(ok, "bad node features: " + "; ".join(errors), rank)


def run_startup_checks(g, args, database_loaded):
    """Run all checks in a fixed order -- each contains collectives, so
    the order must be identical on all ranks."""
    check_database(args, database_loaded, g.rank())
    check_gpus(g.rank())
    check_node_features(g)
