"""Load the environment, set up distributed execution, select devices and the metrics directory."""
import socket
from pathlib import Path

import dgl
import dgl.distributed
import torch as th

from database import Database
from env_loader import EnvLoader


def load_environment(env_path="/root/env.json"):
    """Read run_id and establish the database connection from env.json.

    Returns (run_id, database). `database` may be falsy if no
    connection could be established.
    """
    loader = EnvLoader(env_path)

    run_id = loader.group("general", required=["run_id"]).get("run_id")
    if run_id is None:
        raise KeyError(f"Run id is not given via {env_path}")
    print(f"Run ID: {run_id}", flush=True)

    db_env = loader.group(
        "database",
        required=["db_host", "db_port", "db_name", "db_user", "db_password"],
    )
    database = db_env.build(Database)
    return run_id, database


def init_distributed(args):
    """Initialize DistDGL, the PyTorch process group and the DistGraph."""
    host_name = socket.gethostname()
    print(f"{host_name}: Initializing DistDGL.")
    dgl.distributed.initialize(args.ip_config, use_graphbolt=args.use_graphbolt)
    print(f"{host_name}: Initializing PyTorch process group.")
    th.distributed.init_process_group(backend=args.backend)
    print(f"{host_name}: Initializing DistGraph.")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print(f"Rank of {host_name}: {g.rank()}")
    return g


def pick_device(rank, num_gpus):
    if num_gpus == 0:
        return th.device("cpu")
    return th.device(f"cuda:{rank % num_gpus}")


def prepare_metrics_dir(args, rank):
    """Create the metrics directory on ranks that write output."""
    data_dirpath = Path(args.data_dirpath) if args.data_dirpath else Path("./metrics")
    if rank == 0 or args.save_not_only_rank0:
        data_dirpath.mkdir(parents=True, exist_ok=True)
        print(f"Save metrics to {data_dirpath.as_posix()}", flush=True)
    return data_dirpath
