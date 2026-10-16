import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Distributed GAT (DistDGL).")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--ip_config", type=str, help="The file for IP configuration")
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend", type=str, default="gloo", help="pytorch distributed backend"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="the number of GPU device. Use 0 for CPU training",
    )

    # --- Training ------------------------------------------------------------
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="10,25",
        help="one fanout per GAT layer, e.g. '10,25'",
    )
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="stop when val accuracy has not improved for --patience epochs",
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="early-stopping patience in epochs"
    )

    # --- GAT -----------------------------------------------------------------
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="number of GATConv layers; must match the number of fan_out entries",
    )
    parser.add_argument(
        "--num_hidden", type=int, default=8, help="hidden units per head"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="heads in hidden layers"
    )
    parser.add_argument(
        "--num_out_heads", type=int, default=1, help="heads in the output layer"
    )
    parser.add_argument(
        "--feat_drop", type=float, default=0.6, help="input feature dropout"
    )
    parser.add_argument(
        "--attn_drop", type=float, default=0.6, help="attention dropout"
    )
    parser.add_argument(
        "--negative_slope", type=float, default=0.2, help="negative slope of leaky relu"
    )
    parser.add_argument(
        "--residual", action="store_true", help="use residual connections"
    )

    # --- DistDGL -------------------------------------------------------------
    parser.add_argument("--local_rank", type=int, help="get rank of the process")
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
        "of batches to be the same.",
    )
    parser.add_argument(
        "--use_graphbolt",
        action="store_true",
        help="Use GraphBolt for distributed train.",
    )

    # --- Logging -------------------------------------------------------------
    parser.add_argument(
        "--data-dirpath",
        type=str,
        help="Path to Directory to save data(like epoch metrics or system metrics).",
    )
    parser.add_argument("--send-data-to-database", action="store_true")
    parser.add_argument("--save-data-to-csv", action="store_true")
    parser.add_argument("--save-system-metrics-to-csv", action="store_true")
    parser.add_argument("--save-not-only-rank0", action="store_true")

    args = parser.parse_args()

    # Same on every rank, so raising here (before any collective) is safe.
    n_fanout = len(args.fan_out.split(","))
    if n_fanout != args.num_layers:
        parser.error(
            f"--fan_out has {n_fanout} entries but --num_layers is {args.num_layers}"
        )
    if args.num_layers < 2:
        parser.error("--num_layers must be >= 2")
    return args
