from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("edge_cut", "Edge cut"),
    ("edge_cut_ratio", "Normalized edge cut"),
    ("node_balance", "Node balance"),
    ("edge_balance", "Edge balance"),
    ("replication_factor", "Replication factor"),
    ("core_time", "Core time [s]"),
]


def _safe_float(value: str) -> float | None:
    if value == "":
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def load_metric_files(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        edge_cut = data.get("edge_cut")
        num_edges = data.get("num_edges")
        if (
            "edge_cut_ratio" not in data
            and edge_cut is not None
            and num_edges not in (None, 0)
        ):
            data["edge_cut_ratio"] = edge_cut / num_edges
        rows.append({key: str(value) for key, value in data.items()})
    return rows


def plot_metric(rows: list[dict[str, str]], metric_key: str, metric_label: str, output_dir: Path) -> None:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        value = _safe_float(row.get(metric_key, ""))
        if value is None:
            continue
        dataset = row.get("dataset", "unknown")
        k = row.get("k", row.get("num_parts", "?"))
        partitioning_type = row.get("partitioning_type", "unknown")
        group = f"{dataset} | k={k} | {partitioning_type}"
        grouped[group].append((row["algorithm"], value))

    if not grouped:
        return

    for group_name, entries in grouped.items():
        algorithms = [algorithm for algorithm, _ in entries]
        values = [value for _, value in entries]

        plt.figure(figsize=(10, 5))
        plt.bar(algorithms, values, color="#1f6f8b")
        plt.ylabel(metric_label)
        plt.title(group_name)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        safe_group_name = (
            group_name.replace(" | ", "__")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("=", "")
        )
        output_path = output_dir / f"{metric_key}__{safe_group_name}.png"
        plt.savefig(output_path, dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metric plots from direct metrics.json files.")
    parser.add_argument("--metrics-glob", action="append", required=True, help="Glob for direct metrics.json files.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated plots.")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for pattern in args.metrics_glob:
        rows.extend(load_metric_files(sorted(Path(path) for path in glob.glob(pattern, recursive=True))))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, metric_label in METRICS:
        plot_metric(rows, metric_key, metric_label, output_dir)


if __name__ == "__main__":
    main()
