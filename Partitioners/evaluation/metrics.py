from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_nonempty_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _extract_count_ranges(ranges_obj: Any) -> list[int]:
    if isinstance(ranges_obj, dict):
        values = next(iter(ranges_obj.values()))
    else:
        values = ranges_obj

    counts: list[int] = []
    for value in values:
        if isinstance(value, list) and len(value) == 2:
            counts.append(int(value[1]) - int(value[0]))
        else:
            counts.append(int(value))
    return counts


def compute_node_balance(counts: list[int], total_nodes: int, num_parts: int) -> float:
    if total_nodes == 0 or num_parts == 0:
        return math.nan
    return max(counts, default=0) / (total_nodes / num_parts)


def compute_edge_balance(counts: list[int], total_edges: int, num_parts: int) -> float:
    if total_edges == 0 or num_parts == 0:
        return math.nan
    return max(counts, default=0) / (total_edges / num_parts)


def compute_edge_cut_ratio(edge_cut: int | float, total_edges: int) -> float:
    if total_edges == 0:
        return math.nan
    return edge_cut / total_edges


def compute_partition_set_balance(counts: list[int]) -> float:
    if not counts:
        return math.nan
    total = sum(counts)
    num_parts = len(counts)
    if total == 0 or num_parts == 0:
        return math.nan
    return max(counts, default=0) / (total / num_parts)


def evaluate_edge_triples(path: Path) -> dict[str, Any]:
    edge_counts: dict[int, int] = {}
    partition_nodes: dict[int, set[int]] = {}
    all_nodes: set[int] = set()
    total_edges = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                raise ValueError(f"Invalid edge-triple line in {path}: {line}")
            src, dst, part = int(parts[0]), int(parts[1]), int(parts[2])
            edge_counts[part] = edge_counts.get(part, 0) + 1
            partition_nodes.setdefault(part, set()).update((src, dst))
            all_nodes.update((src, dst))
            total_edges += 1

    num_parts = len(edge_counts)
    replicated_vertices = sum(len(nodes) for nodes in partition_nodes.values())
    replication_factor = (
        replicated_vertices / len(all_nodes) if all_nodes else math.nan
    )

    return {
        "partitioning_type": "edge",
        "num_parts": num_parts,
        "num_nodes": len(all_nodes),
        "num_edges": total_edges,
        "node_balance": compute_partition_set_balance([len(nodes) for nodes in partition_nodes.values()]),
        "edge_balance": compute_edge_balance(list(edge_counts.values()), total_edges, num_parts),
        "replication_factor": replication_factor,
    }


def evaluate_2ps_base_dir(path: Path) -> dict[str, Any]:
    part_dirs = sorted(
        [child for child in path.iterdir() if child.is_dir() and child.name.startswith("part")]
    )
    edge_counts: list[int] = []
    node_counts: list[int] = []
    total_nodes = set()

    for part_dir in part_dirs:
        nodes_file = part_dir / "nodes.txt"
        node_lines = _read_nonempty_lines(nodes_file)
        node_counts.append(len(node_lines))
        total_nodes.update(int(node) for node in node_lines)

        bin_candidates = sorted(part_dir.glob("partition_*.bin"))
        if not bin_candidates:
            raise FileNotFoundError(f"No binary partition file found in {part_dir}")
        bin_path = bin_candidates[0]
        with bin_path.open("rb") as handle:
            edge_count = int.from_bytes(handle.read(8), byteorder="little", signed=False)
        edge_counts.append(edge_count)

    num_parts = len(part_dirs)
    total_edge_count = sum(edge_counts)
    total_vertex_replicas = sum(node_counts)

    return {
        "partitioning_type": "edge",
        "num_parts": num_parts,
        "num_nodes": len(total_nodes),
        "num_edges": total_edge_count,
        "node_balance": compute_partition_set_balance(node_counts),
        "edge_balance": compute_edge_balance(edge_counts, total_edge_count, num_parts),
        "replication_factor": (
            total_vertex_replicas / len(total_nodes) if total_nodes else math.nan
        ),
    }


def evaluate_node_lines(path: Path) -> dict[str, Any]:
    assignments = [int(line) for line in _read_nonempty_lines(path)]
    counts: dict[int, int] = {}
    for assignment in assignments:
        counts[assignment] = counts.get(assignment, 0) + 1

    num_parts = len(counts)
    return {
        "partitioning_type": "node",
        "num_parts": num_parts,
        "num_nodes": len(assignments),
        "node_balance": compute_node_balance(list(counts.values()), len(assignments), num_parts),
    }


def evaluate_node_csv(path: Path) -> dict[str, Any]:
    counts: dict[int, int] = {}
    total_nodes = 0
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"Invalid node-csv row in {path}: {row}")
            part = int(row[1])
            counts[part] = counts.get(part, 0) + 1
            total_nodes += 1

    num_parts = len(counts)
    return {
        "partitioning_type": "node",
        "num_parts": num_parts,
        "num_nodes": total_nodes,
        "node_balance": compute_node_balance(list(counts.values()), total_nodes, num_parts),
    }


def evaluate_dgl_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    num_parts = int(metadata["num_parts"])
    num_nodes = int(metadata["num_nodes"])
    num_edges = int(metadata["num_edges"])
    node_counts = _extract_count_ranges(metadata["node_map"])
    edge_counts = _extract_count_ranges(metadata["edge_map"])
    part_method = str(metadata.get("part_method", "")).lower()

    if "edge" in part_method:
        result = {
            "partitioning_type": "edge",
            "num_parts": num_parts,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "node_balance": compute_partition_set_balance(node_counts),
            "edge_balance": compute_edge_balance(edge_counts, num_edges, num_parts),
            "replication_factor": sum(node_counts) / num_nodes if num_nodes else math.nan,
        }
    else:
        result = {
            "partitioning_type": "node",
            "num_parts": num_parts,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "node_balance": compute_node_balance(node_counts, num_nodes, num_parts),
            "edge_balance": compute_edge_balance(edge_counts, num_edges, num_parts),
        }

    return result


FORMAT_EVALUATORS = {
    "edge-triples": evaluate_edge_triples,
    "2ps-base-dir": evaluate_2ps_base_dir,
    "node-lines": evaluate_node_lines,
    "node-csv": evaluate_node_csv,
    "dgl-json": evaluate_dgl_json,
}


def add_derived_metrics(result: dict[str, Any]) -> dict[str, Any]:
    if "edge_cut" in result and "num_edges" in result:
        result["edge_cut_ratio"] = compute_edge_cut_ratio(
            result["edge_cut"], result["num_edges"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate graph partition metrics.")
    parser.add_argument(
        "--format",
        choices=sorted(FORMAT_EVALUATORS),
        required=True,
        help="Partition output format to evaluate.",
    )
    parser.add_argument("--path", required=True, help="Path to the partition output.")
    parser.add_argument("--output", help="Optional JSON file for the computed metrics.")
    args = parser.parse_args()

    evaluator = FORMAT_EVALUATORS[args.format]
    result = add_derived_metrics(evaluator(Path(args.path)))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
