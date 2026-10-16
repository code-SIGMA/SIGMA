# Evaluation Pipeline

1. Each partitioner generates a `metrics.json` directly during its normal run.
2. `evaluation/plot_metrics.py` later collects these files and generates plots from them.
3. `evaluation/metrics.py` serves as a fallback or validation tool if a partitioner does not write a metrics file directly.

## Expected Metrics Schema

- Node partitioners:
  `algorithm`, `dataset`, `k`, `partitioning_type`, `num_parts`, `num_nodes`,
  `num_edges`, `edge_cut`, `edge_cut_ratio`, `node_balance`, `edge_balance`,
  `core_time`
- Edge partitioners:
  `algorithm`, `dataset`, `k`, `partitioning_type`, `num_parts`, `num_nodes`,
  `num_edges`, `node_balance`, `edge_balance`, `replication_factor`, `core_time`

## Semantics of `core_time`

`core_time` refers repository-wide to the elapsed wall-clock time from entering
`main` until immediately before the final write of `metrics.json`, or
immediately before returning from `main` if no metrics file is written.

The time required to physically write this final metrics value itself cannot,
by definition, be included in the same value.

## Direct Workflow

1. Run the partitioner normally.
   Expected result: exactly one `metrics.json` is produced per run.
2. Afterwards, plot all metrics files together:

```bash
python3 evaluation/plot_metrics.py \
  --metrics-glob "**/metrics.json" \
  --output-dir evaluation/results/plots