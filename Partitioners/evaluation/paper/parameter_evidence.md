# Parameter evidence

The table prioritizes the execution commands supplied in the chat over the SQL snapshot. Code defaults refer to this local checkout, not a verified historical container image. Paths below are relative to the Partitioners repository.

- SIGMA node: supplied command uses `mod_fennel`, `light_plus`, `clustre`; SQL instead records `fennel_mult_obj`, `strong`, `clustre`.
- KaHIP: supplied command uses `strong`; SQL records `eco`. `NODE/KaHIP/app/configuration.h`, `configuration::strong` calls `standard`, with seed 0 and imbalance 3 for normal graph partitioning.
- LDG: `NODE/LDG/run_partition.py`, `parse_args` and `LDGPartitioner`: slack 0; integer capacity max(floor(n/k), 1); random tie breaking; no seed initialization.
- FENNEL: `NODE/FENNEL-Streaming-Graph-Partitioning/codes/fennel.cpp`: alpha 0.5, gammaPower 1.5, srand(42), random_shuffle. These are fixed implementation values, not the graph-dependent alpha formula sometimes used in other FENNEL implementations. Dockerfile compiles this file.
- Random: `BOTH/RandomPartitioner/run_partition.py`: torch.randint for uniform assignment; absent --seed, a seed is generated at runtime. Original command not supplied.
- CatGNN: `NODE/SDT-GNN/run_partitioner.py`: CatGNN output-path example, METHOD=SPRING, seed 42, K=1. `SDT_GNN/Partitioning.py` passes stream_iters=1 to SPRING. Original command not supplied; local wrapper mapping should be confirmed against the run. T is only relevant to automatic partition-count selection and is omitted for fixed k.
- PuLP: `NODE/PuLP/install` builds PuLP 0.2 and XtraPuLP 0.3. Table explicitly uses the shared-memory PuLP 0.2 defaults from `pulp/0.2/pulp_main.cpp`: vertex balance 1.10, BFS initialization, edge/max-cut balancing off, one run. Original executable and invocation require confirmation; XtraPuLP defaults must not be substituted silently.
- DBH: supplied executable matches the dbh target in `EDGE/2ps/CMakeLists.txt`. `EDGE/2ps/dbh/main.cpp` exposes no additional algorithm-specific tuning options beyond partition count; memsize defaults to 100 MB. Do not confuse this with the different DBH implementation under split-merge-partitioner.
- 2PS: `EDGE/2ps/main.cpp`: balance_ratio 1.05, str_iters 1, use_hdrf false. Supplied command selects streamcom.
- HDRF, HeiStream(E), BuffCut, HEP, Cuttana, METIS: explicit parameters transcribed from user-supplied commands. METIS wrapper calls metis.part_graph with nparts and objtype; no extra library defaults are guessed.
- FSM discrepancy: `EDGE/split-merge-partitioner/src/main.cpp`: method defaults to hep; split factor (-k) defaults to 1; FsmPartitioner is instantiated only for a method starting with fsm and split factor >1. The supplied command with only -p and -write onefile selects HEP in this checkout. No replacement FSM method or split factor has been invented.
