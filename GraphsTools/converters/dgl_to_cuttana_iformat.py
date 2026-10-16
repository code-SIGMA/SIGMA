import argparse

def main(args):
    print("Import DGL...")
    import dgl
    from dgl.data.utils import load_graphs

    print("Loading graph...")
    graphs, _ = load_graphs(args.graph_path)
    graph = graphs[0]

    print("Move to CPU...")
    graph = graph.cpu()

    num_nodes = graph.num_nodes()

    print("Extract edges...")
    src, dst = graph.edges()

    # Build adjacency list
    adj = {i: set() for i in range(num_nodes)}

    print("Building adjacency list...")
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].add(v)
        adj[v].add(u)   # ensure undirected safety

    # Compute edge count (undirected)
    edge_count = sum(len(nei) for nei in adj.values()) // 2

    print(f"Writing output to {args.save_path}...")

    with open(args.save_path, "w") as f:
        # header
        f.write(f"{num_nodes} {edge_count}\n")

        # IMPORTANT: convert to 1-based indexing
        for i in range(num_nodes):
            neighbors = sorted(adj[i])
            neighbors_1based = [v + 1 for v in neighbors]

            f.write(f"{i + 1} {len(neighbors_1based)} ")
            f.write(" ".join(map(str, neighbors_1based)))
            f.write("\n")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g", "--graph-path",
        type=str,
        required=True,
        help="Path to DGL graph file"
    )

    parser.add_argument(
        "-s", "--save-path",
        type=str,
        required=True,
        help="Output METIS-like file for C++ pipeline(cuttana)"
    )

    args = parser.parse_args()
    main(args)
