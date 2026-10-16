import argparse

def main(args):

    print("Import dgl...")
    import dgl
    from dgl.data.utils import load_graphs

    print("Load Graph...")
    graphs, _ = load_graphs(args.graph_path)
    graph = graphs[0]

    #====================================
    print("Ensure graph is on CPU...")
    graph = graph.cpu()

    num_nodes = graph.num_nodes()

    print("Get edges...")
    src, dst = graph.edges()

    print("Build adjacency list...")
    adj = {i: set() for i in range(num_nodes)}

    for u, v in zip(src.tolist(), dst.tolist()):
#        if u != v:
        adj[u].add(v)
#            adj[v].add(u)

    #====================================

    print(f"Write adjacency format file ({args.save_path})...")
    with open(args.save_path, "w") as f:
        f.write(f"{num_nodes}\n")

        for i in range(num_nodes):
            neighbors = sorted(adj[i])

            neighbors_1based = [v + 1 for v in neighbors]

            f.write(f"{len(neighbors_1based)} ")
            f.write(" ".join(map(str, neighbors_1based)))
            f.write("\n")

    print("Finish.")


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
        help="Output file path"
    )

    args = parser.parse_args()
    main(args)
