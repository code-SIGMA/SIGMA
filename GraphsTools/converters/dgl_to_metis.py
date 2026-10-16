import argparse
import json

def main(args):

    print("Import dgl...")
    import dgl
    from dgl.data.utils import load_graphs
    print("Import torch...")
    import torch

    print("Load Graph...")
    graphs, _ = load_graphs(args.graph_path)
    graph = graphs[0]


    print("Convert DGL graph to Metis...")

    #====================================
    print("Ensure graph is on CPU...")
    graph = graph.cpu()

    num_nodes = graph.num_nodes()

    # Get edge list
    src, dst = graph.edges()

    # Build adjacency list (1-based indexing for METIS)
    print("Build adjacency list...")
    adj = {i: set() for i in range(1, num_nodes + 1)}

    for u, v in zip(src.tolist(), dst.tolist()):
        u += 1
        v += 1
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    # Count undirected edges
    print("Count undirected edges...")
    num_edges = sum(len(neighbors) for neighbors in adj.values()) // 2

    print(f"Write METIS file({args.save_path})...")
    with open(args.save_path, "w") as f:
        f.write(f"{num_nodes} {num_edges}\n")
        for i in range(1, num_nodes + 1):
            neighbors = sorted(adj[i])
            f.write(" ".join(map(str, neighbors)) + "\n")
    #====================================

    print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--graph-path" , type=str, help = "The Path to the dgl graph file.", required=True)

    parser.add_argument("-s", "--save-path" , type=str, help = "The Path to save the metis graph file.", required=True)

    args = parser.parse_args()

    main(args)
