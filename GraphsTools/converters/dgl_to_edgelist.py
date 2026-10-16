import argparse

def main(args):

    print("Import dgl...")
    import dgl
    from dgl.data.utils import load_graphs

    print("Load Graph...")
    graphs, _ = load_graphs(args.graph_path)
    graph = graphs[0]


    print("Convert DGL graph to Edgelist...")

    #===============================
    print("Ensure graph is on CPU...")
    graph = graph.cpu()

    src, dst = graph.edges()

    print(f"Write Edgelist file({args.save_path})...")
    with open(args.save_path, "w") as f:
        for u, v in zip(src.tolist(), dst.tolist()):
            f.write(f"{u} {v}\n")

    #===============================
    print("Finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--graph-path" , type=str, help = "The Path to the dgl graph file.", required=True)

    parser.add_argument("-s", "--save-path" , type=str, help = "The Path to save the edgelist file.", required=True)

    args = parser.parse_args()

    main(args)
