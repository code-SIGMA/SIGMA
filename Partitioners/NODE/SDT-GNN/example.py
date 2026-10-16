import os
import SDT_GNN

"""
An example to run SDT_GNN.

In this example, we partition the 'cora' dataset into 4 partitions using SPRING,
and train a 2-layer GraphSAGE model with a sampling size of [25, 10] using SDT_GNN.

Different streaming partitioning algorithms, such as SPRING, DBH, PowerGraph, HDRF, and 2PSL, can be used to partition a graph.
Different GNN models, such as GCN, GAT, and GraphSAGE, can be trained by SDT-GNN in a distributed manner.
All the hyperparameters are tunable.

Please refer to our paper for more details.
"""

if __name__ == "__main__":

    path = os.path.abspath(os.getcwd())

    dataset = 'cora'
    ### Please change the dataset name and run the following function to download and process other datasets
    ## SDT_GNN.data.preprocess.process_dataset(dataset, path + '/datasets/')
    print('='*60)
    print('dataset: ', dataset)


    ##### Streaming Partitioning #####
    number_partition = 4
    method = 'SPRING'
    ### Please change the method name for other streaming partitioning algorithms
    print('method: ', method)

    partitioner = SDT_GNN.Partitioning(dataset = dataset,
                                    multilabel = False,
                                    number_partition = number_partition,
                                    path = path + '/datasets/',
                                    method = method,
                                    output_path = path + '/output/' + 'partitions-' + str(number_partition) + '/' + dataset + '/' + method + '/',
                                    partition_features_file = True,
                                    print_partition_statistics = True,
                                    save_dgl_graph = True)
    partitioner.run()

    ##### GNN Model Training #####
    ### Please change the model name and correspoding hyperparameters for other GNN models
    gnn_model = SDT_GNN.GNN(dataset = dataset, multilabel = False,
                            path = path + '/datasets/',
                            output_path = path + '/output/' + 'partitions-' + str(number_partition) + '/' + dataset + '/' + method + '/',
                            number_partition = number_partition,
                            model='GraphSAGE',
                            n_hidden= 256,
                            n_layers = 2,
                            fanout = [25,10],
                            dropout = 0.0,
                            aggregator = 'mean',
                            activation = 'relu',
                            epochs = 100,
                            batch_size = 512,
                            epochs_eval = 1,
                            epochs_avg = 1,
                            optimizer = 'adam',
                            lr = 0.01)
    gnn_model.run()
