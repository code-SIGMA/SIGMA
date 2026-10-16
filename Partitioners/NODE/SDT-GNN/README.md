# Streaming-based Distributed Training Framework for Graph Neural Networks

## Overview

This repository contains the implementation of SDT-GNN, a streaming-based distributed training framework for graph neural networks (GNNs).


## Requirements

<!--PyTorch v2.0.1-->
<!--DGL=1.1.0-->
<!--CUDA=11.8-->

[![](https://img.shields.io/badge/PyTorch-2.0.1-blueviolet)](https://pytorch.org/get-started/)
[![](https://img.shields.io/badge/DGL-1.1.0-blue)](https://www.dgl.ai/pages/start.html)
[![](https://img.shields.io/badge/CUDA-11.8-green)](https://developer.nvidia.com/cuda-11-8-0-download-archive)

GPU versions of [PyTorch](https://pytorch.org/get-started/) and [DGL](https://www.dgl.ai/pages/start.html) are required to run SDT-GNN. Please check the corresponding official websites for installation.

## Installation

We recommend using the Conda virtual environment

```bash
$ conda env create -f artifact.yml
```

The installation of conda can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Framework

<p align="center">
  <img src="https://github.com/xhuang2016/SDT-GNN/blob/main/Framework.png" alt="Framework" width="600">
</p>

## Training workflow

<p align="center">
  <img src="https://github.com/xhuang2016/SDT-GNN/blob/main/trainingworkflow1.png" alt="# partitions is equal to # worker" width="600">
</p>
<p align="center"><b> (a) # partitions is equal to # workers</b></p>

<p align="center">
  <img src="https://github.com/xhuang2016/SDT-GNN/blob/main/trainingworkflow2.png" alt="# partitions is greater than # worker" width="900">
</p>
<p align="center"><b> (b) # partitions is greater than # workers</b></p>



## Dataset
We use 16 real-world graph datasets that are downloaded from [DGL](https://www.dgl.ai/) and [OGB](https://ogb.stanford.edu/), and converted to the formats used by our Framework.
> 1. edge_list.csv: edge list of the graph.
> 2. feats.npy: node features (NumPy array).
> 3. class_map.json: node labels (dictionary).
> 4. role.json: role of Train/Val/Test split (dictionary).


## Running the code

We provide an example in 'example.py'. Follow the command below to run the code.

```bash
$ python3 example.py
```

Note:
> - Different streaming partitioning algorithms, such as SPRING, DBH, PowerGraph, HDRF, and 2PSL, can be used to partition a graph.
> - Different GNN models, such as GCN, GAT, and GraphSAGE, can be trained by SDT-GNN in a distributed manner.
> - All the hyperparameters are tunable.  
> - Please refer to our paper for more details.  
