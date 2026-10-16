"""GAT for DistDGL: mini-batch training on sampled blocks and layer-wise
full-neighbor inference, same interface as DistSAGE (forward(blocks, x),
inference(g, x, batch_size, device)), so evaluation.evaluate works unchanged."""

import torch as th
import torch.nn as nn

import dgl
import dgl.distributed
import dgl.nn.pytorch as dglnn


def with_self_loops(block):
    """Give every dst node of a block exactly one self loop.

    GATConv refuses dst nodes with zero in-degree, and sampled blocks have
    them (nodes without neighbours). The graph itself cannot be changed at
    runtime in DistDGL, so the loop is added per block. In a block the dst
    nodes are the first num_dst_nodes src nodes, so a self loop is the edge
    i -> i in local IDs. Existing self loops (if the graph was partitioned
    with them) are dropped first, so there is never a duplicate.
    """
    src, dst = block.edges()
    keep = src != dst
    n_dst = block.num_dst_nodes()
    loop = th.arange(n_dst, dtype=src.dtype, device=src.device)
    return dgl.create_block(
        (th.cat([src[keep], loop]), th.cat([dst[keep], loop])),
        num_src_nodes=block.num_src_nodes(),
        num_dst_nodes=n_dst,
        idtype=block.idtype,
        device=block.device,
    )


class DistGAT(nn.Module):
    """
    Parameters
    ----------
    in_feats : int
    n_hidden : int        hidden units per head
    n_classes : int
    n_layers : int        number of GATConv layers (= number of fan_out entries)
    heads : list[int]     heads per layer, len == n_layers
    activation : callable
    feat_drop, attn_drop, negative_slope : float
    residual : bool
    """

    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("DistGAT needs at least 2 layers")
        if len(heads) != n_layers:
            raise ValueError(f"len(heads)={len(heads)} != n_layers={n_layers}")

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.heads = heads

        self.layers = nn.ModuleList()
        # input layer (no residual, like the DGL reference)
        self.layers.append(
            dglnn.GATConv(
                in_feats,
                n_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                activation,
            )
        )
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                dglnn.GATConv(
                    n_hidden * heads[i - 1],
                    n_hidden,
                    heads[i],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    activation,
                )
            )
        # output layer
        self.layers.append(
            dglnn.GATConv(
                n_hidden * heads[-2],
                n_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def _apply_layer(self, i, block, h):
        h = self.layers[i](with_self_loops(block), h)
        if i == self.n_layers - 1:
            return h.mean(1)  # average output heads
        return h.flatten(1)  # concatenate hidden heads

    def forward(self, blocks, x):
        h = x
        for i, block in enumerate(blocks):
            h = self._apply_layer(i, block, h)
        return h

    def inference(self, g, x, batch_size, device):
        """Distributed layer-wise inference on full neighbourhoods."""
        nodes = dgl.distributed.node_split(
            th.ones(g.num_nodes(), dtype=th.bool),
            g.get_partition_book(),
            force_even=True,
        )
        sampler = dgl.dataloading.NeighborSampler([-1])

        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                out_dim = self.n_classes
            else:
                out_dim = self.n_hidden * self.heads[i]
            # One name per layer: layer i reads layer i-1's tensor while
            # writing its own, so they must not alias.
            y = dgl.distributed.DistTensor(
                (g.num_nodes(), out_dim),
                th.float32,
                f"gat_infer_h{i}",
                persistent=True,
            )
            print(f"|V|={g.num_nodes()}, inference layer {i}, batch size: {batch_size}")

            dataloader = dgl.distributed.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h = self._apply_layer(i, block, h)
                y[output_nodes] = h.cpu()  # DistTensor lives on CPU

            x = y
            g.barrier()
        return x
