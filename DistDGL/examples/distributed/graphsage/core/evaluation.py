import torch as th

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted labels.
    labels : torch.Tensor
        Ground-truth labels.

    Returns
    -------
    float
        Accuracy.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, nid_dict, batch_size, device):
    """Return {split: accuracy_in_percent} globally across all ranks."""
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()

    res = {}
    for k, nid in nid_dict.items():
        p = pred[nid]
        y = labels[nid].long()
        c = th.tensor(
            [float((th.argmax(p, dim=1) == y).sum()), float(len(nid))],
            dtype=th.float64,
        )
        th.distributed.all_reduce(c, op=th.distributed.ReduceOp.SUM)
        res[k] = c[0].item() / max(c[1].item(), 1.0) * 100
    return res
