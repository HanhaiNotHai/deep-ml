import torch


def layer_norm(x, gamma, beta, eps=1e-5):
    '''normalize over the last dim, then affine-transform with gamma and beta'''

    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * gamma + beta
