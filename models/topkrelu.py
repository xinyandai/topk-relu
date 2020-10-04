from abc import ABC

import torch
from torch import nn
from torch.autograd import Function


def topk_sparsity(x, k, dim=-1):
    assert 0 < k < x.size(dim), "k={} should be within [0 , dim={}]".format(k, x.size(dim))
    topk_matrix = torch.zeros_like(x).scatter_(dim, x.topk(k, dim=dim).indices, x)
    # return topk_matrix.to_sparse()
    return topk_matrix


class TopkReluFunc(Function):
    @staticmethod
    def forward(ctx, x):
        k = max(x.size(-1) // 8, 1)
        x = topk_sparsity(x, k)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, g):
        x = ctx.saved_variables[0]
        k = max(x.size(-1) // 8, 1)
        return g * (x > 0).type(g.type())


class TopkReLU(nn.Module):
    def __init__(self, inplace=False):
        super(TopkReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return TopkReluFunc.apply(x)