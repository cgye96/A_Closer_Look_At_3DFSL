# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    # device = torch.device('cuda')
    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # ()
    return feature


class Econv(nn.Module):
    def __init__(self, k=20, In_dims=6, Out_dims=64):
        super(Econv, self).__init__()
        self.k = k
        self.Conv = nn.Sequential(nn.Conv2d(In_dims, Out_dims, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(Out_dims),
                                  nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.Conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024):
        """ :param k:  Number of KNN points
            :param emb_dims: Dimension of embeddings
        """
        super(DGCNN, self).__init__()
        self.emb_dims = emb_dims
        self.Econv1 = Econv(k=k, In_dims=3*2, Out_dims=64)
        self.Econv2 = Econv(k=k, In_dims=64*2, Out_dims=64)
        self.Econv3 = Econv(k=k, In_dims=64*2, Out_dims=128)
        self.Econv4 = Econv(k=k, In_dims=128*2, Out_dims=256)
        self.MLP = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(emb_dims),
                                 nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        """ :param x:  input point cloud: b,n,3
        """
        x = x.transpose(1, 2)
        xyz = x
        x1 = self.Econv1(x)
        x2 = self.Econv2(x1)
        x3 = self.Econv3(x2)
        x4 = self.Econv4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.MLP(x)
        g_feature = x.max(-1)[0]
        return xyz, x, g_feature

