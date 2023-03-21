# -*- coding:utf-8 -*-
################
# PointCNN model
# reference: https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/points/point_cnn.py
# pytorch 1.6.0  pip=21.0  cuda 10.1
# need install torch_geometric
# env: dl10 42    --16 243--
# bug: only run in cuda:0
# python: >=3.6.1
################

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import XConv, fps, global_mean_pool


class PointCNN(torch.nn.Module):
    def __init__(self, emb_dims):
        super(PointCNN, self).__init__()
        self.emb_dims = emb_dims
        self.conv1 = XConv(0, 48, dim=3, kernel_size=8, hidden_channels=32)
        self.conv2 = XConv(
            48, 96, dim=3, kernel_size=12, hidden_channels=64, dilation=2)
        self.conv3 = XConv(
            96, 192, dim=3, kernel_size=16, hidden_channels=128, dilation=2)
        self.conv4 = XConv(
            192, 384, dim=3, kernel_size=16, hidden_channels=256, dilation=2)

        self.lin1 = Lin(384, self.emb_dims)

    def forward(self, pos, batch):
        '''
        :param pos:  coords of input point, [B*N 3]
        :param batch: batch index of each input point, [B*N]
        :return:
        '''
        xyz = pos
        x = F.relu(self.conv1(None, pos, batch))

        idx = fps(pos, batch, ratio=0.375)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv2(x, pos, batch))

        idx = fps(pos, batch, ratio=0.334)
        x, pos, batch = x[idx], pos[idx], batch[idx]

        x = F.relu(self.conv3(x, pos, batch))
        x = F.relu(self.conv4(x, pos, batch))

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))

        return xyz, x


