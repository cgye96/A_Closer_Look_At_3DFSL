# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .basefsl import FSLBasedModel


class PrototypicalNet(FSLBasedModel):
    def __init__(self, backbone, args):
        super(PrototypicalNet, self).__init__(backbone, args)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y, training=False):
        n_way, n_shot, n_query = self.args.way, self.args.shot, self.args.query

        # feature embedding
        xyz, p_feat, g_feat = self.feature_embedding(x)

        # split support and query samples
        y, xyz, p_feat, z_support, z_query = self.split_supp_and_query(y, xyz, p_feat, g_feat)

        # get proto and query
        z_proto = z_support.contiguous().view(n_way, n_shot, -1).mean(1)
        z_query = z_query.contiguous().view(n_way * n_query, -1)

        # get distance
        if self.args.metric == 'Cos':
            dists = F.cosine_similarity(z_query, z_proto)
        elif self.args.metric == 'Euler':
            dists = -euclidean_dist(z_query, z_proto)
        else:
            raise EOFError('Error metric!! Cos or Euler')

        # get loss
        y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()  # creat query label
        total_loss = self.loss_fn(dists, y_query)

        # get acc
        log_p_y = F.log_softmax(dists, dim=1)
        y_hat = log_p_y.max(1)[1].view(n_way, n_query)
        acc_cla = y_hat.eq(y_query.view(n_way, n_query)).float().mean(dim=1)
        acc_avg = acc_cla.mean()

        # get acc for each class
        class_unique = torch.unique(y)
        acc_dict = dict(map(lambda x, y: [x, y], class_unique.cpu().numpy(), acc_cla.cpu().numpy()))

        # map y_hat to y
        y_hat = y_hat.cpu().numpy()
        for i in range(n_way):
            y_hat[y_hat == i] = class_unique[i].cpu().numpy()

        return total_loss, acc_avg, acc_dict, y_hat


def euclidean_dist(x, y):
    """
    input:  x: N x D
            y: M x D
    return: M x N
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)








