# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .basefsl import FSLBasedModel


class RelationNet(FSLBasedModel):
    def __init__(self, backbone, args, loss_type='mse'):
        super(RelationNet, self).__init__(backbone, args)
        self.loss_type = loss_type            # 'softmax' or 'mse'
        self.relation_module = RelationModule(self.feat_dim, 128, self.loss_type)
        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y, training=False):
        n_way, n_shot, n_query = self.args.way, self.args.shot, self.args.query

        # feature embedding
        xyz, p_feat, g_feat = self.feature_embedding(x)

        # split support and query features
        y, xyz, p_feat, z_support, z_query = self.split_supp_and_query(y, xyz, p_feat, g_feat)

        # get proto and query
        z_proto = z_support.contiguous().view(n_way, n_shot, -1).mean(1)
        z_query = z_query.contiguous().view(n_way * n_query, -1)

        # get relation
        z_proto_ext = z_proto.view(1, n_way, 1, 1, z_proto.size(1)).repeat(n_query * n_way, 1, 1, 1, 1)
        z_query_ext = z_query.view(1, n_query * n_way, 1, 1, z_proto.size(1)).repeat(n_way, 1, 1, 1, 1)
        z_query_ext = torch.transpose(z_query_ext, 0, 1)
        relation_pairs = torch.cat((z_proto_ext, z_query_ext), 2).view(-1, 2, 1, self.feat_dim)
        relations = self.relation_module(relation_pairs).view(-1, n_way)

        # get loss
        y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
        if self.loss_type == 'mse':
            y_oh = Variable(one_hot(y_query, n_way).cuda())
            total_loss = self.loss_fn(relations, y_oh)
        else:
            total_loss = self.loss_fn(relations, y_query)

        # get acc
        log_p_y = F.log_softmax(relations, dim=1)
        y_hat = log_p_y.max(1)[1].view(n_way, n_query)
        acc_cla = y_hat.eq(y_query.view(n_way, n_query)).float().mean(dim=1)
        acc_val = acc_cla.mean()

        # get acc for each class
        class_unique = torch.unique(y)
        acc_dict = dict(map(lambda x, y: [x, y], class_unique.cpu().numpy(), acc_cla.cpu().numpy()))

        # map y_hat to y
        y_hat = y_hat.cpu().numpy()
        for i in range(n_way):
            y_hat[y_hat == i] = class_unique[i].cpu().numpy()

        return total_loss, acc_val, acc_dict, y_hat


class RelationModule(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size, hidden_size, loss_type='mse'):
        super(RelationModule, self).__init__()
        self.loss_type = loss_type

        self.Conv1 = nn.Sequential(nn.Conv2d(2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.Conv2 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(1),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dp = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.dp(self.fc1(out)))
        if self.loss_type == 'mse':
            out = torch.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)

        return out


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).cuda().scatter_(1, y.unsqueeze(1), 1)

