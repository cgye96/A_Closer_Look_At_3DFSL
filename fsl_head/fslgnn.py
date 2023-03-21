# -*- coding: utf-8 -*-
# refer：https://github.com/vgsatorras/few-shot-gnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .basefsl import FSLBasedModel


class FSL_GNN(FSLBasedModel):
    def __init__(self, backbone, args):
        super(FSL_GNN, self).__init__(backbone, args)
        self.loss_fn = nn.CrossEntropyLoss()
        input_features = self.feat_dim + self.args.way
        self.gnn_obj = GNN_nl(args=args, input_features=input_features, nf=128, J=2)

    def forward(self, x, y, training=False):
        n_way, n_shot, n_query = self.args.way, self.args.shot, self.args.query

        # feature embedding
        xyz, p_feat, g_feat = self.feature_embedding(x)

        # split support and query features
        y, xyz, p_feat, z_support, z_query = self.split_supp_and_query(y, xyz, p_feat, g_feat)

        # get proto and query
        z_proto = z_support.contiguous().view(n_way, n_shot, -1).mean(1)
        z_query = z_query.contiguous().view(n_way * n_query, -1)

        # FSL_GNN
        # We build a graph with (way*shot)+1 nodes for each query sample, and there are (way*query) graphs in total.
        zero_pad = torch.zeros(n_way * n_query, n_way).cuda()
        labels = torch.eye(n_way, requires_grad=False).cuda()
        labels = labels.view(n_way, 1, n_way).repeat(1, 1, 1).view(-1, n_way).cuda()  # shot = 1

        q_node = torch.cat((z_query, zero_pad), 1).unsqueeze(0)
        s_node = torch.cat((z_proto, labels), 1).view(n_way*1, 1, -1).repeat(1, n_way*n_query, 1)
        nodes = torch.cat((q_node, s_node), 0).transpose(1, 0)

        logits = self.gnn_obj(nodes).squeeze(-1)
        outputs = torch.sigmoid(logits)

        # get loss
        y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
        total_loss = self.loss_fn(logits, y_query)

        # get acc
        log_p_y = F.log_softmax(logits, dim=1)
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


class GNN_nl(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J
        self.num_layers = 4

        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.way, 2, bn_bool=False)

    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)).cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)

            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]


def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3)  # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x)        # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2)   # output has size (bs, N, J*num_features)
    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.dropout = nn.Dropout(0.5)
        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.dropout(x)
        x = self.fc(x)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x


class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1], num_operators=1):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        # self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        # self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        # self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        # self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        # self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        # self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf*2, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, x, W_id):
        W1 = x.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2)
        W_new = torch.abs(W1 - W2)
        W_new = torch.transpose(W_new, 1, 3)

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)

        # W_new = self.conv2d_2(W_new)
        # W_new = self.bn_2(W_new)
        # W_new = F.leaky_relu(W_new)

        # W_new = self.conv2d_3(W_new)
        # W_new = self.bn_3(W_new)
        # W_new = F.leaky_relu(W_new)
        #
        # W_new = self.conv2d_4(W_new)
        # W_new = self.bn_4(W_new)
        # W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3)

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new, dim=1)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)

        elif self.activation == 'sigmoid':
            W_new = torch.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise(NotImplementedError)

        return W_new

