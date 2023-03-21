# -*- coding: utf-8 -*-
# This code is modified from:
# https://github.com/facebookresearch/low-shot-shrink-hallucinate
# https://github.com/dragen1860/MAML-Pytorch
# https://github.com/katerakelly/pytorch-maml

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.data_utils import rotate_point_cloud,jitter_point_cloud


class MAML(nn.Module):
    def __init__(self, args, approx=False):
        super(MAML, self).__init__()
        self.args = args
        if args.backbone == 'DGCNN':
            self.embedding = MAMLdgcnnBackbone(Out_dims=args.emb_dims)
        elif args.backbone == 'PointNet':
            self.embedding = MAMLPointNetBackbone(Out_dims=args.emb_dims)
        self.classifier = MAMLClassifier(input_dim=args.emb_dims, nclass=5)
        self.loss_fn = nn.CrossEntropyLoss()

        self.n_task = args.n_task
        self.task_update_num = args.n_step
        self.train_lr = args.lr_f
        self.approx = approx   # first order approx.

    def forward(self, x, y, training=True):
        n_way, n_shot, n_query = self.args.way, self.args.shot, self.args.query

        # split support and query examples
        cl = torch.unique(y)
        s_idx = torch.stack(list(map(lambda c: y.eq(c).nonzero()[:n_shot], cl))).view(-1)
        q_idx = torch.stack(list(map(lambda c: y.eq(c).nonzero()[n_shot:], cl))).view(-1)
        support_x, query_x = x[s_idx], x[q_idx]
        support_y = torch.from_numpy(np.repeat(range(n_way), n_shot)).cuda()
        query_y = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()

        # the first gradient calculation is based on original weight
        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            # support features extracting
            s_xyz, s_p_fea, z_support = self.embedding(support_x)

            scores = self.classifier(z_support)
            set_loss = self.loss_fn(scores, support_y)

            # build full graph support gradient of gradient
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True, allow_unused=True)
            if self.approx:
                # do not calculate gradient of gradient if using first order approximation
                grad = [g.detach() for g in grad ]
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in pointnet.py
                if grad[k] is not None:
                    if weight.fast is None:
                        # create weight.fast
                        weight.fast = weight - self.train_lr * grad[k]
                    else:
                        # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                        weight.fast = weight.fast - self.train_lr * grad[k]
                else:
                    if weight.fast is None:
                        weight.fast = weight
                    else:
                        weight.fast = weight.fast
                # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                fast_parameters.append(weight.fast)

        # get loss
        if training:
            # we update the gradients about the query examples
            q_xyz, q_p_fea, z_query = self.embedding(query_x)
            scores = self.classifier(z_query)

        elif not training:
            with torch.no_grad():
                q_xyz, q_p_fea, z_query = self.embedding(query_x)
                scores = self.classifier(z_query)
        loss = self.loss_fn(scores, query_y)

        # get acc
        log_p_y = F.log_softmax(scores, dim=1)
        y_hat = log_p_y.max(1)[1].view(n_way, n_query)
        acc_cla = y_hat.eq(query_y.view(n_way, n_query)).float().mean(dim=1)
        acc_val = acc_cla.mean()

        # get acc for each class
        class_unique = torch.unique(y)
        acc_dict = dict(map(lambda x, y: [x, y], class_unique.cpu().numpy(), acc_cla.cpu().numpy()))

        # map y_hat to y
        y_hat = y_hat.cpu().numpy()
        for i in range(n_way):
            y_hat[y_hat == i] = class_unique[i].cpu().numpy()
        return loss, acc_val, acc_dict, y_hat

    def train_loop(self, tr_iter, optimizer, nclass):  # overwrite parent function
        train_loss = list()
        train_acc = list()
        train_acc_class = np.zeros(nclass)
        train_count = np.zeros(nclass)

        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        # train
        for i, (x, y) in tqdm(enumerate(tr_iter)):
            # Data Augmented
            if self.args.DataAug:
                rotated_data = rotate_point_cloud(x)
                jittered_data = jitter_point_cloud(rotated_data)
                x = torch.from_numpy(jittered_data).type(torch.FloatTensor)
            x, y = x.cuda(), y.cuda()

            # get loss
            loss, acc, acc_dict, y_hat = self.forward(x, y, training=True)

            # log info
            loss_all.append(loss)
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            for keys in acc_dict:
                train_acc_class[keys] += acc_dict[keys]
                train_count[keys] += 1

            # update
            task_count += 1
            if task_count == self.n_task: # MAML update several tasks at one time
                loss_q = torch.stack(loss_all).sum(0)
                loss_q.backward()
                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()

        return train_loss, train_acc, train_acc_class, train_count

    def val_loop(self, val_iter, nclass):
        val_loss = list()
        val_acc = list()
        val_acc_class = np.zeros(nclass)
        val_count = np.zeros(nclass)

        for i, (x, y) in tqdm(enumerate(val_iter)):
            if self.args.DataAug:
                rotated_data = rotate_point_cloud(x)
                jittered_data = jitter_point_cloud(rotated_data)
                x = torch.from_numpy(jittered_data).type(torch.FloatTensor)
            x, y = x.cuda(), y.cuda()

            loss, acc, acc_dict, y_hat = self.forward(x, y, training=False)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
            for keys in acc_dict:
                val_acc_class[keys] += acc_dict[keys]
                val_count[keys] += 1

        return val_loss, val_acc, val_acc_class, val_count

    def test_loop(self, test_iter, nclass):  # overwrite parent function
        test_acc = list()
        test_acc_class = np.zeros(nclass)
        test_count = np.zeros(nclass)
        for i, (x, y) in tqdm(enumerate(test_iter)):
            x, y = x.cuda(), y.cuda()
            loss, acc, acc_dict, y_hat = self.forward(x, y, training=False)

            test_acc.append(acc.item())
            for keys in acc_dict:
                test_acc_class[keys] += acc_dict[keys]
                test_count[keys] += 1

        return test_acc, test_acc_class, test_count


####################################
# adapted embedder for maml
# Here we reproduce the PointNet and DGCNN
####################################

def maml_init_(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            # weight.fast (fast weight) is the temporaily adapted weight
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv1d_fw(nn.Conv1d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv1d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv1d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv1d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv1d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv1d_fw, self).forward(x)

        return out


class BatchNorm1d_fw(nn.BatchNorm1d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm1d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml1/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml1/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out


def get_graph_feature(x, k=20, idx=None):

    def knn(x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

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


class Edgeconv_fw(nn.Module):
    def __init__(self, k=20, In_dims=6, Out_dims=64):
        super(Edgeconv_fw, self).__init__()
        self.k = k
        self.C = Conv2d_fw(In_dims, Out_dims, kernel_size=1, bias=False)
        self.BN = BatchNorm2d_fw(Out_dims)
        init_layer(self.C)
        init_layer(self.BN)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = F.relu(self.BN(self.C(x)))
        x = x.max(dim=-1, keepdim=False)[0]
        # x = F.adaptive_max_pool1d(x, 1).squeeze()
        return x


class MAMLPointNetBackbone(nn.Module):
    def __init__(self, Out_dims):
        super(MAMLPointNetBackbone, self).__init__()
        # feature extracting
        self.conv1 = Conv1d_fw(3, 64, kernel_size=1, bias=False)
        self.conv2 = Conv1d_fw(64, 64, kernel_size=1, bias=False)
        self.conv3 = Conv1d_fw(64, 128, kernel_size=1, bias=False)
        self.conv4 = Conv1d_fw(128, Out_dims, kernel_size=1, bias=False)
        self.bn1 = BatchNorm1d_fw(64)
        self.bn2 = BatchNorm1d_fw(64)
        self.bn3 = BatchNorm1d_fw(128)
        self.bn4 = BatchNorm1d_fw(Out_dims)

    def forward(self, x):
        xyz = x
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        p_fea = F.relu(self.bn4(self.conv4(x)))
        g_fea = p_fea.max(-1)[0]

        return xyz, p_fea, g_fea


class MAMLdgcnnBackbone(nn.Module):
    def __init__(self, k=20, Out_dims=1024):
        super(MAMLdgcnnBackbone, self).__init__()
        # feature extracting

        super(MAMLdgcnnBackbone, self).__init__()
        self.Econv1 = Edgeconv_fw(k=k, In_dims=3 * 2, Out_dims=64)
        self.Econv2 = Edgeconv_fw(k=k, In_dims=64 * 2, Out_dims=64)
        self.Econv3 = Edgeconv_fw(k=k, In_dims=64 * 2, Out_dims=128)
        self.Econv4 = Edgeconv_fw(k=k, In_dims=128 * 2, Out_dims=256)
        self.MLP = nn.Sequential(Conv1d_fw(512, Out_dims, kernel_size=1, bias=False),
                                 BatchNorm1d_fw(Out_dims),
                                 nn.LeakyReLU(negative_slope=0.2))
    def forward(self, x):
        xyz = x
        x = x.transpose(2, 1)
        x1 = self.Econv1(x)
        x2 = self.Econv2(x1)
        x3 = self.Econv3(x2)
        x4 = self.Econv4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        p_fea = self.MLP(x)
        g_fea = p_fea.max(-1)[0]
        return xyz, p_fea, g_fea


class MAMLClassifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(MAMLClassifier, self).__init__()
        # classifier
        self.c_fc1 = Linear_fw(input_dim, 256)
        self.c_fc1_bn = BatchNorm1d_fw(256)
        self.c_fc1_dp = nn.Dropout(0.3)
        self.c_fc2 = Linear_fw(256, nclass)

        maml_init_(self.c_fc1)
        maml_init_(self.c_fc2)

    def forward(self, x):
        x = F.relu(self.c_fc1_dp(self.c_fc1_bn(self.c_fc1(x))))
        x = self.c_fc2(x)
        return x




