# -*- coding: utf-8 -*-
# ref: https://github.com/wyharveychen/CloserLookFewShot

import torch
import numpy as np
import torch.nn as nn
from abc import abstractmethod
from tqdm import tqdm
from utils.data_utils import rotate_point_cloud, jitter_point_cloud


class FSLBasedModel(nn.Module):
    """
    template of FSL
    """
    def __init__(self, backbone, args):
        super(FSLBasedModel, self).__init__()
        self.args = args
        self.backbone = backbone
        self.feat_dim = self.backbone.emb_dims

    @abstractmethod
    def forward(self, x, y, training=False):
        pass

    def feature_embedding(self, x):
        """
        feature embedding
        input:  x [B,N,3]
        return: xyz:       raw coordinate [B,N,3]
                g_feat:    global features [B,C]
                p_feat:    point-wised features [B,N,C]
        """
        xyz, p_feat, g_feat = self.backbone.forward(x)       # others
        return xyz, p_feat, g_feat

    def split_supp_and_query(self, y, xyz, p_feat, g_feat):
        """
        split supp set and query set
        input:  y: labels [B,N]
                g_feat:    global features [B,C]
                p_feat:    point-wised features [B,N,C]
        return: xyz:        raw coordinate [B,N,3]
                z_support:  support set features [nway * nshot, C]
                z_query:    query set features [nway * nquery, C]
                p_feat:     point-wised features [B,N,C]
        """
        class_unique = torch.unique(y)
        s_idx = torch.stack(list(map(lambda c: y.eq(c).nonzero()[:self.args.shot], class_unique))).view(-1)  # support id
        q_idx = torch.stack(list(map(lambda c: y.eq(c).nonzero()[self.args.shot:], class_unique))).view(-1)  # query id
        z_support, z_query = g_feat[s_idx], g_feat[q_idx]
        p_feat = torch.cat((p_feat[s_idx], p_feat[q_idx]), dim=0)
        y, xyz = torch.cat((y[s_idx], y[q_idx]), dim=0), torch.cat((xyz[s_idx], xyz[q_idx]), dim=0)
        return y, xyz, p_feat, z_support, z_query

    def train_loop(self, tr_iter, optimizer, nclass):
        """
        train loop
        input:  tr_iter:   dataload  iter(DataLoader(datatset,sampler))
                optimizer: optimizer
                nclass:    classes of dataset: 40 70 15
                device:    GPU or CPU
        """
        train_loss = list()
        train_acc = list()
        train_acc_class = np.zeros(nclass)
        train_count = np.zeros(nclass)

        for i, batch in tqdm(enumerate(tr_iter)):
            x, y = batch
            if self.args.DataAug:
                rotated_data = rotate_point_cloud(x)
                jittered_data = jitter_point_cloud(rotated_data)
                x = torch.from_numpy(jittered_data).type(torch.FloatTensor)
            x, y = x.cuda(), y.cuda()

            # train
            optimizer.zero_grad()
            loss, acc, acc_dict, y_hat = self.forward(x, y, training=True)
            loss.backward()
            optimizer.step()

            # log info
            train_loss.append(loss.item())
            train_acc.append(acc.item())
            # record acc for each class
            for keys in acc_dict:
                train_acc_class[keys] += acc_dict[keys]
                train_count[keys] += 1

        return train_loss, train_acc, train_acc_class, train_count

    def val_loop(self, val_iter, nclass):
        with torch.no_grad():
            val_loss = list()
            val_acc = list()
            val_acc_class = np.zeros(nclass)
            val_count = np.zeros(nclass)

            for i, batch in tqdm(enumerate(val_iter)):
                x, y = batch
                if self.args.DataAug:
                    rotated_data = rotate_point_cloud(x)
                    jittered_data = jitter_point_cloud(rotated_data)
                    x = torch.from_numpy(jittered_data).type(torch.FloatTensor)
                x, y = x.cuda(), y.cuda()

                # val
                loss, acc, acc_dict, y_hat = self.forward(x, y)

                # log info
                val_loss.append(loss.item())
                val_acc.append(acc.item())
                # record acc for each class
                for keys in acc_dict:
                    val_acc_class[keys] += acc_dict[keys]
                    val_count[keys] += 1

        return val_loss, val_acc, val_acc_class, val_count

    def test_loop(self, test_iter, nclass):
        with torch.no_grad():
            test_acc = list()
            test_acc_class = np.zeros(nclass)
            test_count = np.zeros(nclass)
            for i, batch in tqdm(enumerate(test_iter)):
                x, y = batch
                x, y = x.cuda(), y.cuda()

                loss, acc, acc_dict, y_hat = self.forward(x, y, training=False)

                # log info
                test_acc.append(acc.item())
                # record acc for each class
                for keys in acc_dict:
                    test_acc_class[keys] += acc_dict[keys]
                    test_count[keys] += 1

        return test_acc, test_acc_class, test_count
