# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .basefsl import FSLBasedModel


class FSL3D(FSLBasedModel):
    def __init__(self, backbone, args):
        super(FSL3D, self).__init__(backbone, args)
        self.pcia = PCIA(args)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y, training=False):
        n_way, n_shot, n_query = self.args.way, self.args.shot, self.args.query

        # feature embedding
        xyz, p_feat, g_feat = self.feature_embedding(x)

        # split support and query features
        y, xyz, p_feat, z_support, z_query = self.split_supp_and_query(y, xyz, p_feat, g_feat)

        z_query = z_query.contiguous().view(n_way * n_query, -1)

        # pcia
        z_proto, z_query, center_id, local_id = self.pcia(z_support, z_query, p_feat, xyz)

        # get distance
        if self.args.metric == 'Cos':
            dists = F.cosine_similarity(z_query, z_proto)
        elif self.args.metric == 'Euler':
            dists = -euclidean_dist(z_query, z_proto)
        else:
            raise EOFError('Error metric!! Cos or Euler')

        # get loss
        y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
        total_loss = self.loss_fn(dists, y_query.long())

        # get acc
        log_p_y = F.log_softmax(dists, dim=1)
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


class PCIA(nn.Module):
    def __init__(self, args, emb_dims=1024):
        super(PCIA, self).__init__()
        self.args = args
        self.Emb_Dims = emb_dims

        # SPF
        self.Conv11 = nn.Sequential(nn.Conv2d(self.args.SPFks + 1, 1, kernel_size=1, bias=False))
        
        # SCIp
        self.fuse1 = nn.Sequential(nn.Conv2d(self.args.way * 1, 1, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.fuse2 = nn.Sequential(nn.Conv2d(2, 32, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.fuse3 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.FC1 = nn.Linear(32, 32, bias=False)
        self.FC2 = nn.Linear(32, 32, bias=False)
        self.FC3 = nn.Linear(32, 32, bias=False)

        self.dropout = nn.Dropout(0.4)

        self.softmax = nn.Softmax(dim=1)

        # CIFp
        self.sq_dims = self.args.CIFk + 1
        self.qs_dims = self.args.way + 1
        self.Conv1 = nn.Sequential(nn.Conv2d(self.sq_dims, self.args.CIFh, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.args.CIFh),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.Conv2 = nn.Sequential(nn.Conv2d(self.qs_dims, self.args.CIFh, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.args.CIFh),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.Conv3 = nn.Sequential(nn.Conv2d(self.args.CIFh, self.sq_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.sq_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.Conv4 = nn.Sequential(nn.Conv2d(self.args.CIFh, self.qs_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.qs_dims),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.Conv5 = nn.Sequential(nn.Conv2d(16, self.args.CIFh, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.args.CIFh),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.Conv6 = nn.Sequential(nn.Conv2d(self.args.CIFh, 16, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, s, q, p_fea, coords):
        way, shot, query, dims = self.args.way, self.args.shot, self.args.query, self.args.emb_dims
        b = way * (shot + query)
        center_idx, local_idx = None, None

        # spf
        p_fea = p_fea.transpose(2, 1)                                              
        s_mean = s.view(way, shot, dims).mean(1, keepdim=True)
        s = s_mean.expand(way, shot, dims).reshape(way * shot, dims)
        g_fea = torch.cat((s, q), dim=0).view(b, 1, dims)                           

        # 1,get key_idx
        center_idx = cos_similarity_batch(g_fea, p_fea, nk=self.args.SPFknn).squeeze()

        # 2,knn
        c_fea = index_points(p_fea, center_idx.long())       
        local_idx = knn_point(self.args.SPFks, p_fea, c_fea)
        grouped_fea = index_points(p_fea, local_idx) 

        # 3, fuse local feature
        grouped_fea = grouped_fea.view(-1, self.args.SPFks, dims).unsqueeze(2) 

        g_fea = g_fea.expand(b, self.args.SPFknn, dims)
        g_fea = g_fea.reshape(-1, 1, dims).unsqueeze(2) 
        g_l_feature = torch.cat((grouped_fea, g_fea), dim=1)
        mean_part_fea = self.Conv11(g_l_feature).view(b, self.args.SPFknn, dims) 
        spf_out_fea = torch.max(mean_part_fea, dim=1)[0]

        # 4, update s & q
        s = (spf_out_fea[:way * shot, :].view(way, shot, dims).mean(1) + s_mean.squeeze()) / 2.0
        q = (spf_out_fea[way * shot:, :] + q) / 2.0

        # SCIp
        s_meta = self.fuse1(s.view(way, 1, 1, dims).transpose(0, 1)).view(1, 1, dims)
        s_cat = torch.cat((s.view(way, 1, -1), s_meta.repeat(way, 1, 1)), dim=1)
        q_cat = torch.cat((q.view(way * query, 1, -1), s_meta.repeat(way * query, 1, 1)), dim=1)

        s_fuse = self.fuse2(s_cat.view(way, 2, 1, dims)).view(-1, 32, dims)
        q_fuse = self.fuse2(q_cat.view(way * query, 2, 1, dims)).view(-1, 32, dims)

        s_fuse_q = self.FC1(s_fuse.transpose(2, 1))
        s_fuse_k = self.FC2(s_fuse.transpose(2, 1))
        s_fuse_v = self.FC3(s_fuse.transpose(2, 1))

        q_fuse_q = self.FC1(q_fuse.transpose(2, 1))
        q_fuse_k = self.FC2(q_fuse.transpose(2, 1))
        q_fuse_v = self.FC3(q_fuse.transpose(2, 1))

        s_att_map = torch.bmm(s_fuse_q, s_fuse_k.transpose(2, 1)) / np.power(32, 0.5)
        q_att_map = torch.bmm(q_fuse_q, q_fuse_k.transpose(2, 1)) / np.power(32, 0.5)

        s_att_map = F.softmax(s_att_map, dim=2)
        q_att_map = F.softmax(q_att_map, dim=2)

        s_atten = torch.bmm(s_att_map, s_fuse_v)
        q_atten = torch.bmm(q_att_map, q_fuse_v)

        s_atten = self.fuse3(s_atten.view(-1, 1024, 32, 1).transpose(2, 1)).squeeze()
        q_atten = self.fuse3(q_atten.view(-1, 1024, 32, 1).transpose(2, 1)).squeeze()

        s = (s_atten + s) / 2.0
        q = (q_atten + q) / 2.0

        # CIFp
        s_init = s
        q_init = q

        sq_emd_att_map = torch.mm(s, q.transpose(0, 1)) / np.power(dims, 0.5)
        qs_emd_att_map = torch.mm(q, s.transpose(0, 1)) / np.power(dims, 0.5)

        sq_emd_att_map = F.softmax(sq_emd_att_map, dim=-1)
        qs_emd_att_map = F.softmax(qs_emd_att_map, dim=-1)

        sq_atten = torch.mm(sq_emd_att_map, q)
        qs_atten = torch.mm(qs_emd_att_map, s)

        idxsq = top_cos_similarity(s, q, self.args.CIFk)
        idxqs = top_cos_similarity(q, s, s.size(0))

        sq_a = torch.cat((s.view(way, 1, 1, dims), q[idxsq].view(way, self.args.CIFk, 1, dims)), dim=1)
        qs_a = torch.cat((s[idxqs].view(way * query, way, 1, dims), q.view(way * query, 1, 1, dims)), dim=1)

        sq_aa = F.softmax(self.Conv3(self.Conv1(sq_a)).squeeze(), dim=1)
        qs_aa = F.softmax(self.Conv4(self.Conv2(qs_a)).squeeze(), dim=1)

        s_att = torch.mul(sq_aa, sq_a.squeeze()).sum(dim=1)
        q_att = torch.mul(qs_aa, qs_a.squeeze()).sum(dim=1)

        s = (s_att + s_init + sq_atten) / 3.0
        q = (q_att + q_init + qs_atten) / 3.0

        idxsq = top_cos_similarity(s, q, 15)
        sq_a = torch.cat((s.view(way, 1, 1, dims), q[idxsq].view(way, 15, 1, dims)), dim=1)
        sq_aa = F.softmax(self.Conv6(self.Conv5(sq_a)).squeeze(), dim=1)
        s_att = torch.mul(sq_aa, sq_a.squeeze()).sum(dim=1)

        s = (s_att + s) / 2.0

        return s, q, center_idx, local_idx


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def euclidean_dist_batch(x, y):
    # x: B x N x D
    # y: B x M x D
    b = x.size(0)
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    assert d == y.size(2)
    x = x.unsqueeze(2).expand(b, n, m, d)
    y = y.unsqueeze(1).expand(b, n, m, d)
    return torch.pow(x - y, 2).sum(3)


def cos_similarity(x, y):
    # x: N x D
    # y: M x D
    cos = torch.mm(F.normalize(x, dim=-1),
                   F.normalize(y, dim=-1).transpose(1, 0))
    return cos


def top_cos_similarity(x, y, k, nomal=True):
    # x: N x D
    # y: M x D
    cos = torch.mm(F.normalize(x, dim=-1),
                   F.normalize(y, dim=-1).transpose(1, 0))
    cos = 0.5 * cos + 0.5 if nomal else cos
    index = cos.topk(k, dim=-1)[1]
    return index


def cos_similarity_batch(x, y, nk, nomal=True):
    # x: B x N x D
    # y: B x M x D
    cos = torch.bmm(F.normalize(x, dim=-1),
                    F.normalize(y, dim=-1).transpose(2, 1))
    cos = 0.5 * cos + 0.5 if nomal else cos
    index = cos.topk(nk, dim=-1)[1]
    return index


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    group_idx = topk(sqrdists, nsample, dim=-1, largest=False)
    return group_idx


def topk(inputs, k, dim=None, largest=True):
    if dim is None:
        dim = -1
    if dim < 0:
        dim += inputs.ndim
    transpose_dims = [i for i in range(inputs.ndim)]
    transpose_dims[0] = dim
    transpose_dims[dim] = 0
    inputs = inputs.permute(2, 1, 0)
    index = torch.argsort(inputs, dim=0, descending=largest)
    indices = index[:k]
    indices = indices.permute(2, 1, 0)
    return indices


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx1: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


