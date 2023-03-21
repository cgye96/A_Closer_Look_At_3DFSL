# -*- coding: utf-8 -*-
'''
created by Chuangguan Ye
system init
'''

import os
import torch
import time
import random
import numpy as np
from shutil import copyfile

from dataloader import data_split_for_cross_val, FSLDataset, FSLBatchSampler
from backbone import *
from fsl_head import *


def init_device(args):
    '''
    Initialize the device
    '''
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        print('Use GPU: {}'.format(args.cuda))
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def init_seed(arg):
    '''
    Set random seed and disable cudnn to maximize reproducibility
    '''
    # torch.cuda.cudnn_enabled = False
    random.seed(arg.seed)
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    os.environ['PYTHONHASHSEED'] = str(arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True  ## for pointcnn
    return arg.seed


def init_dataloader(args, root, k):
    '''
    Initialize the dataloader
    '''
    dataloader_dict = {}
    args.data_path = os.path.join(root, f'data/{args.dataset}')
    tra_list, val_list, nclass = data_split_for_cross_val(args=args)
    tra_dataset = FSLDataset(args=args, k=k, split=tra_list[k], mode='train')
    val_dataset = FSLDataset(args=args, k=k, split=val_list[k], mode='train')
    tes_dataset = FSLDataset(args=args, k=k, split=None, mode='test')

    tra_sampler = FSLBatchSampler(args=args, labels=tra_dataset.label, iterations=args.trtask)
    val_sampler = FSLBatchSampler(args=args, labels=val_dataset.label, iterations=args.vatask)
    tes_sampler = FSLBatchSampler(args=args, labels=tes_dataset.label, iterations=args.tetask)

    dataloader_dict['train'] = torch.utils.data.DataLoader(tra_dataset, batch_sampler=tra_sampler)
    dataloader_dict['val'] = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler)
    dataloader_dict['test'] = torch.utils.data.DataLoader(tes_dataset, batch_sampler=tes_sampler)

    return dataloader_dict, nclass


def init_backbone(args):
    if args.backbone == 'PointNet':
        out = PointNet(emb_dims=args.emb_dims)
    elif args.backbone == 'DGCNN':
        out = DGCNN(emb_dims=args.emb_dims)
    elif args.backbone == 'PointNet2':
        out = Pointnet2SSG(emb_dims=args.emb_dims)
    elif args.backbone == 'RSCNN':
        out = RSCNN_SSN(emb_dims=args.emb_dims)
    elif args.backbone == 'DensePoint':
        out = Densepoint(emb_dims=args.emb_dims)
    elif args.backbone == 'PointCNN':
        out = PointCNN(emb_dims=args.emb_dims)
    else:
        raise EOFError('Not implementation!')
    return out


def init_fsl_head(args, backbone):
    if args.method == 'protonet':
        model = PrototypicalNet(backbone=backbone, args=args)
    elif args.method == 'relationnet':
        model = RelationNet(backbone=backbone, args=args)
    elif args.method == 'fslgnn':
        model = FSL_GNN(backbone=backbone, args=args)
    elif args.method == 'maml':
        model = MAML(args=args)
    elif args.method == 'metaoptnet':
        model = MetaOptNet(backbone=backbone, args=args)
    elif args.method == 'fsl3d':
        model = FSL3D(backbone=backbone, args=args)
    else:
        raise RuntimeError('Not implementation!')
    return model.cuda()


def init_path(args):
    '''
    Initialize paths to output files
    '''
    file_path = {}

    # base info
    root_path = args.base_path
    time_flag = time.strftime("%m%d%H%M%S_", time.localtime(time.time()))
    pref_Info = f'{args.backbone}_{args.method}_{args.metric}' \
                f'_{args.way}w{args.shot}s{args.query}q{args.npoint}p_{args.note}'
    save_dir = f'output/exp{args.exp}/{args.dataset}/{args.backbone}/{args.method}/{time_flag + pref_Info}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # train log path
    Log_File1 = time_flag + pref_Info + '.log'
    log_dir = save_dir + '/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    Log_path = os.path.join(root_path, log_dir, Log_File1)
    file_path['log_file'] = Log_path

    # ckpt path
    ckpt_dir = save_dir + '/ckpt'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = [pref_Info + f'_ckpt_{i}.pth.tar' for i in range(args.k_fold)]
    ckpt_path = [os.path.join(root_path, ckpt_dir, ckpt_file[i]) for i in range(args.k_fold)]
    file_path['ckpt_path'] = ckpt_path
    file_path['ckpt_file'] = ckpt_file

    # model path
    best_model_dir = save_dir + '/bestmodel'
    last_model_dir = save_dir + '/lastmodel'
    if not os.path.exists(best_model_dir) or not os.path.exists(last_model_dir):
        os.makedirs(best_model_dir)
        os.makedirs(last_model_dir)
    best_model_name = [pref_Info + f'_best_model_{i}.pth' for i in range(args.k_fold)]
    last_model_name = [pref_Info + f'_last_model_{i}.pth' for i in range(args.k_fold)]
    best_model_path = [os.path.join(root_path, best_model_dir, best_model_name[i]) for i in range(args.k_fold)]
    last_model_path = [os.path.join(root_path, last_model_dir, last_model_name[i]) for i in range(args.k_fold)]
    file_path['best_model'] = best_model_path
    file_path['last_model'] = last_model_path

    # save args file
    file_path['arg'] = save_dir + '/arg.log'

    # save fsl_method.py file
    source_file = root_path + f'/fsl_head/{args.method}.py'
    destination_file = save_dir + f'/{args.method}.py'
    copyfile(source_file, destination_file)

    return file_path

