# -*- coding: utf-8 -*-
import os
import sys
import argparse


def arg_setting():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    parser = argparse.ArgumentParser(description="few-shot learning for point cloud classification")
    parser.add_argument('--base_path', type=str, default=BASE_DIR)
    parser.add_argument('--mode',      type=str, default='train',           help='train or test')
    parser.add_argument('--cuda',      type=str, default='0',               help='GPU ID')
    parser.add_argument('--exp',       type=str, default='_benchmark',      help='exp name')
    parser.add_argument('--dataset',   type=str, default='ModelNet40_FS', help='dataset')
    parser.add_argument('--backbone',  type=str, default='PointNet',      help='backbone network')
    parser.add_argument('--method',    type=str, default='protonet',      help='few-shot classification algorithms')
    parser.add_argument('--metric',    type=str, default='Euler', help='metric function for Proto: Euler or Cos')
    parser.add_argument('--note',      type=str, default='fsl3d', help='note info')
    parser.add_argument('--DataAug',  type=bool, default=True,    help='Data Augmentation')

    # few-shot settings
    parser.add_argument('--way',     type=int, default=5,   help='Default:train_way = test_way = way')
    parser.add_argument('--shot',    type=int, default=1,   help='Default:train_shot = test_shot = shot')
    parser.add_argument('--query',   type=int, default=15,  help='Default:train_query = test_query = query')
    parser.add_argument('--trepo',   type=int, default=80,  help='number of training epoch')
    parser.add_argument('--teepo',   type=int, default=1,   help='number of testing epoch')
    parser.add_argument('--trtask',  type=int, default=400, help='number of train episodes')
    parser.add_argument('--vatask',  type=int, default=600, help='number of validate episodes')
    parser.add_argument('--tetask',  type=int, default=700, help='number of test episodes')
    parser.add_argument('--stops',   type=int, default=80,  help='early stopping epoch')
    
    # pcia settings
    parser.add_argument('--CIFh',     type=int,  default=64,     help='[4,8,16,32,64,128]')
    parser.add_argument('--CIFk',     type=int,  default=20,     help='[5,15,30,45,60,75]')
    parser.add_argument('--SPFknn',   type=int,  default=16,     help='[1,4,8,16,32]')
    parser.add_argument('--SPFks',    type=int,  default=8,      help='[4,8,12,16,20,24,28]')
    parser.add_argument('--SCId',     type=int,  default=32,     help='[4,8,12,16,20,24,28]')
    parser.add_argument('--npoint',  type=int, default=512, help='number of points in each instance')
    parser.add_argument('--emb_dims', type=int, default=1024, help='dim of output feature')
    
    # Opt settings
    parser.add_argument('--gamma',  type=float, default=0.5,     help='gamma for Adam')
    parser.add_argument('--step',   type=int,   default=10,      help='steps for learning scheduler ')
    parser.add_argument('--lr',     type=float, default=0.0008,  help='learning rate for training, 0.0002 for FEAT')
    parser.add_argument('--lr_f',   type=float, default=0.1,     help='learning rate in MAML fast adaptaion')
    parser.add_argument('--n_task', type=int,   default=1,       help='adaptation task in MAML')
    parser.add_argument('--n_step', type=int,   default=2,       help='adaptation step in MAML')

    # sys settings
    parser.add_argument('--ckpt',   action='store_true',    help='resume_ckpt')
    parser.add_argument('--resume', type=str,  default='',  help='path to resume_ckpt file')
    parser.add_argument('--seed',   type=int,  default=7,   help='input for the manual seeds initializations')
    parser.add_argument('--k_fold', type=int,  default=5,   help='K-folds cross val: 5 for Model&Shape 3 for ScanObject')
    parser.add_argument('--config', help='config file for 3D Point cloud SFL')
    args = parser.parse_args()
    return args
