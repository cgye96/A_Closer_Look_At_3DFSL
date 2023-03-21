# -*- coding: utf-8 -*-
'''
created by Chuangguan Ye
dataloader for FSL3D
'''


import os
import numpy as np
import torch
import torch.utils.data as data
import h5py
import random


class FSLDataset(data.Dataset):
    def __init__(self, args, split=None, mode='train', k=None, transform=None):
        super(FSLDataset, self).__init__()

        self.root = args.data_path
        self.transform = transform
        self.mode = mode

        # dataset path
        if args.dataset == 'ModelNet40_FS':
            hdf5_file = os.path.join(self.root, 'ModelNet40_FS_' + mode + '.h5')
        elif args.dataset == 'ShapeNet70_FS':
            hdf5_file = os.path.join(self.root, 'ShapeNet70_FS_' + mode + '.h5')
        elif args.dataset == 'ScanObjectNN_FS':
            hdf5_file = os.path.join(self.root, mode + f'_scanobjectnn_PB_T50_RS_fsl_{k + 1}.h5')
        else:
            raise RuntimeError('Dataset not found.')

        # load data and label
        f = h5py.File(hdf5_file, 'r')
        self.data = np.array(f['data'][:])
        self.label = np.array(f['label'][:])

        # split train/val/test set
        if split is not None:
            self.split = np.array(split)
            labels = np.unique(self.label)
            choice = labels[self.split]
            idx = list()
            for j in range(len(self.label)):
                for i in range(len(choice)):
                    if choice[i] == self.label[j]:
                        idx.append(j)
                        break
            idx = np.array(idx)
            self.data = self.data[idx]
            self.label = self.label[idx]

        # Randomly choose N Points
        index_p = np.random.choice(self.data.shape[1], args.npoint, replace=False)
        self.data = self.data[:, index_p, :]
        print(f"== Dataset: Found {len(self.data)} items for {mode}")
        self.x = torch.from_numpy(self.data)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.label[idx]

    def __len__(self):
        return len(self.data)


class FSLBatchSampler(object):
    def __init__(self, args, labels, iterations):
        super(FSLBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = args.way
        self.sample_per_class = args.shot + args.query
        self.iterations = iterations
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)
        self.labels = torch.LongTensor(self.labels)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = np.random.choice(len(self.classes), cpi, replace=False)
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = np.random.choice(self.numel_per_class[label_idx], spc, replace=False)
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[np.random.choice(len(batch), len(batch), replace=False)]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


def data_split_for_cross_val(args):

    random.seed(args.seed)   # 7

    v_list = [0] * args.k_fold
    t_list = [0] * args.k_fold

    if args.dataset == 'ScanObjectNN_FS':
        t_list = [None] * args.k_fold
        v_list = [None] * args.k_fold
        total_class = 15
        return t_list, v_list, total_class

    if args.dataset == 'ModelNet40_FS':
        total_class, train_class, test_class = 40, 30, 10

    elif args.dataset == 'ShapeNet70_FS':
        total_class, train_class, test_class = 70, 50, 20
    else:
        raise EOFError('ERROR OF DATASET!')

    if args.way == 5:
        index = random.sample(range(0, train_class), train_class)
        n_pre = int(train_class / args.k_fold)
        for i in range(args.k_fold):
            v_list[i] = list(set(index[i * n_pre: (i + 1) * n_pre]))  # val
            t_list[i] = list(set(index).difference(set(v_list[i])))   # train

    elif args.way > 5:
        for i in range(args.k_fold):
            v_list[i] = list(set(random.sample(range(0, train_class), train_class)[:args.way]))  # val
            t_list[i] = list(set(random.sample(range(0, train_class), train_class)[args.way:]))  # train

    return t_list, v_list, total_class




