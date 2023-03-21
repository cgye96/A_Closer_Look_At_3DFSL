# -*- coding:utf-8 -*-
'''
created by Chuangguan Ye
code for A Closer Look at Few-Shot 3D Point Cloud Classification, IJCV 2022
       & What Makes for Effective Few-Shot Point Cloud Classification, WACV 2022
'''

import os
import torch
import numpy as np

from args import arg_setting
from init import init_device, init_seed, init_path, init_backbone, init_fsl_head, init_dataloader
from utils.log_utils import log_string, write_log, save_ckpt, \
                            resume_ckpt, copy_checkpoint, \
                            mean_confidence_interval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def trainval(args, model, dloader, path_dic, k_fold, nclass, device):
    # init opt
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=args.gamma, step_size=args.step)
    best_acc, last_epoch, best_epoch = 0.0, 0, 0

    # init log
    log_string(path_dic['log_file'], f'{k_fold}-fold trainval')

    # train from ckpt
    if args.ckpt:
        if not os.path.exists(f"{args.resume}/ckpt/{path_dic['ckpt_file'][k_fold]}"):
            log_string(path_dic['log_file'], 'ckpt file does not exist, start to train from scratch!')
            start_epoch = 0
        else:
            # copy_checkpoint
            copy_checkpoint(args, path_dic, k_fold)

            # load ckpt
            ckpt_file = f"{args.resume}/ckpt/{path_dic['ckpt_file'][k_fold]}"
            train_info, model, optim, lr_scheduler = resume_ckpt(
                model, optim, lr_scheduler, ckpt_file, device)
            [best_epoch, last_epoch, best_acc] = train_info
            log_string(path_dic['log_file'], f"load the ckpt from:{ckpt_file}")
            log_string(path_dic['log_file'], f"best_epoch:{best_epoch}"
                                             f"last_epoch:{last_epoch}"
                                             f"best_acc:{best_acc}")
            start_epoch = last_epoch + 1
    else:
        start_epoch = 0

    # start training
    for epoch in range(start_epoch, args.trepo + 1, 1):

        # early stop
        if (epoch - best_epoch) == args.stops or epoch == args.trepo:
            torch.save(model.state_dict(), path_dic['last_model'][k_fold])
            break

        # get data
        tr_iter = iter(dloader['train'])
        val_iter = iter(dloader['val'])

        model.train()
        # train_info = [t_loss, t_acc, t_acc4cla, t_count4cla]
        train_info = model.train_loop(tr_iter, optim, nclass)

        model.eval()
        # val_info = [v_loss, v_acc, v_acc4cla, v_count4cla]
        val_info = model.val_loop(val_iter, nclass)

        lr_scheduler.step()

        # save model according to the mean validating acc
        avg_v_acc = np.mean(val_info[1]) * 100
        if avg_v_acc >= best_acc:
            best_acc, best_epoch = avg_v_acc, epoch
            torch.save(model.state_dict(), path_dic['best_model'][k_fold])
            write_log(args, path_dic['log_file'], epoch, train_info, val_info)
        else:
            write_log(args, path_dic['log_file'], epoch, train_info, val_info, best_acc)

        # save ckpt
        ckpt_info = [best_epoch, epoch, best_acc]
        save_ckpt(path_dic['ckpt_path'][k_fold], ckpt_info, model, optim, lr_scheduler)


def test(args,  model, dloader, ckpt_path, path_dic, k_fold, nclass):
    test_acc = list()
    acc_avg = np.zeros(nclass)

    # init log
    log_string(path_dic['log_file'], f'{k_fold}-fold test')

    # load model
    model.load_state_dict(torch.load(path_dic[ckpt_path][k_fold]), strict=False)

    # start testing
    for epoch in range(args.teepo):
        test_iter = iter(dloader['test'])

        model.eval()
        acc, acc4cla, count4cla = model.test_loop(test_iter, nclass)

        avg_acc = np.mean(acc)
        test_acc.extend(acc)

        # save log for each epoch
        log_string(path_dic['log_file'], 'epoch:{}, avg test acc:{:.4f}'.format(epoch, np.mean(avg_acc)))

    # save log for each fold
    log_string(path_dic['log_file'], '{}_fold Average Test Acc: {:.2f}+-{:.2f}%'
               .format(k_fold, np.mean(test_acc) * 100, mean_confidence_interval(test_acc) * 100))

    return test_acc


def main(cross_val):

    # get args
    args = arg_setting()
    path_dic = init_path(args)

    # save args
    argDict = args.__dict__
    for eachArg, value in argDict.items():
        log_string(path_dic['arg'], f'{eachArg}: {str(value)}')
    log_string(path_dic['log_file'], str(args), if_print=False)

    # Init device, seed, backbone
    init_seed(args)
    device = init_device(args)
    backbone = init_backbone(args)

    # Cross train-Val
    if cross_val:
        test_acc_best = list()
        test_acc_last = list()

        # k-fold cross validation
        for k in range(args.k_fold):
            # Init fsl head
            model = init_fsl_head(args, backbone)
            dataloader, nclass = init_dataloader(args, BASE_DIR, k)

            # train & val
            if args.mode == 'train':
                print('start to train & val..')
                trainval(args=args,
                         model=model,
                         dloader=dataloader,
                         nclass=nclass,
                         path_dic=path_dic,
                         k_fold=k,
                         device=device,)
            # test
            if args.mode == 'train' or 'test':
                # test the best model
                print('testing the best model..')
                acc_best = test(args=args,
                                model=model,
                                dloader=dataloader,
                                ckpt_path='best_model',
                                path_dic=path_dic,
                                k_fold=k,
                                nclass=nclass)
                test_acc_best.extend(acc_best)

                # test the last model
                print('testing the last model..')
                acc_last = test(args=args,
                                model=model,
                                dloader=dataloader,
                                ckpt_path='last_model',
                                path_dic=path_dic,
                                k_fold=k,
                                nclass=nclass)
                test_acc_last.extend(acc_last)

        log_string(path_dic['log_file'], '{}-Fold Average Test Acc of Best Model = {:.2f} +- {:.2f}%'
                   .format(args.k_fold, np.mean(test_acc_best) * 100, mean_confidence_interval(test_acc_best) * 100))
        log_string(path_dic['log_file'], '{}-FoldAverage Test Acc of Last Model = {:.2f} +- {:.2f}%'
                   .format(args.k_fold, np.mean(test_acc_last) * 100, mean_confidence_interval(test_acc_last) * 100))
    else:
        raise EOFError('please set : Cross_Val=True')


if __name__ == '__main__':
    main(cross_val=True)
