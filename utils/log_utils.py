# -*- coding: utf-8 -*-
'''
created by Chuangguan Ye
log utils
'''
import numpy as np
import scipy.stats
import torch
import os
from shutil import copyfile


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def log_string(path, out_str, if_print=True):
    file = open(path, 'a')
    file.write(out_str + '\n')
    file.flush()
    if if_print:
        print(out_str)


def write_log(arg, log_file, ep, train_info, val_info, best_acc=None):
    LOG_FOUT = log_file
    t_loss, t_acc, t_acc4cla, t_count4cla = train_info
    v_loss, v_acc, v_acc4cla, v_count4cla = val_info
    avg_tacc, avg_tloss = np.mean(t_acc), np.mean(t_loss)
    avg_vacc, avg_vloss = np.mean(v_acc), np.mean(v_loss)

    if best_acc is not None:
        log_string(LOG_FOUT, 'epoch {},    '
                              'train, loss={:.3f} acc={:.2f}+-{:.2f}%   '
                              'val,   loss={:.3f} acc={:.2f}+-{:.2f}%(best={:.3f})'
                   .format(ep,
                           avg_tloss, avg_tacc * 100, mean_confidence_interval(t_acc[-arg.trtask:]) * 100,
                           avg_vloss, avg_vacc * 100, mean_confidence_interval(v_acc[-arg.vatask:]) * 100,
                           best_acc))
    else:
        log_string(LOG_FOUT, 'epoch {},    '
                              'train, loss={:.3f} acc={:.2f}+-{:.2f}%   '
                              'val,   loss={:.3f} acc={:.2f}+-{:.2f}%(best)'
                   .format(ep,
                           avg_tloss, avg_tacc * 100, mean_confidence_interval(t_acc[-arg.trtask:]) * 100,
                           avg_vloss, avg_vacc * 100, mean_confidence_interval(v_acc[-arg.vatask:]) * 100))


def save_ckpt(save_path, train_info, model, optim, lr_schedule):
    torch.save({
        't_inf': train_info,
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'lr_schedule': lr_schedule.state_dict()
    }, save_path)
    return save_path


def resume_ckpt(model, optim, lr_schedule, resume, device):
    ckpt = torch.load(resume, map_location=device)
    train_info = ckpt['t_inf']
    model.load_state_dict(ckpt['model'])
    optim.load_state_dict(ckpt['optim'])
    if lr_schedule is not None:
        lr_schedule.load_state_dict(ckpt['lr_schedule'])
    return train_info, model, optim, lr_schedule


def copy_checkpoint(args, path_dic, k_fold):
    # copy the best model from checkpoint
    log_string(path_dic['log_file'], f"copy the best model from:{args.resume}")
    source_file = f"{args.resume}/bestmodel/{os.path.basename(path_dic['best_model'][k_fold])}"
    destination_file = path_dic['best_model'][k_fold]
    copyfile(source_file, destination_file)

    # copy the last model from checkpoint
    log_string(path_dic['log_file'], f"copy the last model from:{args.resume}/lastmodel")
    source_file = f"{args.resume}/lastmodel/{os.path.basename(path_dic['last_model'][k_fold])}"
    destination_file = path_dic['last_model'][k_fold]
    if os.path.exists(source_file):
        copyfile(source_file, destination_file)

    # copy the ckpt
    log_string(path_dic['log_file'], f"copy the ckpt from:{args.resume}/ckpt")
    source_file = f"{args.resume}/ckpt/{path_dic['ckpt_file'][k_fold]}"
    destination_file = path_dic['ckpt_path'][k_fold]
    copyfile(source_file, destination_file)
