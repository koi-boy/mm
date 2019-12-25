#!/usr/bin/env python
# encoding:utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software:
@file: analyze_logs.py
@time: 2019/9/6 16:23
@desc: mmdetection-master/tools/analyze_logs
"""
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print('{}Analyze train time of {}{}'.format('-' * 5, args.json_logs[i],
                                                    '-' * 5))
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print('slowest epoch {}, average time is {:.4f}'.format(
            slowest_epoch + 1, epoch_ave_time[slowest_epoch]))
        print('fastest epoch {}, average time is {:.4f}'.format(
            fastest_epoch + 1, epoch_ave_time[fastest_epoch]))
        print('time std over epochs is {:.4f}'.format(std_over_epoch))
        print('average iter time: {:.4f} s/iter'.format(np.mean(all_times)))
        print()


def plot_curve(log_dicts, args, max_iter=None):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append('{}_{}'.format(json_log, metric))
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):  # for each log file
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):  # for each metric
            print('plot curve of {}, metric is {}'.format(
                args.json_logs[i], metric))
            if metric not in log_dict[epochs[0]]:
                raise KeyError('{} does not contain metric {}'.format(
                    args.json_logs[i], metric))

            if 'mAP' in metric:
                xs = np.arange(1, max(epochs) + 1)
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                ax = plt.gca()
                ax.set_xticks(xs)
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                num_iters_per_epoch = log_dict[epochs[0]]['iter'][-1]
                for epoch in epochs:
                    iters = log_dict[epoch]['iter']
                    if log_dict[epoch]['mode'][-1] == 'val':
                        iters = iters[:-1]
                    xs.append(
                        np.array(iters) + (epoch - 1) * num_iters_per_epoch)
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                plt.xlabel('iter')
                if max_iter is not None:
                    cur = 0
                    for k in range(len(xs)):
                        cur = k
                        if xs[k] > max_iter:
                            break
                    xs = xs[:cur]
                    ys = ys[:cur]

                plt.plot(
                    xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
            plt.legend()
        if args.title is not None:
            plt.title(args.title)

    if args.out is None:
        plt.show()
    else:
        print('save curve to: {}'.format(args.out))
        plt.savefig(args.out)
        plt.cla()


def main_plot_log_curve(log_file, fig_title,
                        save_file=None, fig_style='dark', metrics=None):

    if metrics is None:
        raise Exception("give a list of metrics for plotting, such as ['acc, loss']")
    json_logs = [log_file]
    log_dicts = load_json_logs(json_logs)

    parser = argparse.ArgumentParser(description='Analyze Json Log')
    args = parser.parse_args()

    args.backend = None

    args.legend = []
    for metric in metrics:
        args.legend.append('{}_{}'.format('log', metric))

    args.json_logs = json_logs
    args.keys = metrics
    args.title = fig_title
    args.style = fig_style
    args.out = save_file

    plot_curve(log_dicts, args, max_iter=200)


def main_training_time_analysis(log_file, include_outliers=True):

    parser = argparse.ArgumentParser(description='Analyze train time ')
    args = parser.parse_args()
    args.json_logs = [log_file]
    args.include_outliers = include_outliers
    log_dicts = load_json_logs(args.json_logs)
    cal_train_time(log_dicts, args)


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for l in log_file:
                log = json.loads(l.strip())
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


# if __name__ == '__main__':
#     # main()
#     log_file_ = '/home/scj/mm_detection_proj/stations/visionox_v3/training_models/20190823/20190823_235910.log.json'
#     # main_plot_log_curve(log_file_, 'visonox_training', metrics=['acc', 'loss'])
#     main_training_time_analysis(log_file_)
