#!/usr/bin/env python
# encoding:utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software:
@file: plotting.py
@time: 2019/9/6 16:23
@desc:
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def auto_label(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def bar_plot(name2count, save_fig, control_line=500,
             count_label='Train', y_label='Count', title='Image Count for Each Code'):
    labels = []
    count = []
    for key in sorted(name2count.keys()):
        labels.append(key)
        count.append(name2count[key])

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(x, count, width, label=count_label)

    line1 = [(-1, control_line), (len(labels), control_line)]
    (line1_xs, line1_ys) = zip(*line1)

    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='red'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, rotation_mode='anchor', ha='right', size='x-small')
    ax.legend()

    auto_label(rects, ax)
    fig.tight_layout()
    plt.savefig(save_fig)

    plt.show()


def bar_plot_pair(name2count1, name2count2, save_fig, control_line=500,
                  count1_label='Train', count2_label='Test',
                  y_label='Count', title='Image Count for Each Code'):
    labels = []
    count1 = []
    count2 = []
    for key in sorted(name2count1.keys()):
        if key in name2count2.keys():
            labels.append(key)
            count1.append(name2count1[key])
            count2.append(name2count2[key])

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, count1, width, label=count1_label)
    rects2 = ax.bar(x + width / 2, count2, width, label=count2_label)

    line1 = [(-1, control_line), (len(labels), control_line)]
    (line1_xs, line1_ys) = zip(*line1)

    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='red'))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    auto_label(rects1, ax)
    auto_label(rects2, ax)

    fig.tight_layout()
    plt.savefig(save_fig)

    plt.show()


def pie_plot(name2count, save_fig):
    """

    :param name2count:
    :param save_fig:
    :return:
    """

    labels = list(name2count.keys())
    fracs = [name2count[x] for x in labels]

    fig, axs = plt.subplots()
    # A standard pie plot
    axs.pie(fracs, labels=labels, autopct='%1.2f%%', shadow=False)
    plt.savefig(save_fig)
    plt.show()


def hist_plot(data_list, save_fig, x_label=None, bins=10):
    """

    """
    plt.figure()
    plt.hist(data_list, bins=bins)
    if x_label is not None:
        plt.xlabel(x_label)
    plt.ylabel('Probability')
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(save_fig)


# if __name__ == '__main__':
#     labels = ['G1', 'G2', 'G3', 'G4', 'G5']
#     men_means = [20, 34, 30, 35, 27]
#     women_means = [25, 32, 34, 20, 25]
#     name2count1 = dict()
#     name2count2 = dict()
#     for i, value in enumerate(men_means):
#         name2count1[labels[i]] = value
#         name2count2[labels[i]] = women_means[i]
#     # bar_plot_pair(name2count1, name2count2, 'test.png', 20)
#     # bar_plot(name2count1, 'test2.png', 20)
#     pie_plot(name2count1, 'test3.png')
