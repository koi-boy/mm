#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
from collections import Counter
import matplotlib.pyplot as plt


if __name__ == '__main__':

    root = '/Users/dyy/Desktop/chongqing/round2'
    img_dir = os.path.join(root, 'images')
    json_file = os.path.join(root, 'annotations.json')

    with open(json_file) as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    class_d = {}
    for cate in categories:
        id = cate['id']
        name = cate['name']
        class_d[id] = name
    print(class_d)

    bbox_lst, m_lst = [], []
    for anno in annotations:
        x, y, w, h = anno['bbox']
        category = anno['category_id']
        if category != 0:
            bbox_lst.append(category)
            m_lst.append(min(w, h))

    ############################################################
    counter = dict(Counter(bbox_lst))
    # print(counter)
    x = list(counter.keys())
    y = list(counter.values())
    plt.bar(x, y)
    for a, b in zip(x, y):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    plt.xlabel('category')
    plt.ylabel('count')
    plt.title('defect category count')
    plt.show()

    ############################################################
    c0, c1, c2, c3 = 0, 0, 0, 0
    for m in m_lst:
        if m < 40:
            c0 += 1
        elif 40 <= m < 120:
            c1 += 1
        elif 120 <= m < 420:
            c2 += 1
        else:
            c3 += 1
    x = list(range(4))
    y = [c0, c1, c2, c3]
    plt.pie(y, labels=['m<40', '40<=m<120', '120<=m<420', 'm>=420'])
    plt.title('short edge of bbox')
    plt.show()












