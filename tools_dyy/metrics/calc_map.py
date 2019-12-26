#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from Evaluator import *
import matplotlib.pyplot as plt


def get_bbox(json_file, is_gt):
    with open(json_file, 'r') as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']

    img_dict = {}
    for img in images:
        img_name = img['file_name']
        img_id = img['id']
        img_dict[img_id] = img_name

    bbox_lst = []
    for anno in annotations:
        img_id = anno['image_id']
        img_name = img_dict[img_id]
        category_id = anno['category_id']
        bbox = anno['bbox']
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        if is_gt:
            bbox_lst.append([img_name, category_id, 1, bbox])
        else:
            score = anno['score']
            bbox_lst.append([img_name, category_id, score, bbox])
    return bbox_lst


if __name__ == '__main__':
    root = r'C:\Users\EDZ\Desktop\chongqing1_round1_train1_20191223'
    gt_json = os.path.join(root, r'train_test\test.json')
    det_json = os.path.join(root, r'model_predict\result_keep15.json')
    gt_lst = get_bbox(gt_json, is_gt=True)
    det_lst = get_bbox(det_json, is_gt=False)

    evaluator = Evaluator()
    ret, mAP = evaluator.GetPascalVOCMetrics(
        gt_lst,
        det_lst,
        method='EveryPointInterpolation'
    )
    for metricsPerClass in ret:
        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']
        if totalPositives > 0:
            ap_str = "{0:.2f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('\nmAP: %s' % mAP_str)

    save_dir = os.path.join(root, r'model_predict\plot')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    evaluator.PlotPrecisionRecallCurve(
        gt_lst,
        det_lst,
        method='EveryPointInterpolation',
        showAP=True,
        showInterpolatedPrecision=False,
        savePath=save_dir,
        showGraphic=False
    )
    # x = list(range(2000))
    # y = [IOU_THRESHOLD(i) for i in x]
    # plt.plot(x, y)
    # plt.show()


