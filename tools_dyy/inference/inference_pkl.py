#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import pickle
import glob
import json
import numpy as np


def model_test(imgs, pkl_file, score_thr=0.1):
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    assert len(imgs) == len(results)

    img_dict, anno_dict = [], []
    for i, img in enumerate(imgs):  # loop for images
        img_name = img.split('/')[-1]
        img_id = i + 1
        img_dict.append({"file_name": img_name, "id": img_id})

        img_data = cv2.imread(img)
        h, w, _ = img_data.shape
        if w < 1000:
            class_dict = [1, 2, 3, 4, 5, 9, 10]
        else:
            class_dict = [6, 7, 8]

        result = results[i]
        total_bbox = []
        for idx, bboxes in enumerate(result):  # loop for categories
            category_id = idx + 1
            if len(bboxes) != 0 and category_id in class_dict:
                for bbox in bboxes:  # loop for bbox
                    conf = bbox[4]
                    if conf > score_thr:
                        total_bbox.append(list(bbox) + [category_id])

        for bbox in np.array(total_bbox):
            xmin, ymin, xmax, ymax = bbox[:4]
            coord = [xmin, ymin, xmax - xmin, ymax - ymin]
            coord = [round(x, 2) for x in coord]
            conf = round(bbox[4], 2)
            category_id = int(bbox[5])
            anno_dict.append({'image_id': img_id, 'bbox': coord, 'category_id': category_id, 'score': conf})

    return img_dict, anno_dict


if __name__ == '__main__':
    imgs = sorted(glob.glob('/data/sdv1/a/testA/*.jpg'))
    pkl_file = 'result.pkl'

    img_dict, anno_dict = model_test(imgs, pkl_file, score_thr=0.01)
    predictions = {"images": img_dict, "annotations": anno_dict}
    with open('output.json', 'w') as f:
        json.dump(predictions, f, indent=4)
