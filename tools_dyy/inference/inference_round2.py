#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import time
import glob
import json
import numpy as np
from mmdet.apis import inference_detector, init_detector


def pgps_model_test(imgs, cfg_file, ckpt_file,
                    score_thr=0.01, device='cuda:0'):
    model = init_detector(cfg_file, ckpt_file, device=device)
    img_dict, anno_dict = [], []
    for i, img in enumerate(imgs):  # loop for images
        st = time.time()
        img_name = img.split('/')[-1]
        img_id = i + 1
        img_dict.append({"file_name": img_name, "id": img_id})

        img_data = cv2.imread(img)
        h, w, _ = img_data.shape
        code = [1, 2, 3, 4, 5, 9, 10, 12, 13]
        if w < 1000:
            class_dict = [1, 2, 3, 4, 5, 9, 10]
        else:
            class_dict = [12, 13]

        result = inference_detector(model, img_data)
        total_bbox = []
        for idx, bboxes in enumerate(result):  # loop for categories
            category_id = code[idx]
            if len(bboxes) != 0 and category_id in class_dict:
                for bbox in bboxes:  # loop for bbox
                    conf = bbox[4]
                    if conf > score_thr:
                        total_bbox.append(list(bbox) + [category_id])

        for bbox in np.array(total_bbox):
            xmin, ymin, xmax, ymax = bbox[:4]
            coord = [xmin, ymin, xmax - xmin, ymax - ymin]
            coord = [round(x, 2) for x in coord]
            conf = bbox[4]
            category_id = int(bbox[5])
            anno_dict.append({'image_id': img_id, 'bbox': coord, 'category_id': category_id, 'score': conf})
        print(i, img_name, time.time() - st)
    return img_dict, anno_dict


def jiuye_model_test(imgs, cfg_file, ckpt_file, img_dict=[], anno_dict=[],
                     score_thr=0.01, device='cuda:0'):
    model = init_detector(cfg_file, ckpt_file, device=device)
    for i, img_path in enumerate(imgs):  # loop for images
        st = time.time()
        image = []
        for k in range(5):
            name = img_path.replace('_0.jpg', '_{}.jpg'.format(str(k)))
            data = cv2.imread(name, 0)[..., np.newaxis]
            image.append(data)
        concat_img = np.concatenate(image, axis=-1)
        result = inference_detector(model, concat_img)

        base_name = img_path.split('/')[-1].replace('_0.jpg', '')
        for idx, bboxes in enumerate(result):  # loop for categories
            start_id = len(img_dict)
            img_id = start_id + i*5 + idx + 1
            img_name = base_name + '_{}.jpg'.format(str(idx))
            img_dict.append({"file_name": img_name, "id": img_id})

            total_bbox = []
            if len(bboxes) != 0:
                for bbox in bboxes:  # loop for bbox
                    conf = bbox[4]
                    if conf > score_thr:
                        total_bbox.append(bbox)

            for bbox in np.array(total_bbox):
                xmin, ymin, xmax, ymax = bbox[:4]
                coord = [xmin, ymin, xmax - xmin, ymax - ymin]
                coord = [round(x, 2) for x in coord]
                conf = bbox[4]
                anno_dict.append({'image_id': img_id, 'bbox': coord, 'category_id': 11, 'score': conf})
        print(i, base_name, time.time() - st)
    return img_dict, anno_dict


if __name__ == '__main__':

    total = sorted(glob.glob('/tcdata/testA/images/*.jpg'))
    jiuye, pgps = [], []
    for img in total:
        img_name = img.split('/')[-1]
        if img_name.startswith('imgs'):
            if img_name.endswith('_0.jpg'):
                jiuye.append(img)
        else:
            pgps.append(img)

    pgps_cfg = 'xx.py'
    pgps_ckpt = 'xx.pth'
    jiuye_cfg = 'xx.py'
    jiuye_ckpt = 'xx.pth'

    img_dict, anno_dict = pgps_model_test(pgps, pgps_cfg, pgps_ckpt, score_thr=0.0)
    img_dict_2, anno_dict_2 = jiuye_model_test(jiuye, jiuye_cfg, jiuye_ckpt, img_dict, anno_dict, score_thr=0.0)
    predictions = {"images": img_dict_2, "annotations": anno_dict_2}
    with open('result.json', 'w') as f:
        json.dump(predictions, f, indent=4)
