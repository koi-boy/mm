#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import time
import glob
import json
import numpy as np
from mmdet.apis import inference_detector, init_detector
from progressbar import ProgressBar


def model_test(imgs, pg_cfg, pg_ckpt, ps_cfg, ps_ckpt,
               score_thr=0.01, device='cuda:0', for_submit=False):
    pg_model = init_detector(pg_cfg, pg_ckpt, device=device)
    ps_model = init_detector(ps_cfg, ps_ckpt, device=device)

    pbar = ProgressBar().start()
    img_dict, anno_dict = [], []
    for i, img in enumerate(imgs):  # loop for images
        # st = time.time()
        pbar.update(int(i / (len(imgs) - 1) * 100))

        img_data = cv2.imread(img)
        h, w, _ = img_data.shape
        img_name = img.split('/')[-1]
        img_id = i + 1
        if for_submit:
            img_dict.append({"file_name": img_name, "id": img_id})
        else:
            img_dict.append({"file_name": img_name, "id": img_id, "height": h, "width": w})

        if w < 1000:
            result = inference_detector(pg_model, img_data)
            class_dict = [1, 2, 3, 4, 5, 9, 10]
        else:
            result = inference_detector(ps_model, img_data)
            class_dict = [6, 7, 8]

        total_bbox = []
        for idx, bboxes in enumerate(result):  # loop for categories
            category_id = class_dict[idx]
            if len(bboxes) != 0:
                for bbox in bboxes:  # loop for bbox
                    conf = bbox[4]
                    if conf > score_thr:
                        total_bbox.append(list(bbox) + [category_id])

        for bbox in np.array(total_bbox):
            xmin, ymin, xmax, ymax = bbox[:4]
            coord = [xmin, ymin, xmax - xmin, ymax - ymin]
            coord = [round(x, 2) for x in coord]
            conf = round(bbox[4], 4)
            category_id = int(bbox[5])
            anno_dict.append({'image_id': img_id, 'bbox': coord, 'category_id': category_id, 'score': conf})
        # print(i, img_name, w, time.time() - st)
    pbar.finish()
    return img_dict, anno_dict


if __name__ == '__main__':
    imgs = glob.glob('../a/CQ/val_new/*.jpg')
    pg_cfg = 'cfg/pg_libra_cascade_r50.py'
    pg_ckpt = '../a/CQ_work_dirs/pg_libra_cascade_r50/epoch_21.pth'
    ps_cfg = 'cfg/ps_cascade_se_r50.py'
    ps_ckpt = '../a/CQ_work_dirs/ps_cascade_se_r50/epoch_22.pth'

    img_dict, anno_dict = model_test(imgs, pg_cfg, pg_ckpt, ps_cfg, ps_ckpt, score_thr=0.01, for_submit=True)
    predictions = {"images": img_dict, "annotations": anno_dict}
    with open('pg_libra_cascade.json', 'w') as f:
        json.dump(predictions, f, indent=4)
