#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os, json
import random, shutil
import numpy as np
from collections import Counter
from pascal_voc_io import PascalVocWriter

NUM_CLASSES = 20
HEIGHT = 1000
WIDTH = 2446

"""
p = np.ones(NUM_CLASSES)
p[[9, 10, 14]] += 1
p[[6, 12, 17]] += 2
"""
p = 1 / np.array([89, 48, 82, 64, 80, 73, 19, 75, 73, 34,
                  39, 51, 27, 83, 57, 75, 51, 37, 57, 60])


def get_gt_boxes(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    img_names = []
    for img in images:
        img_names.append(img['file_name'])

    gt_dict = {}
    cate_lst = []
    for anno in annotations:
        img_id = anno['image_id']
        img_name = img_names[img_id-1]
        cate_id = anno['category_id']
        bbox = anno['bbox']
        gt_dict.setdefault(img_name, []).append([cate_id] + bbox)
        cate_lst.append(cate_id)
    print('\nNumber of images: ', len(gt_dict))

    cate_count = dict(Counter(cate_lst))
    cate_sorted = sorted(cate_count.items(), key=lambda x: x[1], reverse=True)
    print('\nGroundTruth Count: ', cate_sorted)
    return gt_dict, len(gt_dict)


def defect_sampler(gt_dict, defect_dir, save_dir, alpha=2, iters=10):
    new_dict = gt_dict.copy()

    img_d = {}
    for k, v in gt_dict.items():
        cls = np.array(v)[..., 0]
        prob = [p[int(id)-1] for id in cls]
        img_d[k] = max(prob)

    print("\nStarting Defect Images' Mixup ...")
    i = 0
    while i < iters:
        print("Iter: ", i)
        imgs = random.choices(list(img_d.keys()), weights=list(img_d.values()), k=2)
        if len(set(imgs)) != 1:
            img_name1 = imgs[0]
            img_name2 = imgs[1]
            img1 = cv2.imread(os.path.join(defect_dir, img_name1))
            img2 = cv2.imread(os.path.join(defect_dir, img_name2))
            if alpha != None:
                lam = np.random.beta(alpha, alpha)
                img = (lam*img1 + (1-lam)*img2).astype(int)
            else:
                img = (0.5*img1 + 0.5*img2).astype(int)
            img_name = img_name1[:-4] + '_' + img_name2
            cv2.imwrite(os.path.join(save_dir, img_name), img)

            bbox1 = gt_dict[img_name1]
            bbox2 = gt_dict[img_name2]
            total_bbox = bbox1 + bbox2
            new_dict[img_name] = total_bbox
            i += 1
    print("Number of images after Defect Images' Mixup: ", len(new_dict))
    return new_dict


def normal_sampler(gt_dict, gt_dict2, defect_dir, normal_dir, save_dir, alpha=2, iters=10):
    new_dict = gt_dict2.copy()

    img_d = {}
    for k, v in gt_dict.items():
        cls = np.array(v)[..., 0]
        prob = [p[int(id) - 1] for id in cls]
        img_d[k] = max(prob)

    normal_imgs = os.listdir(normal_dir)

    print("\nStarting Defect & Normal Mixup ...")
    i = 0
    while i < iters:
        print("Iter: ", i)
        defect_img = random.choices(list(img_d.keys()), weights=list(img_d.values()), k=1)[0]
        normal_img = random.choice(normal_imgs)
        defect = cv2.imread(os.path.join(defect_dir, defect_img))
        normal = cv2.imread(os.path.join(normal_dir, normal_img))
        if alpha != None:
            lam = np.random.beta(alpha, alpha)
            img = (lam*defect + (1-lam)*normal).astype(int)
        else:
            img = (0.5*defect + 0.5*normal).astype(int)
        img_name = defect_img[:-4] + '_' + normal_img
        cv2.imwrite(os.path.join(save_dir, img_name), img)

        new_dict[img_name] = gt_dict[defect_img]
        i += 1
    print("Number of images after Defect & Normal Mixup: ", len(new_dict))
    return new_dict


def normal(gt_dict, defect_dir, normal_dir, save_dir, alpha=2):
    new_dict = gt_dict.copy()

    img_d = {}
    for k, v in gt_dict.items():
        cls = np.array(v)[..., 0]
        prob = [p[int(id) - 1] for id in cls]
        img_d[k] = max(prob)

    print("\nStarting Defect & Normal Mixup ...")

    normal_imgs = os.listdir(normal_dir)[:100]
    for i, normal_img in enumerate(normal_imgs):
        print(i, normal_img)
        defect_img = random.choices(list(img_d.keys()), weights=list(img_d.values()), k=1)[0]
        defect = cv2.imread(os.path.join(defect_dir, defect_img))
        normal = cv2.imread(os.path.join(normal_dir, normal_img))
        if alpha != None:
            lam = np.random.beta(alpha, alpha)
            img = (lam*defect + (1-lam)*normal).astype(int)
        else:
            img = (0.5*defect + 0.5*normal).astype(int)
        img_name = normal_img[:-4] + '_' + defect_img
        cv2.imwrite(os.path.join(save_dir, img_name), img)

        new_dict[img_name] = gt_dict[defect_img]
        i += 1
    print("Number of images after Defect & Normal Mixup: ", len(new_dict))
    return new_dict


def write_new_json(gt_dict, new_json):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    bnd_id = 1
    for i, (name, anno) in enumerate(gt_dict.items()):
        img_id = i+1
        image = {'file_name': name, 'height': HEIGHT, 'width': WIDTH, 'id': img_id}
        json_dict['images'].append(image)

        for box in anno:
            ann = {'bbox': box[1:], 'category_id': box[0], 'image_id': img_id,
                   'iscrowd': 0, 'ignore': 0, 'area': box[3]*box[4], 'segmentation': [], 'id': bnd_id}
            json_dict['annotations'].append(ann)
            bnd_id += 1

    for i in range(NUM_CLASSES):
        cat = {'supercategory': 'none', 'id': i+1, 'name': str(i+1)}
        json_dict['categories'].append(cat)

    with open(new_json, 'w') as f:
        json.dump(json_dict, f, indent=4)


def write_xmls(foldername, gt_dict, num_raw_imgs, save_dir):
    imgs = list(gt_dict.keys())
    annotations = list(gt_dict.values())
    imgSize = [HEIGHT, WIDTH, 3]
    for i in range(num_raw_imgs, len(imgs)):
        filename = imgs[i]
        # print(i, filename)
        localImgPath = os.path.join(foldername, filename)
        XMLWriter = PascalVocWriter(foldername, filename, imgSize, localImgPath)
        for box in annotations[i]:
            XMLWriter.addBndBox(box[1], box[2], box[1]+box[3], box[2]+box[4], str(box[0]))
        XMLWriter.save(os.path.join(save_dir, filename[:-4]+'.xml'))


if __name__ == '__main__':
    gt_json = 'C:\\Users\\EDZ\\Desktop\\train.json'
    defect_dir = 'C:\\Users\\EDZ\\Desktop\\train'
    normal_dir = 'C:\\Users\\EDZ\\Desktop\\normal_Images'
    new_json = 'C:\\Users\\EDZ\\Desktop\\mixup_train.json'
    save_dir = 'C:\\Users\\EDZ\\Desktop\\mixup'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    gt_dict, num_raw_imgs = get_gt_boxes(gt_json)
    gt_dict2 = defect_sampler(gt_dict, defect_dir, save_dir, iters=2000)
    gt_dict3 = normal_sampler(gt_dict, gt_dict2, defect_dir, normal_dir, save_dir, iters=2000)

    print("\nWriting new train json...")
    write_new_json(gt_dict3, new_json)

    print("\nWriting mixup xmls...")
    xml_dir = 'C:\\Users\\EDZ\\Desktop\\mixup_xmls'
    if os.path.exists(xml_dir):
        shutil.rmtree(xml_dir)
    os.makedirs(xml_dir)
    write_xmls(save_dir, gt_dict3, num_raw_imgs, xml_dir)