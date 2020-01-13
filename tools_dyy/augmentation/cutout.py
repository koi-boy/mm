#! /usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os, json
import random, shutil
import numpy as np
from pascal_voc_io import PascalVocWriter

NUM_CLASSES = 20
HEIGHT = 1000
WIDTH = 2446

# according to previous map
# p = 1 / np.array([89, 48, 82, 64, 80, 73, 19, 75, 73, 34,
#                   39, 51, 27, 83, 57, 75, 51, 37, 57, 60])

# according to defects count
p = 1 / np.array([254, 306, 840, 1597, 113, 132, 152, 180, 320, 239,
                  128, 770, 354, 270, 377, 229, 451, 291, 378, 281])


def get_gt_boxes(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    img_names = []
    for img in images:
        img_names.append(img['file_name'])

    gt_dict = {}
    gt_lst = []
    for anno in annotations:
        img_id = anno['image_id']
        img_name = img_names[img_id - 1]
        cate_id = anno['category_id']
        bbox = anno['bbox']
        gt_dict.setdefault(img_name, []).append([cate_id] + bbox)
        gt_lst.append([img_name] + [cate_id] + bbox)

    print('\nNumber of Images: ', len(gt_dict))
    print('\nNumber of GroundTruths: ', len(gt_lst))
    return len(gt_dict), gt_dict, gt_lst



def cutout(gt_dict, gt_lst,
           defect_dir, normal_dir, save_dir, max_per_img=3, mix_ratio=0.):

    cate_lst = list(map(int, np.array(gt_lst)[..., 1]))
    p_lst = [p[cate-1] for cate in cate_lst]

    normal_imgs = os.listdir(normal_dir)
    for i, img_name in enumerate(normal_imgs):
        print(i, img_name)
        normal_img = cv2.imread(os.path.join(normal_dir, img_name))
        defects = random.choices(gt_lst, weights=p_lst,
                                k=random.randint(1, max_per_img))
        # first big, then small, to prevent big defects from covering small one
        defects = sorted(defects, key=(lambda x: x[-1]*x[-2]), reverse=True)
        if defects[0][-1]*defects[0][-2] > HEIGHT * WIDTH / 4:
            defects = [defects[0]]

        total_bbox = []
        for det in defects:
            defect_img = cv2.imread(os.path.join(defect_dir, det[0]))
            category = det[1]
            x, y, w, h = list(map(int, det[2:]))
            defect = defect_img[y:y+h, x:x+w, :]

            xmin = random.randint(0, WIDTH - w)
            ymin = random.randint(0, HEIGHT - h)
            xmax = xmin + w
            ymax = ymin + h
            normal_img[ymin:ymax, xmin:xmax, :] = \
                mix_ratio * normal_img[ymin:ymax, xmin:xmax, :] \
                + (1 - mix_ratio) * defect

            total_bbox.append([category] + [xmin,ymin,w,h])
        gt_dict[img_name] = total_bbox

        cv2.imwrite(os.path.join(save_dir, img_name), normal_img)
    return gt_dict


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
    new_json = 'C:\\Users\\EDZ\\Desktop\\cutout_train.json'
    save_dir = 'C:\\Users\\EDZ\\Desktop\\cutout'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    num_raw_imgs, gt_dict, gt_lst = get_gt_boxes(gt_json)
    print("\nStarting cutout...")
    gt_dict = cutout(gt_dict, gt_lst, defect_dir,
                     normal_dir, save_dir, max_per_img=5, mix_ratio=0.)

    print("\nWriting new train json...")
    write_new_json(gt_dict, new_json)

    print("\nWriting cutout xmls...")
    xml_dir = 'C:\\Users\\EDZ\\Desktop\\cutout_xmls'
    if os.path.exists(xml_dir):
        shutil.rmtree(xml_dir)
    os.makedirs(xml_dir)
    write_xmls(save_dir, gt_dict, num_raw_imgs, xml_dir)


