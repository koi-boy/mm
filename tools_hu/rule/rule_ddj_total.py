import numpy as np
import os
import itertools
import pandas as pd
import json
import pickle
import cv2


def bboxes_iou(boxes1, boxes2):
    """
    boxes: [xmin, ymin, xmax, ymax, score, class] format coordinates.
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, 0.0)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    best_bboxes_idx = []

    while len(bboxes) > 0:
        max_ind = np.argmax(bboxes[:, 4])
        best_bbox = bboxes[max_ind]
        best_bboxes_idx.append(max_ind)

        bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
        iou = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
        weight = np.ones((len(iou),), dtype=np.float32)

        assert method in ['nms', 'soft-nms']
        if method == 'nms':
            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0
        if method == 'soft-nms':
            weight = np.exp(-(1.0 * iou ** 2 / sigma))

        bboxes[:, 4] = bboxes[:, 4] * weight
        score_mask = bboxes[:, 4] > 0.
        bboxes = bboxes[score_mask]

    return best_bboxes_idx


# def check_in(boxes1, boxes2, thr=0.9):
#     """
#     boxes: [xmin, ymin, xmax, ymax, score, class] format coordinates.
#     check if boxes2 in boxes1 using threshold thr.
#     """
#     boxes1 = np.array(boxes1)
#     boxes2 = np.array(boxes2)
#
#     boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
#     boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
#
#     left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
#     right_down = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])
#
#     inter_section = np.maximum(right_down - left_up, 0.0)
#     inter_area = inter_section[..., 0] * inter_section[..., 1]
#     if inter_area / boxes2_area >= thr:
#         return 2
#     elif inter_area / boxes1_area >= thr:
#         return 1
#     else:
#         return 0
#
#
# def check_in_filter(original_df, df, thr=0.9):
#     """
#     filter result df using check in function.
#     :param df:
#     :param thr: threshold
#     :return: filtered df
#     """
#     delect_bbox = []
#     for i in itertools.combinations(df.index, 2):
#         bbox1 = df.loc[i[0], 'bbox']
#         bbox2 = df.loc[i[1], 'bbox']
#         check = check_in(bbox1, bbox2, thr)
#         if check:
#             delect_bbox.append(i[check - 1])
#     delect_bbox = set(delect_bbox)
#     for bbox_idx in delect_bbox:
#         original_df.drop(index=bbox_idx, inplace=True)
#
#     return original_df

def check_concat(boxes1, boxes2, thr=0):
    """
    boxes: [xmin, ymin, xmax, ymax] format coordinates.
    check if boxes1 contact boxes2
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    if inter_area > thr:
        return True
    else:
        return False


def show_and_save_images(img_path, img_name, bboxes, codes, out_dir=None):
    img = cv2.imread(os.path.join(img_path, img_name))
    for i, bbox in enumerate(bboxes):
        bbox = np.array(bbox)
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        code = codes[i]
        label_txt = code + ': ' + str(round(bbox[4], 2))
        cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 1)
        cv2.putText(img, label_txt, (bbox_int[0], max(bbox_int[1] - 2, 0)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    if out_dir is not None:
        cv2.imwrite(os.path.join(out_dir, img_name), img)

    return img


def prio_check(prio_lst, code_lst):
    idx_lst = []
    for code in code_lst:
        assert code in prio_lst, '{} should be in priority file'.format(code)
        idx = prio_lst.index(code)
        idx_lst.append(idx)
    final_code = prio_lst[min(idx_lst)]
    return final_code


def filter_code(df, code, thr, replace=None):
    check_code = df[(df['category'] == code) & (df['score'] < thr)]
    if replace is None:
        df = df.drop(index=check_code.index)
    else:
        df.loc[check_code.index, 'category'] = replace
    return df


def model_test(result,
               img_name,
               codes,
               score_thr=0.05):
    output_bboxes = []
    json_dict = []
    pattern = img_name.split('_')[1][0]

    total_bbox = []
    for id, boxes in enumerate(result):  # loop for categories
        category_id = id + 1
        if len(boxes) != 0:
            for box in boxes:  # loop for bbox
                conf = box[4]
                if conf > score_thr:
                    total_bbox.append(list(box) + [category_id])

    bboxes = np.array(total_bbox)
    best_bboxes = bboxes
    output_bboxes.append(best_bboxes)
    for bbox in best_bboxes:
        coord = [round(i, 2) for i in bbox[:4]]
        conf, category = bbox[4], codes[int(bbox[5]) - 1]
        json_dict.append({'name': img_name, 'category': category, 'bbox': coord, 'score': conf, 'bbox_score': bbox[:5], 'pattern': pattern})

    return json_dict


def default_rule(result_lst, img_path_lst, img_name_lst, config, codes, draw_img=False, **kwargs):
    """
    :param det_lst: list, in order B, G, R, W
    :param img_path: list,
    :param img_name: list,
    :param size: float, size from .gls file
    :param json_config_file: str, config parameters in json format
    :param code_file: str, code file in txt format
    :param draw_img: Boolean,
    :return: main code, bbox, score, image
    """

    # get product id
    product = kwargs.get('product', None)

    df_lst = []
    for i in range(4):
        det_lst = result_lst[i]
        img_path = img_path_lst[i]
        img_name = img_name_lst[i]

        # convert list result to dict
        json_dict = model_test(det_lst, img_name, codes)
        det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score', 'pattern'])

        # prio parameters
        prio_weight = config['prio_weight']
        prio_lst = config['prio_order']
        if config['false_name'] not in prio_lst:
            prio_lst.append(config['false_name'])
        if config['other_name'] not in prio_lst:
            prio_lst.append(config['other_name'])

        # change other name using threshold
        det_df.loc[det_df['score'] < config['other_thr'], 'category'] = config['other_name']

        # filter pattern for r, g, b
        pattern = img_name.split('_')[1][0]
        if pattern == 'R':
            det_df = filter_code(det_df, 'V06-R', 0.7)
            det_df = filter_code(det_df, 'V06-G', 1.1)
            det_df = filter_code(det_df, 'V06-B', 1.1)
            det_df = filter_code(det_df, 'E07-R', 0.9)
            det_df = filter_code(det_df, 'E07-G', 1.1)
            det_df = filter_code(det_df, 'E07-B', 1.1)
            det_df = filter_code(det_df, 'E02-R', 0.8)
            det_df = filter_code(det_df, 'E02-G', 1.1)
            det_df = filter_code(det_df, 'E02-B', 1.1)
        if pattern == 'G':
            det_df = filter_code(det_df, 'V06-R', 1.1)
            det_df = filter_code(det_df, 'V06-G', 0.7)
            det_df = filter_code(det_df, 'V06-B', 1.1)
            det_df = filter_code(det_df, 'E07-R', 1.1)
            det_df = filter_code(det_df, 'E07-G', 0.9)
            det_df = filter_code(det_df, 'E07-B', 1.1)
            det_df = filter_code(det_df, 'E02-R', 1.1)
            det_df = filter_code(det_df, 'E02-G', 0.8)
            det_df = filter_code(det_df, 'E02-B', 1.1)
        if pattern == 'B':
            det_df = filter_code(det_df, 'V06-R', 1.1)
            det_df = filter_code(det_df, 'V06-G', 1.1)
            det_df = filter_code(det_df, 'V06-B', 0.7)
            det_df = filter_code(det_df, 'E07-R', 1.1)
            det_df = filter_code(det_df, 'E07-G', 1.1)
            det_df = filter_code(det_df, 'E07-B', 0.9)
            det_df = filter_code(det_df, 'E02-R', 1.1)
            det_df = filter_code(det_df, 'E02-G', 1.1)
            det_df = filter_code(det_df, 'E02-B', 0.8)
        if pattern == 'W':
            det_df = filter_code(det_df, 'V06-R', 0.3)
            det_df = filter_code(det_df, 'V06-G', 0.3)
            det_df = filter_code(det_df, 'V06-B', 0.3)
            det_df = filter_code(det_df, 'E07-R', 0.9)
            det_df = filter_code(det_df, 'E07-G', 0.9)
            det_df = filter_code(det_df, 'E07-B', 0.9)
            det_df = filter_code(det_df, 'E02-R', 0.8)
            det_df = filter_code(det_df, 'E02-G', 0.8)
            det_df = filter_code(det_df, 'E02-B', 0.8)

        # filtering
        det_df = filter_code(det_df, 'notch', 0.6)
        det_df = filter_code(det_df, 'L01', 0.5)
        det_df = filter_code(det_df, 'L02', 0.5)
        det_df = filter_code(det_df, 'L09', 0.5)
        det_df = filter_code(det_df, 'L10', 0.5)
        det_df = filter_code(det_df, 'V04', 0.8)
        det_df = filter_code(det_df, 'V01', 0.9)
        det_df = filter_code(det_df, 'V03', 0.5)
        det_df = filter_code(det_df, 'M07', 0.7)
        det_df = filter_code(det_df, 'M07-64', 0.6)
        det_df = filter_code(det_df, 'V99', 0.3)
        det_df = filter_code(det_df, 'M97', 0.3)

        # filter M97
        chip = img_name.split('_')[0]
        position = chip[-4:]
        if (position[:2] not in ('01', '05')) and (position[2:] not in ('01', '18')):
            det_df = filter_code(det_df, 'M97', 1.1)

        # filter V04
        if 'V04' in det_df['category']:
            det_df = filter_code(det_df, 'V01', 1.1)

        # judge C08
        if product == '639' and ('notch' in det_df['category'].values):
            notch_bbox = det_df.loc[det_df['category'] == 'notch', 'bbox'].values[0]
            for idx in det_df.index:
                cate = det_df.loc[idx, 'category']
                if cate in ('L01', 'L02', 'L09', 'L10'):
                    bbox = det_df.loc[idx, 'bbox']
                    if check_concat(bbox, notch_bbox):
                        det_df.loc[idx, 'category'] = 'C08'

        # judge L17
        cates = det_df['category'].values
        idx_lst = []
        if ('L01' in cates or 'L02' in cates) and ('L09' in cates or 'L10' in cates):
            for idx1 in det_df[(det_df['category'] == 'L01') | (det_df['category'] == 'L02')].index:
                for idx2 in det_df[(det_df['category'] == 'L09') | (det_df['category'] == 'L10')].index:
                    bbox1 = det_df.loc[idx1, 'bbox']
                    bbox2 = det_df.loc[idx2, 'bbox']
                    if check_concat(bbox1, bbox2):
                        idx_lst.append(idx1)
                        idx_lst.append(idx2)
        idx_lst = list(set(idx_lst))
        det_df.loc[idx_lst, 'category'] = 'L17'

        # delect notch
        det_df = filter_code(det_df, 'notch', 1.1)

        df_lst.append(det_df)

    final_det_df = pd.concat(df_lst)
    final_det_df.reset_index(inplace=True)

    # ET judge
    final_code = list(det_df['category'].values)
    best_bbox = list(det_df['bbox'].values)
    best_score = list(det_df['score'].values)

    # draw images
    if draw_img:
        img = show_and_save_images(img_path,
                                   img_name,
                                   det_df.bbox_score.values,
                                   det_df.category.values)
    else:
        img = None

    return final_code, best_bbox, best_score, img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    config_file = r'C:\Users\huker\Desktop\G6_21101-V1.0\G6_21101-V1.0.json'
    code_file = r'C:\Users\huker\Desktop\G6_21101-V1.0\G6_21101-V1.0.txt'
    img_path = r'C:\Users\huker\Desktop\G6_21101-V1.0'
    img_name = r'w97kp1222a0216_-142801_-511998_before.jpg'
    result_pkl = r'C:\Users\huker\Desktop\G6_21101-V1.0\21101_testimg.pkl'

    with open(result_pkl, 'rb') as f:
        result_lst = pickle.load(f)

    main_code, bbox, score, img = default_rule(result_lst, img_path, img_name, config_file, code_file)
    print(main_code, bbox, score)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    plt.imshow(img2)
    plt.show()
