#!/usr/bin/env python
# encoding:utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software:
@file: code_rule.py
@time: 2019/9/6 16:23
@desc:
"""
import pandas as pd
import json

from configs.global_info import main_info


# ok_code = 'False'
# others_code = 'Others'


class CodeRule(object):

    def __init__(self, rule_method='MaxConf_Prior', code_prior_list=None,
                 alpha=0.9, thr_ok=0.05, thr_others=0.6, model_categories=None):
        """

        :param rule_method:
        :param code_prior_list:
        :param alpha:
        :param thr_ok:
        :param thr_others:
        :param model_categories:
        """
        assert rule_method in ('MaxConf', 'Prior', 'MaxConf_Prior'), 'rule method must be one of' \
                                                                     ' [MaxConf, Prior and MaxConf_Prior]'
        assert (rule_method == 'MaxConf') or (((rule_method == 'Prior') or (rule_method == 'MaxConf_Prior'))
                                              and (rule_method is not None))
        assert model_categories is not None, 'the model categories cannot be None'
        self.model_categories = model_categories

        self.rule_method = rule_method

        if code_prior_list is not None:
            not_exist_codes = [i for i in model_categories if i not in code_prior_list]
            assert len(not_exist_codes) == 0, \
                'the code {} not in prior list categories'.format(not_exist_codes)

        self.code_prior_list = code_prior_list

        self.alpha = alpha
        self.thr_ok = thr_ok
        self.thr_others = thr_others

    def get_main_code(self, img_result):
        """

        :return:
        """
        img_result_df = self.img_result_2_df(img_result)
        img_result_df = self.initial_filter(img_result_df)

        if self.rule_method == 'MaxConf':
            return self.__max_conf(img_result_df)
        elif self.rule_method == 'Prior':
            return self.__prior(img_result_df)
        elif self.rule_method == 'MaxConf_Prior':
            return self.__prior_with_conf_thr(img_result_df)
        else:
            raise Exception('Wrong rule method with {}'.format(self.rule_method))

    def img_result_2_df(self, img_result):
        """
        convert img result to data frame
        :param img_result:
        :return:
        """
        result_lst = []
        for idx, item in enumerate(img_result):
            code_id = idx + 1
            if not len(item) == 0:
                for i, bbox in enumerate(item):
                    bbox_id = i
                    xmin = bbox[0]
                    ymin = bbox[1]
                    xmax = bbox[2]
                    ymax = bbox[3]
                    conf = bbox[4]
                    width = xmax - xmin
                    height = ymax - ymin

                    result_lst.append(
                        {'code_name': self.model_categories[code_id - 1],
                         'bbox_id': bbox_id, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                         'width': width, 'height': height, 'area': width * height, 'confidence': conf})
        return pd.DataFrame(result_lst)

    def initial_filter(self, img_result_df):
        """

        :param img_result_df: data frame
        :return:
        """
        if len(img_result_df) == 0:
            return img_result_df

        # drop bbox with too low confidence
        img_result_df = img_result_df[img_result_df['confidence'] >= self.thr_ok]
        # change the code with confidence less than thr_others to Others
        img_result_df['code_name'] = \
            img_result_df.apply(
                lambda row: row['code_name'] if row['confidence'] >= self.thr_others else main_info.others_code, axis=1)

        return img_result_df

    @staticmethod
    def __max_conf(img_result_df):
        """
        get the image category by maximizing the confidence score
        :param img_result_df:
        :return:
        """
        max_conf = 0.0
        max_conf_category = None
        max_conf_bbox = None

        if len(img_result_df) > 0:
            max_idx = img_result_df.confidence.idxmax()
            max_conf = img_result_df.confidence[max_idx]
            max_conf_category = img_result_df.loc[max_idx, 'code_name']
            max_conf_bbox = [img_result_df.loc[max_idx, 'xmin'], img_result_df.loc[max_idx, 'ymin'],
                             img_result_df.loc[max_idx, 'xmax'], img_result_df.loc[max_idx, 'ymax']]

        return max_conf_category, max_conf, max_conf_bbox

    def __prior(self, img_result_df):
        """
        get the image category by the code prior
        :return:
        """

        max_prior_category = None
        max_prior_bbox = None
        max_conf = 0.0

        code_idx_lst = []
        for i, row in img_result_df.iterrows():
            code = row['code_name']
            code_idx = self.code_prior_list.index(code)
            code_idx_lst.append(code_idx)
        if len(code_idx_lst) > 0:
            final_code_idx = min(code_idx_lst)
            max_prior_category = self.code_prior_list[final_code_idx]
            max_conf = max(img_result_df.loc[img_result_df.code_name == max_prior_category, 'confidence'].values)
            select_df = img_result_df[(img_result_df['code_name'] == max_prior_category)
                                      & (img_result_df['confidence'] == max_conf)]
            max_prior_bbox = [select_df.iloc[0]['xmin'], select_df.iloc[0]['ymin'],
                              select_df.iloc[0]['xmax'], select_df.iloc[0]['ymax']]

        return max_prior_category, max_conf, max_prior_bbox

    def __prior_with_conf_thr(self, img_result_df):
        """

        :param img_result_df:
        :return:
        """
        out_conf = 0.0
        out_category = None
        out_bbox = None

        if len(img_result_df) > 0:
            max_idx = img_result_df.confidence.idxmax()
            max_conf = img_result_df.confidence[max_idx]
            predict = img_result_df.loc[max_idx, 'code_name']

            for i, row in img_result_df.iterrows():
                predict_idx = self.code_prior_list.index(predict)
                code = row['code_name']
                code_idx = self.code_prior_list.index(code)
                code_confidence = row['confidence']
                if code_idx < predict_idx and code_confidence > self.alpha * max_conf:
                    predict = code

            max_conf = max(img_result_df.loc[img_result_df.code_name == predict, 'confidence'].values)
            select_df = img_result_df[(img_result_df['code_name'] == predict)
                                      & (img_result_df['confidence'] == max_conf)]
            out_bbox = [select_df.iloc[0]['xmin'], select_df.iloc[0]['ymin'],
                        select_df.iloc[0]['xmax'], select_df.iloc[0]['ymax']]
            out_conf = max_conf
            out_category = predict

        return out_category, out_conf, out_bbox

    @staticmethod
    def load_model_categories_from_coco_json(coco_json_file):
        """

        :param coco_json_file:
        :return:
        """
        with open(coco_json_file, 'r') as f:
            load_dict = json.load(f)

        categories = load_dict['categories']
        id_name_list = []
        for category_info in categories:
            id_name_list.append((int(category_info['id']), category_info['name']))
        result = sorted(id_name_list, key=lambda x: x[0])
        model_categories = [x[1] for x in result]
        return model_categories

    @staticmethod
    def load_code_prior_list_from_file(code_prior_file):
        """

        :param code_prior_file:
        :return:
        """
        data = pd.read_csv(code_prior_file)
        code_prior_list = list(data.iloc[:, 0])
        return code_prior_list
