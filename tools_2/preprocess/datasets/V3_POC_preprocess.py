#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@l@icense: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: liusili@unionbigdata.com
@software:
@file: V3_POC_preprocess
@time: 2019/10/25
@desc:
"""
import os
import shutil
from tqdm import tqdm
sample_root = '/home/Visionox/V2/12620V3'
key_code = ['A2DMR', 'A2SIP', 'A2WBD']
poc_data_dir = '/home/Visionox/V3/poc_data'
if not os.path.exists(poc_data_dir):
    os.makedirs(poc_data_dir)

pbar = tqdm(os.listdir(sample_root))
for code_dir in pbar:
    if code_dir.split('_')[0] in key_code:
        print(code_dir)
        old_code_dir = os.path.join(sample_root,code_dir)
        new_code_dir = os.path.join(poc_data_dir,code_dir)
        shutil.copytree(old_code_dir, new_code_dir)
    pbar.set_description('Processing code:{}'.format(code_dir))
pass