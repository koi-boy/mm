# model settings
model = dict(
    type='EfficientDet',
    pretrained=None,
    backbone=dict(
        type='EfficientNet',
        modelname='efficientnet-b4',
        out_indices=(1, 2, 3),
        frozen_stages=-1),
    neck=dict(
        type='BIFPN',
        in_channels=[56, 160, 448],
        out_channels=224,
        start_level=0,
        add_extra_convs_before_bifpn=True,
        num_outs=5,
        stack=6,
        norm_cfg={'type': 'BN'},
        activation='relu'),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=81,
        in_channels=224,
        stacked_convs=4,
        feat_channels=224,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.2, 0.5, 1.0, 2.0, 5.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='BalancedL1Loss',
            alpha=0.5,
            gamma=1.5,
            beta=1.0,
            loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.3),
    max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/sdv1/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment', augmentation_name='v0',
                  cutout_max_pad_fraction=0.25,
                  cutout_bbox_replace_with_mean=False,
                  cutout_const=20,
                  translate_const=50,
                  cutout_bbox_const=10,
                  translate_bbox_const=10),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=128),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.00004)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=30000,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=3)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../a/CQ_work_dirs/test_efficientdet_0204'
load_from = None
resume_from = None
workflow = [('train', 1)]
