import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head import AnchorHead
from mmcv.cnn import xavier_init


@HEADS.register_module
class EfficientDetHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(EfficientDetHead, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Sequential(
                ConvModule(
                    chn,
                    chn,
                    3,
                    stride=1,
                    padding=1,
                    groups=chn,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg),
                ConvModule(
                    chn,
                    self.feat_channels,
                    1,
                    stride=1,
                    padding=1,
                    groups=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg
                ))
            )
            self.reg_convs.append(
                nn.Sequential(
                    ConvModule(
                        chn,
                        chn,
                        3,
                        stride=1,
                        padding=1,
                        groups=chn,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg),
                    ConvModule(
                        chn,
                        self.feat_channels,
                        1,
                        stride=1,
                        padding=1,
                        groups=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg
                    )))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
