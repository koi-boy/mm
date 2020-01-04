from ..registry import PIPELINES
import mmcv
import numpy as np
import cv2
from glob import glob


@PIPELINES.register_module
class Cutout(object):
    """Matting defect bbox to normal image.
    Args:
        normal_path: normal images path
    """
    def __init__(self, normal_path, cutout_ratio=0.5, mix_range=(0.5, 1)):
        self.normal_path = normal_path
        self.cutout_ratio = cutout_ratio
        self.mix_lower, self.mix_upper = mix_range
        self.normal_imgs = glob(normal_path + '*.jpg')

    def __call__(self, results):
        cutout = True if np.random.rand() < self.cutout_ratio else False
        if cutout:
            inx = np.random.randint(len(self.normal_imgs))
            normal_name = self.normal_imgs[inx]
            normal_img = mmcv.imread(normal_name)
            normal_img = mmcv.imresize_like(normal_img, results['img'])
            for bbox in results['gt_bboxes']:
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2])
                ymax = int(bbox[3])
                mix_ratio = np.random.uniform(self.mix_lower, self.mix_upper)
                normal_img[ymin:ymax, xmin:xmax, :] = cv2.addWeighted(results['img'][ymin:ymax, xmin:xmax, :], mix_ratio,
                                                                      normal_img[ymin:ymax, xmin:xmax, :], 1-mix_ratio, 0)
            results['img'] = normal_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(normal_path={}, cutout_ratio={}, mix_range={}-{})'.format(
            self.normal_path, self.cutout_ratio, self.mix_lower, self.mix_upper)
        return repr_str