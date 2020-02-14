import os
import cv2
import numpy as np
from ..registry import PIPELINES


@PIPELINES.register_module
class Concat(object):
    """Concat five images.
    Args:
        img_path (str): images should be placed in one path.
    """

    def __init__(self, img_path):
        self.img_path = img_path

    def __call__(self, results):
        filename = results['img_info']['filename']
        image = []
        for i in range(5):
            img_name = filename.replace('_0.jpg', '_{}.jpg'.format(str(i)))
            img = cv2.imread(os.path.join(self.img_path, img_name), 0)
            img = img[..., np.newaxis]
            image.append(img)
        concat = np.concatenate(image, axis=-1)
        results['img'] = concat
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(img_path={})'.format(
            self.img_path)
        return repr_str
