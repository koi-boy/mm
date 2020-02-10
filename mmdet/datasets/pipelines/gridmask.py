import numpy as np
from PIL import Image
from ..registry import PIPELINES


@PIPELINES.register_module
class GridMask(object):
    """Apply GridMask Data Augmentation to image."""

    def __init__(self, use_h, use_w, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def grid(self, img):
        h, w, _ = img.shape
        d1 = 2
        d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(d1, d2)
        # d = self.d
        # self.l = int(d*self.ratio+0.5)
        if self.ratio == 1:
            l = np.random.randint(1, d)
        else:
            l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        # mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]
        if self.mode == 1:
            mask = 1 - mask

        mask = mask[..., np.newaxis]
        mask = np.tile(mask, [1, 1, 3])
        img = img * mask
        return img

    def __call__(self, results):
        if 'grid' not in results:
            grid = True if np.random.rand() < self.prob else False
            results['grid'] = grid
        if results['grid']:
            results['img'] = self.grid(results['img'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(use_h={}, use_w={}, rotate={}, '
                     'ratio={}, mode={}, prob={})').format(
            self.use_h, self.use_w, self.rotate,
            self.ratio, self.mode, self.prob)
        return repr_str
