import cv2
from PIL import Image, ImageEnhance, ImageFilter
from skimage.util import random_noise

import mmcv
import numpy as np
from imagecorruptions import corrupt
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES


@PIPELINES.register_module
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = masks

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        w = img_shape[1]
        flipped = bboxes.copy()
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)
            
            
@PIPELINES.register_module
class RandomVerticalFlip(object):
    """Vertical Flip the image & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes vertically.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        h = img_shape[0]
        flipped = bboxes.copy()
        flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
        flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'], direction='vertical')
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class BBoxJitter(object):
    """
    bbox jitter
    Args:
        min (int, optional): min scale
        max (int, optional): max scale
        ## origin w scale
    """

    def __init__(self, min=0, max=2):
        self.min_scale = min
        self.max_scale = max
        self.count = 0

    def bbox_jitter(self, bboxes, img_shape):
        """
        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        if len(bboxes) == 0:
            return bboxes
        jitter_bboxes = []
        for box in bboxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            scale = np.random.uniform(self.min_scale, self.max_scale)
            w = w * scale / 2.
            h = h * scale / 2.
            xmin = center_x - w
            ymin = center_y - h
            xmax = center_x + w
            ymax = center_y + h
            box2 = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            jitter_bboxes.append(box2)
        jitter_bboxes = np.array(jitter_bboxes, dtype=np.float32)
        jitter_bboxes[:, 0::2] = np.clip(jitter_bboxes[:, 0::2], 0, img_shape[1] - 1)
        jitter_bboxes[:, 1::2] = np.clip(jitter_bboxes[:, 1::2], 0, img_shape[0] - 1)
        return jitter_bboxes

    def __call__(self, results):
        for key in results.get('bbox_fields', []):
            results[key] = self.bbox_jitter(results[key],
                                          results['img_shape'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(bbox_jitter={}-{})'.format(
            self.min_scale, self.max_scale)
            
            
@PIPELINES.register_module
class RandomRotate(object):
    """Rotate the image & bbox.

    If the input dict contains the key "rotate", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        rotate_ratio (float, optional): The rotating probability.
    """

    def __init__(self, angle=180, rotate_ratio=None):
        assert angle == 180, 'Only 180 degrees supported now.'
        self.angle = angle
        self.rotate_ratio = rotate_ratio
        if rotate_ratio is not None:
            assert rotate_ratio >= 0 and rotate_ratio <= 1
    
    def get_corners(self, bboxes):
      """Get corners of bounding boxes.
      
      Args:
          bboxes(ndarray): (..., 4*k)
      
      Returns:
          Four corners(ndarray): (..., 8*k)
      """
      assert bboxes.shape[-1] % 4 == 0
      width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
      height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)
      x1 = bboxes[:, 0].reshape(-1, 1)
      y1 = bboxes[:,1].reshape(-1, 1)
      x2 = x1 + width
      y2 = y1 
      x3 = x1
      y3 = y1 + height
      x4 = bboxes[:, 2].reshape(-1, 1)
      y4 = bboxes[:, 3].reshape(-1, 1)
      corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
      return corners
      
    def bbox_rotate(self, corners, img_shape):
        """Rotate bboxes.

        Args:
            corners(ndarray): (..., 8*k)
            img_shape(tuple): (height, width)
        
        Returns:
            Four corners of rotated bounding boxes.
        """
        corners = corners.reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
        h, w = img_shape[0:2]
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        M = cv2.getRotationMatrix2D((cx, cy), -self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = h * sin + w * cos
        nH = h * cos + w * sin
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW - w) * 0.5
        M[1, 2] += (nH - h) * 0.5
        # Prepare the vector to be transformed
        calculated = np.dot(M, corners.T).T
        calculated = calculated.reshape(-1, 8)
        return calculated

    def bbox_clip(self, bboxes, clip_box, alpha=0.25):
        """Clip the bounding boxes to the borders of an image.

        Args:
            bboxes(ndarray): (..., 4*k)
            clip_box(ndarray):
                An array of shape (4,) specifying the diagonal co-ordinates of the image.
            alpha(float):
                If the fraction of a bounding box left in the image after being clipped is 
                less than `alpha` the bounding box is dropped.
        
        Returns:
            Bboxes left after being clipped.
        """
        def bbox_area(bboxes):
            return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        area = bbox_area(bboxes)
        xmin = np.maximum(bboxes[:, 0], clip_box[0]).reshape(-1, 1)
        ymin = np.maximum(bboxes[:, 1], clip_box[1]).reshape(-1, 1)
        xmax = np.minimum(bboxes[:, 2], clip_box[2]).reshape(-1, 1)
        ymax = np.minimum(bboxes[:, 3], clip_box[3]).reshape(-1, 1)
        bboxes = np.hstack((xmin, ymin, xmax, ymax, bboxes[:, 4:]))
        delta_area = (area - bbox_area(bboxes)) / area
        mask = (delta_area < (1 - alpha)).astype(int)
        bboxes = bboxes[mask == 1, :]
        return bboxes
        
    def get_enclosing_box(self, corners):
        """Get an enclosing box for ratated corners of a bounding box.
        
        Args:
            corners(ndarray): (..., 8*k)

        Returns 
            bboxes(ndarray): (..., 4*k)
        """
        x_ = corners[:, [0,2,4,6]]
        y_ = corners[:, [1,3,5,7]]
        xmin = np.min(x_, 1).reshape(-1, 1)
        ymin = np.min(y_, 1).reshape(-1, 1)
        xmax = np.max(x_, 1).reshape(-1, 1)
        ymax = np.max(y_, 1).reshape(-1, 1)  
        bboxes = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))
        return bboxes

    def __call__(self, results):
        if 'rotate' not in results:
            rotate = True if np.random.rand() < self.rotate_ratio else False
            results['rotate'] = rotate
        if results['rotate']:
            # rotate image
            image = results['img']
            h, w = image.shape[:2]
            rotated_image = mmcv.imrotate(image, angle=self.angle)
            results['img'] = rotated_image
            # rotate bboxes
            for key in results.get('bbox_fields', []):
                bboxes = results[key]
                corners = self.get_corners(bboxes)
                corners = np.hstack((corners, bboxes[:, 4:]))
                corners[:, :8] = self.bbox_rotate(corners[:, :8],
                                                  results['img_shape'])
                new_bboxes = self.get_enclosing_box(corners)
                # new_bboxes = self.bbox_clip(new_bboxes, [0, 0, w, h])
                results[key] = new_bboxes
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(angle={}, rotate_ratio={})'.format(
            self.angle, self.rotate_ratio)


@PIPELINES.register_module
class RandomColor(object):
    """Adjust image color balance.

    If the input dict contains the key "color", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        color_ratio (float, optional): The adjusting probability.
    """

    def __init__(self, color_range=(0.5, 1.5), color_ratio=None):
        self.color_lower, self.color_upper = color_range
        self.color_ratio = color_ratio
        if color_ratio is not None:
            assert color_ratio >= 0 and color_ratio <= 1

    def color(self, img):
        """Adjust image color.

        Args:
            image(ndarray): opencv type(bgr)
        """
        factor = random.uniform(self.color_lower, self.color_upper)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        enh_color = ImageEnhance.Color(image)
        image_enhanced = enh_color.enhance(factor)
        image_enhanced = np.asarray(image_enhanced)
        return image_enhanced[:,:,[2,1,0]]

    def __call__(self, results):
        if 'color' not in results:
            color = True if np.random.rand() < self.color_ratio else False
            results['color'] = color
        if results['color']:
            # adjust image color
            results['img'] = self.color(results['img'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(color_range={}, color_ratio={})'.format(
            self.color_range, self.color_ratio)


@PIPELINES.register_module
class RandomContrast(object):
    """Adjust contrast of image.

    If the input dict contains the key "contrast", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        contrast_ratio (float, optional): The adjusting probability.
    """

    def __init__(self, contrast_range=(0.5, 1.5), contrast_ratio=None):
        self.contrast_lower, self.contrast_upper = contrast_range
        self.contrast_ratio = contrast_ratio
        if contrast_ratio is not None:
            assert contrast_ratio >= 0 and contrast_ratio <= 1

    def contrast(self, img):
        """Adjust image contrast.

        Args:
            image(ndarray): opencv type(bgr)
        """
        factor = random.uniform(self.contrast_lower, self.contrast_upper)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        enh_contrast = ImageEnhance.Contrast(image)
        image_enhanced = enh_contrast.enhance(factor)
        image_enhanced = np.asarray(image_enhanced)
        return image_enhanced[:,:,[2,1,0]]

    def __call__(self, results):
        if 'contrast' not in results:
            contrast = True if np.random.rand() < self.contrast_ratio else False
            results['contrast'] = contrast
        if results['contrast']:
            # adjust image contrast
            results['img'] = self.contrast(results['img'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(contrast_range={}, contrast_ratio={})'.format(
            self.contrast_range, self.contrast_ratio)


@PIPELINES.register_module
class RandomBrightness(object):
    """Adjust brightness of image.

    If the input dict contains the key "brightness", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        brightness_ratio (float, optional): The adjusting probability.
    """

    def __init__(self, brightness_range=(0.5, 1.5), brightness_ratio=None):
        self.brightness_lower, self.brightness_upper = brightness_range
        self.brightness_ratio = brightness_ratio
        if brightness_ratio is not None:
            assert brightness_ratio >= 0 and brightness_ratio <= 1

    def brightness(self, img):
        """Adjust image brightness.

        Args:
            image(ndarray): opencv type(bgr)
        """
        factor = random.uniform(self.brightness_lower, self.brightness_upper)
        image = Image.fromarray(img)
        enh_brightness = ImageEnhance.Brightness(image)
        image_enhanced = enh_brightness.enhance(factor)
        image_enhanced = np.asarray(image_enhanced)
        return image_enhanced

    def __call__(self, results):
        if 'brightness' not in results:
            brightness = True if np.random.rand() < self.brightness_ratio else False
            results['brightness'] = brightness
        if results['brightness']:
            # adjust image brightness
            results['img'] = self.brightness(results['img'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(brightness_range={}, brightness_ratio={})'.format(
            self.brightness_range, self.brightness_ratio)


@PIPELINES.register_module
class RandomNoise(object):
    """Add noise to image.

    If the input dict contains the key "noise", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        noise_ratio (float, optional): The adjusting probability.
    """

    def __init__(self, noise_type="gaussian", noise_ratio=None):
        """
        noise_type = ["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"]
        """
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        if noise_ratio is not None:
            assert noise_ratio >= 0 and noise_ratio <= 1

    def add_noise(self, img):
        """Add noise to image.

        Args:
            image(ndarray): opencv type(bgr)
        """
        noised_image = (random_noise(img, mode=self.noise_type)*255).astype(np.uint8)
        return noised_image

    def __call__(self, results):
        if 'noise' not in results:
            noise = True if np.random.rand() < self.noise_ratio else False
            results['noise'] = noise
        if results['noise']:
            # add noise to image
            results['img'] = self.add_noise(results['img'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(noise_type={}, noise_ratio={})'.format(
            self.noise_type, self.noise_ratio)


@PIPELINES.register_module
class RandomFilter(object):
    """Image Filtering.

    If the input dict contains the key "blur", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        blur_ratio (float, optional): The adjusting probability.
    """

    def __init__(self, blur_type="gaussian", blur_ratio=None):
        """
        filter_type = ["original", "gaussian", "detail", "edge_enhance", "sharpen"]
        """
        self.blur_type = blur_type
        self.blur_ratio = blur_ratio
        if blur_ratio is not None:
            assert blur_ratio >= 0 and blur_ratio <= 1

    def filtering(self, img):
        """Image Filtering.

        Args:
            image(ndarray): opencv type(bgr)
        """
        image = Image.fromarray(img)
        if self.blur_type == "original":
            blured_image = image.filter(ImageFilter.BLUR)
        elif self.blur_type == "gaussian":
            radius = random.uniform(0, 3)
            blured_image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif self.blur_type == "detail":
            blured_image = image.filter(ImageFilter.DETAIL)
        elif self.blur_type == "edge_enhance":
            blured_image = image.filter(ImageFilter.EDGE_ENHANCE)
        elif self.blur_type == "sharpen":
            blured_image = image.filter(ImageFilter.SHARPEN)
        else:
            print("ERROR! Blur type not support, please check it later!")
        blured_image = np.asarray(blured_image)
        return blured_image
        
    def __call__(self, results):
        if 'blur' not in results:
            blur = True if np.random.rand() < self.blur_ratio else False
            results['blur'] = blur
        if results['blur']:
            # image filtering
            results['img'] = self.filtering(results['img'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(blur_type={}, blur_ratio={})'.format(
            self.blur_type, self.blur_ratio)


@PIPELINES.register_module
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = mmcv.impad(results['img'], self.size)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            padded_masks = [
                mmcv.impad(mask, pad_shape, pad_val=self.pad_val)
                for mask in results[key]
            ]
            results[key] = np.stack(padded_masks, axis=0)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


@PIPELINES.register_module
class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        for i in range(1000):
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
    
            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            img_shape = img.shape
            results['img'] = img
            results['img_shape'] = img_shape
    
            # crop bboxes accordingly and clip to the image boundary
            for key in results.get('bbox_fields', []):
                bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                       dtype=np.float32)
                bboxes = results[key] - bbox_offset
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
                results[key] = bboxes
    
            # filter out the gt bboxes that are completely cropped
            if 'gt_bboxes' in results:
                gt_bboxes = results['gt_bboxes']
                valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                    gt_bboxes[:, 3] > gt_bboxes[:, 1])
                """
                # if no gt bbox remains after cropping, just skip this image
                if not np.any(valid_inds):
                    return None
                """
                # if no gt bbox remains after cropping, jump to next circle
                if not np.any(valid_inds):
                    continue
                results['gt_bboxes'] = gt_bboxes[valid_inds, :]
                if 'gt_labels' in results:
                    results['gt_labels'] = results['gt_labels'][valid_inds]
    
                # filter and crop the masks
                if 'gt_masks' in results:
                    valid_gt_masks = []
                    for i in np.where(valid_inds)[0]:
                        gt_mask = results['gt_masks'][i][crop_y1:crop_y2, crop_x1:
                                                         crop_x2]
                        valid_gt_masks.append(gt_mask)
                    results['gt_masks'] = valid_gt_masks
    
            return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)


@PIPELINES.register_module
class SegResizeFlipPadRescale(object):
    """A sequential transforms to semantic segmentation maps.

    The same pipeline as input images is applied to the semantic segmentation
    map, and finally rescale it by some scale factor. The transforms include:
    1. resize
    2. flip
    3. pad
    4. rescale (so that the final size can be different from the image size)

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        if results['keep_ratio']:
            gt_seg = mmcv.imrescale(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        else:
            gt_seg = mmcv.imresize(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        if results['flip']:
            gt_seg = mmcv.imflip(gt_seg)
        if gt_seg.shape != results['pad_shape']:
            gt_seg = mmcv.impad(gt_seg, results['pad_shape'][:2])
        if self.scale_factor != 1:
            gt_seg = mmcv.imrescale(
                gt_seg, self.scale_factor, interpolation='nearest')
        results['gt_semantic_seg'] = gt_seg
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor={})'.format(
            self.scale_factor)


@PIPELINES.register_module
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        img = results['img']

        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
            img = img + delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img = img * alpha
        
        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img.astype(np.uint8))

        # random saturation
        if random.randint(2):
            img[..., 1] = img[..., 1] * random.uniform(self.saturation_lower,
                                                       self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] = img[..., 0] + random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360
        
        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img.astype(np.uint8))

        # random contrast
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img = img * alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img.astype(np.uint8)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={}, '
                     'saturation_range={}, hue_delta={})').format(
                         self.brightness_delta, self.contrast_range,
                         self.saturation_range, self.hue_delta)
        return repr_str


@PIPELINES.register_module
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
    """

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, results):
        if random.randint(2):
            return results

        img, boxes = [results[k] for k in ('img', 'gt_bboxes')]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        boxes = boxes + np.tile((left, top), 2)

        results['img'] = expand_img
        results['gt_bboxes'] = boxes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, to_rgb={}, ratio_range={})'.format(
            self.mean, self.to_rgb, self.ratio_range)
        return repr_str


@PIPELINES.register_module
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, results):
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)
                
                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str


@PIPELINES.register_module
class MinIoFRandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, min_iofs=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_iofs, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, results):
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4), mode='iof_crop').reshape(-1)
                if overlaps.max() > min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                        center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                               center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str


@PIPELINES.register_module
class Corrupt(object):

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(corruption={}, severity={})'.format(
            self.corruption, self.severity)
        return repr_str
