from .utils import *
from ..registry import PIPELINES


@PIPELINES.register_module
class AutoAugment(object):
    """Applies the AutoAugment policy to `image` and `bboxes`.
    Args:
      augmentation_name: The name of the AutoAugment policy to use. The available
        options are `v0`, `v1`, `v2`, `v3` and `test`. `v0` is the policy used for
        all of the results in the paper and was found to achieve the best results
        on the COCO dataset. `v1`, `v2` and `v3` are additional good policies
        found on the COCO dataset that have slight variation in what operations
        were used during the search procedure along with how many operations are
        applied in parallel to a single image (2 vs 3).
    """

    def __init__(self,
                 augmentation_name='v0',
                 cutout_max_pad_fraction=0.75,
                 cutout_bbox_replace_with_mean=False,
                 cutout_const=100,
                 translate_const=250,
                 cutout_bbox_const=50,
                 translate_bbox_const=120
                 ):
        self.augmentation_name = augmentation_name
        self.cutout_max_pad_fraction = cutout_max_pad_fraction
        self.cutout_bbox_replace_with_mean = cutout_bbox_replace_with_mean
        self.cutout_const = cutout_const
        self.translate_const = translate_const
        self.cutout_bbox_const = cutout_bbox_const
        self.translate_bbox_const = translate_bbox_const

    def __call__(self, results):
        image = results['img']
        img_h, img_w, _ = results['img_shape']
        bboxes = []
        for bbox in results['gt_bboxes']:
            xmin, ymin, xmax, ymax = bbox[:4]
            min_y = ymin / img_h
            min_x = xmin / img_w
            max_y = ymax / img_h
            max_x = xmax / img_w
            bboxes.append([min_y, min_x, max_y, max_x])
        bboxes = np.array(bboxes)

        augmented_image, augmented_bboxes = distort_image_with_autoaugment(
                image, bboxes,
                self.augmentation_name,
                self.cutout_max_pad_fraction,
                self.cutout_bbox_replace_with_mean,
                self.cutout_const,
                self.translate_const,
                self.cutout_bbox_const,
                self.translate_bbox_const
        )
        augmented_bboxes_ = []
        for bbox in augmented_bboxes:
            ymin = bbox[0] * img_h
            xmin = bbox[1] * img_w
            ymax = bbox[2] * img_h
            xmax = bbox[3] * img_w
            augmented_bboxes_.append([xmin, ymin, xmax, ymax])

        results['img'] = augmented_image
        results['gt_bboxes'] = np.array(augmented_bboxes_)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '''augmentation_name={}, cutout_max_pad_fraction={},
                       cutout_bbox_replace_with_mean={}, cutout_const={},
                       translate_const={}, cutout_bbox_const={}, 
                       translate_bbox_const={}'''.format(
            self.augmentation_name, self.cutout_max_pad_fraction,
            self.cutout_bbox_replace_with_mean, self.cutout_const,
            self.translate_const, self.cutout_bbox_const,
            self.translate_bbox_const)
        return repr_str
