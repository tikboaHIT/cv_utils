@PIPELINES.register_module()
class AA(object):
    def __init__(self, level='v0'):

        self.augmentation_name = level

    def __call__(self, results):
        if results['gt_bboxes'] is None or len(results['gt_bboxes']) == 0:
            return results
        img = results['img']
        h, w, _ = img.shape
        bboxes = results['gt_bboxes']
        bboxes[:, 0::2] = bboxes[:, 0::2] / w
        bboxes[:, 1::2] = bboxes[:, 1::2] / h
        bboxes = bboxes[:, [1, 0, 3, 2]]
        img, bboxes = distort_image_with_autoaugment(img, bboxes, self.augmentation_name)
        h, w, _ = img.shape
        bboxes = bboxes[:, [1, 0, 3, 2]]
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, 1)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, 1)
        invalid_inds = (bboxes[:, 2] <= bboxes[:, 0]) | (
                bboxes[:, 3] <= bboxes[:, 1])
        if np.any(invalid_inds):
            print('AA: invalid bbox')
            return None

        bboxes[:, 0::2] = bboxes[:, 0::2] * w
        bboxes[:, 1::2] = bboxes[:, 1::2] * h
        results['img'] = img
        results['gt_bboxes'] = bboxes
        return results

    def __repr__(self):
        return self.__class__.__name__
