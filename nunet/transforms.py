import random

import albumentations as A
import cv2
import scipy.ndimage as ndi


class RandomPickChannel(A.ImageOnlyTransform):
    """For Cellpose

    1. Pick the red channel for all images
    2. Broadcast it to RGB back
    """
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        ch = img.shape[-1]  # img.shape[2]
        c = random.randint(0, ch - 1)
        return cv2.cvtColor(img[..., c], cv2.COLOR_GRAY2RGB)

    def get_transform_init_args_names(self):
        return ()


class TransformsCellpose(A.ImageOnlyTransform):
    """For Cellpose

    1. Pick the red channel for all images
    2. Broadcast it to RGB back
    """
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2RGB)

    def get_transform_init_args_names(self):
        return ()


class RelabelMask(A.DualTransform):
    """Relabel instance mask after padding (or other ops.)

    Targets:
        mask

    """
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        mask_relabeld, _ = ndi.label(mask)
        return mask_relabeld

    def get_transform_init_args_names(self):
        return ()
