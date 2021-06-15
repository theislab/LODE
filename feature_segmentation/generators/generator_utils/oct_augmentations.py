import cv2
from albumentations import (
    HorizontalFlip,
    RandomBrightnessContrast,
    Compose,
    Rotate,
    ShiftScaleRotate)


def get_augmentations(params):
    augmentations = {
        "light": Compose([
            HorizontalFlip(p = 0.5),
            Rotate(p = 0.5)])
        ,
        "medium": Compose([HorizontalFlip(p = 0.5),
                           RandomBrightnessContrast(p = 0.2,
                                                    brightness_limit = (-0.3, 0.2),
                                                    contrast_limit = (-0.2, 0.2)),
                           Rotate(p = 0.4, interpolation = cv2.INTER_NEAREST),
                           ShiftScaleRotate(p = 0.2,
                                            shift_limit = 0.2,
                                            rotate_limit = 0,
                                            interpolation = cv2.INTER_NEAREST,
                                            border_mode = cv2.BORDER_CONSTANT,
                                            value = 0)],
                          )}
    return augmentations
