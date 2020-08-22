from albumentations import (
    HorizontalFlip,
    RandomBrightnessContrast,
    RandomGamma,
    GaussNoise,
    ElasticTransform,
    Compose,
    Rotate,
    OneOf, RandomSizedCrop, PadIfNeeded, VerticalFlip, RandomRotate90, GridDistortion, CLAHE)


def get_augmentations(params):
    augmentations = {
        "light": Compose([
            HorizontalFlip(p = params.flip),
            RandomBrightnessContrast(p = params.brightness),
            RandomGamma(p = params.gamma),
            GaussNoise(p = params.noise),
            ElasticTransform(p = params.elastic)]),

        "medium": Compose([
            HorizontalFlip(p = params.flip),
            RandomBrightnessContrast(p = params.brightness),
            RandomGamma(p = params.gamma),
            GaussNoise(p = params.noise),
            ElasticTransform(p = params.elastic),
            Rotate(p = 0.5)]),

        "severe": Compose([
            OneOf([RandomSizedCrop(min_max_height = (180, 180),
                                   height = params.img_shape,
                                   width = params.img_shape, p = 0.1),
                   PadIfNeeded(min_height = params.img_shape,
                               min_width = params.img_shape,
                               p = 0.5)], p = 1),
            VerticalFlip(p = 0.5),
            RandomRotate90(p = 0.5),
            OneOf([
                ElasticTransform(p = 0.5, alpha = 100, sigma = 8, alpha_affine = 120 * 0.03),
                GridDistortion(p = 0.5),
            ], p = 0.8),
            CLAHE(p = 0.1),
            Rotate(p = 0.3),
            GaussNoise(p = 0.5),
            RandomBrightnessContrast(p = 0.1),
            RandomGamma(p = 0.1)])}
    return augmentations
