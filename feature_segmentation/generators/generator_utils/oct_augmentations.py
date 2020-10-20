from albumentations import (
    HorizontalFlip,
    RandomBrightnessContrast,
    RandomGamma,
    GaussNoise,
    ElasticTransform,
    Compose,
    Rotate,
    ISONoise,
    GlassBlur,
    OneOf, RandomSizedCrop, PadIfNeeded, VerticalFlip, RandomRotate90, GridDistortion, CLAHE)


def get_augmentations(params):
    augmentations = {
        "light": Compose([
            HorizontalFlip(p = params.flip)]),

        "medium":  Compose([
            OneOf([RandomSizedCrop(min_max_height=(150, 150), height=256, width=256, p=0.1),
                PadIfNeeded(min_height=256, min_width=256, p=0.5)], p=1),    
            VerticalFlip(p=0.2),              
            RandomRotate90(p=0.2),
            OneOf([
                ElasticTransform(p=0.5, alpha=100, sigma=8, alpha_affine=120 * 0.03),
                GridDistortion(p=0.5),
                RandomBrightnessContrast(p=0.8),    
                ], p=0.5),
            Rotate(p=0.5),
            GaussNoise(var_limit=200.0,p=0.2),
            ISONoise(color_shift=(0.01, 0.5), intensity=(0.1, 0.9), p=0.3),
            GlassBlur(sigma=0.5, max_delta=1, p=0.2),
            RandomGamma(gamma_limit=(50, 150), p=0.2)]),

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
