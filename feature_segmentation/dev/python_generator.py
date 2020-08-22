import numpy as np
import tensorflow.keras
import cv2
import os
import random
import skimage as sk
import matplotlib.pyplot as plt
from skimage import transform
import scipy
from PIL import Image
from skimage.transform import AffineTransform, warp

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    ShiftScaleRotate
)


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, params, is_training):
        'Initialization'
        self.shape = (params.img_shape, params.img_shape)
        self.batch_size = params.batch_size
        self.list_IDs = list_IDs
        self.shuffle = True
        self.on_epoch_end()
        self.image_path = os.path.join(params.data_path, "images")
        self.label_path = os.path.join(params.data_path, "masks")
        self.is_training = is_training

        self.augment_box = Compose([
            CenterCrop(p = 0.1, height = params.img_shape, width = params.img_shape),
            HorizontalFlip(p = 0.5),
            Transpose(p = 0.5),
            RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.2, p = 0.5),
            RandomGamma(p = 0.2),
            ElasticTransform(p = 0.2),
            GridDistortion(p = 0.2),
            OpticalDistortion(p = 0.2, distort_limit = 2, shift_limit = 0.5),
        ])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(int(len(self.list_IDs)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def resize(self, im):
        desired_size = self.shape[0]
        im = Image.fromarray(im)

        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = im.resize(new_size, Image.NEAREST)
        # create a new image and paste the resized on it

        new_im = Image.new("L", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))

        return np.array(new_im)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.shape[0], self.shape[1], 3))
        y = np.empty((self.batch_size, self.shape[0], self.shape[1], 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load samples
            im = Image.open(os.path.join(self.image_path, ID))
            lbl = Image.open(os.path.join(self.label_path, ID))

            # convert to numpy
            im = np.array(im)
            lbl = np.array(lbl)

            # resize samples
            im_resized = self.resize(im)
            lbl_resized = self.resize(lbl)

            # if image grey scale, make 3 channel
            if len(im_resized.shape) == 2:
                im_resized = np.stack((im_resized,) * 3, axis = -1)

            # Store sample
            X[i,] = im_resized.reshape(self.shape[0], self.shape[1], 3)
            y[i,] = lbl_resized.reshape((self.shape[0], self.shape[1], 1))

            X[i,], y[i,] = self.__pre_process(X[i,], y[i,])

        return X, y.astype(np.int32)

    def example_record(self):

        record_idx = random.randint(0, len(self.list_IDs))
        # load samples
        im = Image.open(os.path.join(self.image_path, self.list_IDs[record_idx - 1]))
        lbl = Image.open(os.path.join(self.label_path, self.list_IDs[record_idx - 1]))

        # convert to numpy
        im = np.array(im)
        lbl = np.array(lbl)

        # resize samples
        im_resized = self.resize(im)
        lbl_resized = self.resize(lbl)

        # if image grey scale, make 3 channel
        if len(im_resized.shape) == 2:
            im_resized = np.stack((im_resized,) * 3, axis = -1)

        # Store sample
        image = im_resized.reshape(self.shape[0], self.shape[1], 3)
        label = lbl_resized.reshape((self.shape[0], self.shape[1], 1))

        image, label = self.__pre_process(image, label)

        record = (image, label[:, :, 0])
        return record

    def __pre_process(self, train_im, label_im):

        # label_im = np.nan_to_num(label_im)
        train_im = np.nan_to_num(train_im)

        if self.is_training:
            aug = self.augment_box(image = train_im.astype(np.uint8), mask = label_im.astype(np.uint8))
            train_im = aug['image']
            label_im = aug['mask']

        # label_im = np.divide(label_im, 500., dtype=np.float32)
        train_im = np.divide(train_im, 255., dtype = np.float32)
        return (train_im.reshape(self.shape[0], self.shape[1], 3), label_im.reshape((self.shape[0],
                                                                                     self.shape[0],
                                                                                     1)))

    def augment(self, train_im, label_im):
        # get boolean if rotate
        flip_hor = bool(random.getrandbits(1))
        flip_ver = bool(random.getrandbits(1))
        rot90_ = bool(random.getrandbits(1))
        shift = False  # bool(random.getrandbits(1))
        noise = bool(random.getrandbits(1))
        rotate = False  # bool(random.getrandbits(1))

        if (rotate):
            train_im, label_im = self.random_rotation(train_im, label_im)
        if (shift):
            train_im, label_im = self.random_shift(train_im, label_im)
        if (noise):
            train_im = self.random_noise(train_im)

        # label preserving augmentations
        if rot90_:
            train_im, label_im = self.rot90(train_im, label_im)
        if (flip_hor):
            train_im, label_im = self.flip_horizontally(train_im, label_im)
        if (flip_ver):
            train_im, label_im = self.flip_vertically(train_im, label_im)

        return train_im, label_im

    def random_rotation(self, image_array, label_array):

        (h, w) = image_array.shape[:2]
        center = (w / 2, h / 2)

        random_degree = random.uniform(-50, 50)
        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, random_degree, 1.0)

        r_im = cv2.warpAffine(image_array, M, (w, h), cv2.INTER_NEAREST)
        l_im = cv2.warpAffine(label_array, M, (w, h), cv2.INTER_NEAREST)
        return r_im, l_im

    def flip_vertically(self, image_array, label_array):
        flipped_image = np.fliplr(image_array)
        flipped_label = np.fliplr(label_array)
        return flipped_image, flipped_label

    def rot90(self, image_array, label_array):
        rot_image_array = np.rot90(image_array)
        rot_label_array = np.rot90(label_array)
        return rot_image_array, rot_label_array

    def flip_horizontally(self, image_array, label_array):
        flipped_image = np.flipud(image_array)
        flipped_label = np.flipud(label_array)
        return flipped_image, flipped_label

    def random_noise(self, image_array):
        # add random noise to the image
        return sk.util.random_noise(image_array)

    def shift(self, image, vector):
        transform = AffineTransform(translation = vector)
        shifted = warp(image, transform, mode = 'wrap', preserve_range = True)

        shifted = shifted.astype(image.dtype)
        return shifted

    def random_shift(self, image_array, label_array):

        rand_x = random.uniform(-40, 40)
        rand_y = random.uniform(-40, 40)

        image_array = self.shift(image_array, (rand_x, rand_y))
        label_array = self.shift(label_array, (rand_x, rand_y))

        return (image_array.reshape(self.shape[0], self.shape[1], 3),
                label_array.reshape(self.shape[0], self.shape[1], 1))
