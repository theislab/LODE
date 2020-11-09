from copy import deepcopy

import numpy as np
import keras
import os
import random
from PIL import Image
import glob
from collections import Counter
import itertools
import pandas as pd

from generators.generator_utils.image_processing import resize, read_resize
from generators.generator_utils.oct_augmentations import get_augmentations

from feature_segmentation.generators.generator_utils.image_processing import read_resize_random_invert
from feature_segmentation.generators.generator_utils.utils import get_class_distribution, upsample


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, params, is_training, pretraining, choroid_latest):
        'Initialization'
        self.shape = (params.img_shape, params.img_shape)
        self.batch_size = params.batch_size
        self.shuffle = True
        self.list_IDs = list_IDs
        self.on_epoch_end()
        self.params = params
        self.image_path = os.path.join(params.data_path, "images")
        self.label_path = os.path.join(params.data_path, "masks")
        self.is_training = is_training
        self.augment_box = get_augmentations(params)[params.aug_strategy]
        self.val_aug_box = get_augmentations(params)["light"]

        if params.balance_dataset and is_training:
            upsampling_factors, label_repr = get_class_distribution(self.label_path, list_IDs)
            train_ids = deepcopy(list_IDs)
            for label in [5, 8, 13]:
                new_ids = upsample(train_ids, label, label_repr, upsampling_factors)
                train_ids = deepcopy(new_ids)
                upsampling_factors, label_repr = get_class_distribution(self.label_path, train_ids)

            self.list_IDs = new_ids

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
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.shape[0], self.shape[1], 3))
        y = np.empty((self.batch_size, self.shape[0], self.shape[1], 1), dtype = np.int32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load samples
            im_path = glob.glob(os.path.join(self.image_path, ID.replace(".png", "*")))[0]
            lbl_path = glob.glob(os.path.join(self.label_path, ID.replace(".png", "*")))[0]
            # im_path = os.path.join(self.image_path, ID)
            # lbl_path = os.path.join(self.label_path, ID)
            im_resized, lbl_resized = read_resize_random_invert(im_path, lbl_path, self.shape)

            # Store sample
            X[i,] = im_resized
            y[i,] = lbl_resized

            X[i,], y[i,] = self.__pre_process(X[i,], y[i,])
        return X, y.astype(np.int32)

    def example_record(self):
        record_idx = random.randint(0, len(self.list_IDs))
        print("number of ids are: ", len(self.list_IDs))
        # load samples
        im_path = os.path.join(self.image_path, self.list_IDs[record_idx - 1])
        lbl_path = os.path.join(self.label_path, self.list_IDs[record_idx - 1])
        image, label = read_resize_random_invert(im_path, lbl_path, self.shape)

        image, label = self.__pre_process(image, label)

        record = [image, label[:, :, 0]]
        return record, self.list_IDs[record_idx - 1]

    def __pre_process(self, train_im, label_im):

        # label_im = np.nan_to_num(label_im)
        train_im = np.nan_to_num(train_im)

        if self.is_training:
            aug = self.augment_box(image = train_im.astype(np.uint8), mask = label_im.astype(np.uint8))
            train_im = aug['image']
            label_im = aug['mask']

        else:
            aug = self.val_aug_box(image = train_im.astype(np.uint8), mask = label_im.astype(np.uint8))
            train_im = aug['image']
            label_im = aug['mask']

        # label_im = np.divide(label_im, 500., dtype=np.float32)
        train_im = np.divide(train_im, 255., dtype = np.float32)
        return (train_im.reshape(self.shape[0], self.shape[1], 3), label_im.reshape((self.shape[0],
                                                                                     self.shape[0],
                                                                                     1)))
