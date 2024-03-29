from copy import deepcopy
import numpy as np
import tensorflow.keras
import os

from utils.oct_augmentations import get_augmentations
from utils.image_processing import read_resize_random_invert
from utils.utils import get_class_distribution, upsample, label_mapping


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for tensorflow.keras'

    def __init__(self, list_IDs, params, is_training):
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
            im_path = os.path.join(self.image_path, ID)
            lbl_path = os.path.join(self.label_path, ID)

            im_resized, lbl_resized = read_resize_random_invert(im_path, lbl_path, self.shape)

            # convert Serous PED to Fibro PED and artifact to neuro sensory retina
            lbl_resized = label_mapping(lbl_resized)

            # Store sample
            X[i, ] = im_resized
            y[i, ] = lbl_resized

            X[i, ], y[i, ] = self.__pre_process(X[i, ], y[i, ])
        return X, y.astype(np.float32)

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


if __name__ == "__main__":
    print("imports works!")
