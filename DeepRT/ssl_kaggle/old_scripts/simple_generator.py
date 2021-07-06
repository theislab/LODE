import numpy as np
import tensorflow.keras
import cv2
import pandas as pd
import os
import random
import skimage as sk
from skimage import transform
import scipy
from params import *

class DataGenerator_simple(tensorflow.keras.utils.Sequence):
    'Generates data for tensorflow.keras'
    def __init__(self, list_IDs,fundus_path, label_path, is_training, batch_size, dim,
                 brightness_factor,
                 contrast_factor,
                 n_channels=3, shuffle=True):
        'Initialization'
        self.shape = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.fundus_path = fundus_path
        self.label_path = label_path
        self.is_training = False
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

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

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.shape[0],self.shape[1], self.n_channels))
        y = np.empty((self.batch_size,5))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #load samples
            im = cv2.imread(os.path.join(self.fundus_path, ID + '.jpeg'))

            label_pd = pd.read_csv(os.path.join(self.label_path, "trainLabels.csv"))

            label = label_pd[label_pd.image == ID]["level"].values[0]
            #resize samples
            im_resized = cv2.resize(im, (self.shape[0],self.shape[1])).reshape(params["img_shape"])
            # Store sample
            X[i,] = im_resized
            y[i,] = np.eye(5)[label]

            X[i,] = self.__pre_process(X[i,],scale=300)

        return X, y.astype(np.int64)

    def __pre_process(self, train_im,scale):
        # scale image to a given radius

        # scaling
        mod_img = np.divide(train_im, 255., dtype=np.float32)
        mod_img = cv2.resize(mod_img, (self.shape[0], self.shape[0]))

        if self.is_training:
            self.augment(mod_img)
        return mod_img.reshape(params["img_shape"])

    def augment(self,train_im):
        # get boolean if rotate
        # get boolean if rotate
        flip_hor = bool(random.getrandbits(1))
        flip_ver = bool(random.getrandbits(1))
        rot90_ = bool(random.getrandbits(1))
        # label preserving augmentations
        if rot90_:
            train_im = self.rot90(train_im)
        if (flip_hor):
            train_im = self.flip_horizontally(train_im)
        if (flip_ver):
            train_im = self.flip_vertically(train_im)

        return train_im

    def rot90(self,image_array):
        rot_image_array = np.rot90(image_array)
        return rot_image_array

    def flip_vertically(self,image_array):
        flipped_image = np.fliplr(image_array)
        return flipped_image

    def flip_horizontally(self,image_array):
        flipped_image = np.flipud(image_array)
        return (flipped_image)

