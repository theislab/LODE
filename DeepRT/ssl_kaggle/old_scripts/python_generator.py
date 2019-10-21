import numpy as np
import keras
import cv2
import pandas as pd
import os
import random
import skimage as sk
from skimage import transform
import scipy
from params import *

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,fundus_path, is_training,label_path, batch_size, dim,
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
        x = train_im[int(train_im.shape[0] / 2), :, :].sum(1)  # sum over axis=1
        r = (x > x.mean() / 10).sum() / 2
        s = scale * 1.0 / r
        mod_img = cv2.resize(train_im, (0, 0), fx=s, fy=s)

        # substract local mean color
        blurred = cv2.GaussianBlur(mod_img, (0, 0), scale / 30)
        mod_img = cv2.addWeighted(mod_img, 4, blurred, -4, 128)

        # remove outer 10%
        b = np.zeros(mod_img.shape, mod_img.dtype)
        cv2.circle(b, (int(mod_img.shape[1] / 2), int(mod_img.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
        mod_img = (mod_img * b) + 128 * (1 - b)

        # scaling
        mod_img = np.divide(mod_img, 255., dtype=np.float32)
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
        shift = bool(random.getrandbits(1))
        noise = bool(random.getrandbits(1))
        brightness = bool(random.getrandbits(1))
        contrast = bool(random.getrandbits(1))
        rotate = bool(random.getrandbits(1))

        if (rotate):
            train_im = self.random_rotation(train_im)
        if (shift):
            train_im = self.random_shift(train_im)
        if (noise):
            train_im = self.gaussian_noise(train_im)
        if (brightness):
            train_im = self.brightness(train_im, self.brightness_factor)
        if (contrast):
            train_im = self.contrast_range(train_im, self.contrast_factor)

        # label preserving augmentations
        if rot90_:
            train_im = self.rot90(train_im)
        if (flip_hor):
            train_im = self.flip_horizontally(train_im)
        if (flip_ver):
            train_im = self.flip_vertically(train_im)

        return train_im

    def gaussian_noise(self,image_array):
        '''
        :param image_array: Image onto which gaussian noise is added, numpy array, float
        :return: transformed image array
        '''
        value = random.uniform(0, 1)
        image_array = image_array + np.random.normal(0, value)
        return (image_array)

    def brightness(self, image_array,brightness_factor):
        range = np.random.uniform(-1, 1) * brightness_factor
        image_array = image_array + range * image_array
        return (image_array)

    def contrast_range(self,image_array,contrast_factor):
        range = np.random.uniform(-1,1) * contrast_factor + 1
        min_image_array = np.min(image_array)
        image_array = ((image_array -min_image_array) * range) + min_image_array
        return(image_array)

    def rot90(self,image_array):
        rot_image_array = np.rot90(image_array)
        return rot_image_array

    def random_rotation(self,image_array):
        # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-25, 25)
        r_im = sk.transform.rotate(image_array, random_degree, preserve_range=True)
        return r_im.astype(np.float32)

    def flip_vertically(self,image_array):
        flipped_image = np.fliplr(image_array)
        return flipped_image

    def flip_horizontally(self,image_array):
        flipped_image = np.flipud(image_array)
        return (flipped_image)

    def random_shift(self,image_array):
        rand_x = random.uniform(-15, 15)
        rand_y = random.uniform(-40, 40)
        image_array = scipy.ndimage.shift(image_array[:, :, 0], (rand_x, rand_y))
        return (image_array.reshape(self.shape[0], self.shape[1], 1))
