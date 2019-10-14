import numpy as np
import keras
import cv2
import os
import random
import skimage as sk
from skimage import transform
import scipy
from PIL import Image
from skimage.transform import AffineTransform, warp


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, image_path, label_path, is_training, batch_size, dim, brightness_factor,
                 contrast_factor, n_channels=1, shuffle=True):
        'Initialization'
        self.shape = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_path = image_path
        self.label_path = label_path
        self.is_training = is_training
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
        X = np.empty((self.batch_size, self.shape[0],self.shape[1], 3))
        y = np.empty((self.batch_size, self.shape[0],self.shape[1], 1))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #load samples
            im = cv2.imread(os.path.join(self.image_path,"all_images", str(ID) + '.tif'))
            lbl = cv2.imread(os.path.join(self.label_path,"all_labels", str(ID) + '.tif'))[:,:,0]

            #resize samples
            im_resized = cv2.resize(im, self.shape).reshape(self.shape[0],self.shape[1],3)
            lbl_resized = np.array(Image.fromarray(lbl).resize(self.shape)).reshape((self.shape[0],
                                                                                     self.shape[1],
                                                                                     1))
            # Store sample
            X[i,] = im_resized
            y[i,] = lbl_resized


            X[i,], y[i,] = self.__pre_process(X[i,],y[i,])
        return X, y.astype(np.int32)

    def __pre_process(self, train_im, label_im):

        # scaling
        label_im = label_im /  255.
        train_im = np.divide(train_im, 255., dtype=np.float32)
        #label_im = np.nan_to_num(label_im)
        train_im = np.nan_to_num(train_im)
        if self.is_training:
            self.augment(train_im,label_im)
        return (train_im.reshape(self.shape[0],self.shape[1],3), label_im.reshape((self.shape[0],
                                                                                   self.shape[0],
                                                                                     1)))

    def augment(self,train_im, label_im):
        # get boolean if rotate
        flip_hor = bool(random.getrandbits(1))
        flip_ver = bool(random.getrandbits(1))
        rot90_ = bool(random.getrandbits(1))
        shift =  bool(random.getrandbits(1))
        noise =  bool(random.getrandbits(1))
        brightness = bool(random.getrandbits(1))
        contrast = bool(random.getrandbits(1))
        rotate = bool(random.getrandbits(1))

        if (rotate):
            train_im, label_im = self.random_rotation(train_im,label_im)
        if (shift):
            train_im, label_im = self.random_shift(train_im, label_im)
        if (noise):
            train_im = self.gaussian_noise(train_im)
        if (brightness):
            train_im = self.brightness(train_im,self.brightness_factor)
        if (contrast):
            train_im = self.contrast_range(train_im,self.contrast_factor)

        #label preserving augmentations
        if rot90_:
            train_im, label_im = self.rot90(train_im, label_im)
        if (flip_hor):
            train_im, label_im = self.flip_horizontally(train_im, label_im)
        if (flip_ver):
            train_im, label_im = self.flip_vertically(train_im, label_im)


        return train_im, label_im

    def random_rotation(self,image_array, label_array):

        (h, w) = image_array.shape[:2]
        center = (w / 2, h / 2)

        random_degree = random.uniform(-50, 50)
        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, random_degree, 1.0)

        r_im = cv2.warpAffine(image_array, M, (w, h), cv2.INTER_NEAREST)
        l_im = cv2.warpAffine(label_array, M, (w, h), cv2.INTER_NEAREST)
        return (r_im, l_im)

    def flip_vertically(self,image_array, label_array):
        flipped_image = np.fliplr(image_array)
        flipped_label = np.fliplr(label_array)
        return (flipped_image, flipped_label)

    def rot90(self,image_array, label_array):
        rot_image_array = np.rot90(image_array)
        rot_label_array = np.rot90(label_array)
        return (rot_image_array, rot_label_array)

    def flip_horizontally(self,image_array, label_array):
        flipped_image = np.flipud(image_array)
        flipped_label = np.flipud(label_array)
        return (flipped_image, flipped_label)

    def random_noise(self, image_array):
        # add random noise to the image
        return sk.util.random_noise(image_array)


    def shift(self, image, vector):
        transform = AffineTransform(translation=vector)
        shifted = warp(image, transform, mode='wrap', preserve_range=True)

        shifted = shifted.astype(image.dtype)
        return (shifted)

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

    def random_shift(self, image_array, label_array):

        rand_x = random.uniform(-40, 40)
        rand_y = random.uniform(-40, 40)

        image_array = self.shift(image_array, (rand_x, rand_y))
        label_array = self.shift(label_array, (rand_x, rand_y))

        return (image_array.reshape(self.shape[0], self.shape[1], self.n_channels),
                label_array.reshape(self.shape[0], self.shape[1], 1))
