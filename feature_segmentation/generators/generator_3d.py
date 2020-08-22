import numpy as np
import keras
import os
import random
from PIL import Image

from generators.generator_utils.image_processing import resize, read_resize
from generators.generator_utils.oct_augmentations import get_augmentations


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

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
        self.augment_box = get_augmentations(params)[params.aug_stretegy]

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

    @classmethod
    def decode_filename(cls, filename):
        # check if no series id is logged
        if len(filename.split("_")) < 5:
            patient_id, lat, study_date, frame = filename.replace(".png", "").split("_")
            lat = cls.laterality_formatting(lat)
            series_id = "MISSING"
        else:
            patient_id, lat, study_date, series_id, frame = filename.replace(".png", "").split("_")
            lat = cls.laterality_formatting(lat)
        return patient_id, lat, study_date, series_id, frame

    @staticmethod
    def laterality_formatting(lat):
        if lat == "Right":
            lat = "R"
        elif lat == "Left":
            lat = "L"
        return lat

    @staticmethod
    def generate_volume_indices(frame, number_of_indices):
        sample_size = (number_of_indices - 1) // 2
        upper_sample_size = (number_of_indices - 1) // 2
        lower_sample_size = (number_of_indices - 1) // 2

        frame_idx = int(frame)
        if (frame_idx - sample_size) < 0:
            # not enough lower frames
            lower_sample_size = np.abs(frame_idx - sample_size)

        if (frame_idx + sample_size) > 48:
            # not enough upper frames
            upper_sample_size = np.abs(frame_idx + sample_size - 48)

        # sample indices without replacement
        lower_indices = np.random.choice(np.arange(0,frame_idx - 1), lower_sample_size, replace=False)
        upper_indices = np.random.choice(np.arange(frame_idx + 1, 48), upper_sample_size, replace = False)
        return np.sort(lower_indices), np.sort(upper_indices)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(int(len(self.list_IDs)))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, self.params.number_of_scans, self.shape[0], self.shape[1], 3))
        y = np.empty((self.batch_size, self.params.number_of_scans, self.shape[0], self.shape[1], 1), dtype = np.int32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            patient_id, lat, study_date, series_id, frame = DataGenerator.decode_filename(ID)

            # generate oct volume indices
            lower_indices, upper_indices = DataGenerator.upgenerate_volume_indices(frame, self.params.number_of_frames)

            for i in range(4):
                lower_idx = lower_indices[i]
                upper_idx = upper_indices[i]

            # load samples
            im_path = os.path.join(self.image_path, ID)
            lbl_path = os.path.join(self.label_path, ID)
            im_resized, lbl_resized = read_resize(im_path, lbl_path, self.shape)

            # Store sample
            X[i, ] = im_resized
            y[i, ] = lbl_resized

            X[i, ], y[i, ] = self.__pre_process(X[i,], y[i,])
        return X, y.astype(np.int32)

    def example_record(self):
        record_idx = random.randint(0, len(self.list_IDs))

        # load samples
        im_path = os.path.join(self.image_path, self.list_IDs[record_idx - 1])
        lbl_path = os.path.join(self.label_path, self.list_IDs[record_idx - 1])
        image, label = read_resize(im_path, lbl_path, self.shape)

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

        # label_im = np.divide(label_im, 500., dtype=np.float32)
        train_im = np.divide(train_im, 255., dtype = np.float32)
        return (train_im.reshape(self.shape[0], self.shape[1], 3), label_im.reshape((self.shape[0],
                                                                                     self.shape[0],
                                                                                     1)))
