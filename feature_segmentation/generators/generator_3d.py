import numpy as np
import keras
import os
import random
import matplotlib.pyplot as plt
from generators.generator_utils.image_processing import resize, read_resize, read_resize_image
from generators.generator_utils.oct_augmentations import get_augmentations
from utils.plotting import save_segmentation_plot
import cv2


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, params, is_training, pretraining=False,
            choroid_latest = False):
        'Initialization'
        self.shape = (params.img_shape, params.img_shape)
        self.batch_size = params.batch_size
        self.shuffle = True
        self.list_IDs = list_IDs
        self.on_epoch_end()
        self.params = params
        self.pretraining = pretraining
        self.is_training = is_training
        self.augment_box = get_augmentations(params)[params.aug_strategy]
        self.choroid = choroid_latest
        
        self.image_path = os.path.join(params.data_path, "images")
        if self.choroid:
            self.label_path = os.path.join(params.data_path, "masks_choroid")
        else:
            self.label_path = os.path.join(params.data_path, "masks")

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
    def record_identifier(cls, patient_id, lat, study_date):
        return f"{patient_id}_{lat}_{study_date}_*", f"{patient_id}_{lat}_{study_date}_*"

    @classmethod
    def decode_filename(cls, filename):
        series_id = None
        frame = None

        components = filename.split("_")
        # check if no series id is logged
        if len(components) < 5:
            patient_id, lat, study_date = components[:3]

            # decide final component is frame or series id
            final_ = components[-1].replace(".png", "")
            if int(final_) in range(0, 49):
                frame = final_
            else:
                series_id = final_
            lat = cls.laterality_formatting(lat)
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

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(int(len(self.list_IDs)))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((self.batch_size, self.params.n_scans, self.shape[0], self.shape[1], 3), dtype = np.float64)
        y = np.zeros((self.batch_size, self.shape[0], self.shape[1], 1), dtype = np.int32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            try:
                patient_id, lat, study_date, series_id, frame = DataGenerator.decode_filename(ID)
            except:
                frame = 1

            # load samples
            im_path = os.path.join(self.image_path, ID)
            lbl_path = os.path.join(self.label_path, ID)
            im_resized, lbl_resized = read_resize(im_path, lbl_path, self.shape)

            volume_path = os.path.join(self.params.data_path, "volumes",
                                       ID.replace(".png", ""))

            # add annotated scan
            X[i, self.params.n_scans // 2,] = im_resized
            y[i,] = lbl_resized

            req_number = False
            # check volume path
            if os.path.exists(volume_path):
                num_octs = len(os.listdir(volume_path + "/volume"))
                if num_octs == 49:
                    req_number = True

            # if frame info not available, pad with zeros
            if (not frame) or (not os.path.exists(volume_path) or (not req_number)):
                X[i, self.params.n_scans // 2,] = im_resized

            elif frame:
                # get right and left available indices
                _right_frames = np.arange(int(frame) + 1, 49)
                _left_frames = np.arange(0, int(frame))

                n_right = len(_right_frames)
                n_left = len(_left_frames)

                side_sample_size = (self.params.n_scans - 1) // 2

                if n_right >= side_sample_size:
                    right_samples = _right_frames[np.arange(0, side_sample_size - 1)]
                else:
                    right_samples = _right_frames

                if n_left >= side_sample_size:
                    left_samples = _left_frames[- np.arange(side_sample_size - 1, 0, -1)]
                else:
                    left_samples = _left_frames

                # add sample oct's to volume
                for k, right_sample in enumerate(right_samples, 1):
                    oct_ = read_resize_image(os.path.join(volume_path, "volume", str(right_sample) + ".png"),
                                             self.shape)
                    # add to volume
                    X[i, self.params.n_scans // 2 + k,] = oct_

                for k, left_sample in enumerate(left_samples, 0):
                    oct_ = read_resize_image(os.path.join(volume_path, "volume", str(left_sample) + ".png"),
                                             self.shape)
                    # add to volume
                    X[i, self.params.n_scans // 2 - (len(left_samples) - k),] = oct_

            X[i,], y[i,] = self.__pre_process(X[i,], y[i,])

            if self.pretraining:
                y[y == 14] = 15
                y[y == 13] = 14
        return X, y.astype(np.int32)

    def example_record(self):
        record_idx = random.randint(0, len(self.list_IDs))
        X, y = self.__getitem__(record_idx)

        save_path = os.path.join(self.params.model_directory, "examples", f"example_{record_idx}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_segmentation_plot(save_path + "/_label", y[0, :, :, 0])
        for i in range(X.shape[1]):
            cv2.imwrite(os.path.join(save_path + f"/image_scan_{i}.png"), X[0, i,] * 255)

    def __pre_process(self, train_im, label_im):
        train_im = np.nan_to_num(train_im)

        if self.is_training:
            for n in range(self.params.n_scans):
                if n == self.params.n_scans // 2:
                    aug = self.augment_box(image = train_im[n,].astype(np.uint8), mask = label_im.astype(np.uint8))
                    train_im[n,] = aug['image']
                    label_im = aug['mask']
                else:
                    aug = self.augment_box(image = train_im[n,].astype(np.uint8))
                    train_im[n,] = aug['image']

        train_im = np.divide(train_im, 255., dtype = np.float32)
        return train_im.reshape(1, self.params.n_scans, self.shape[0], self.shape[1], 3), \
               label_im.reshape(1, self.shape[0], self.shape[1], 1)
