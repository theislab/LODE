"""General utility functions"""
import json
from random import shuffle

import matplotlib.pyplot as plt
from matplotlib import colors
import glob
import shutil
import tensorflow
from PIL import Image
import numpy as np
import matplotlib.gridspec as gridspec
import os
from sklearn.metrics import jaccard_score
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard, EarlyStopping
from pydicom import read_file

from utils.loss_functions import dice_loss, gen_dice
from utils.plotting import color_mappings


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.learning_rate = None
        self.batch_size = None
        self.num_epochs = None
        self.data_path = None
        self.img_shape = None
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent = 4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class Logging():

    def __init__(self, logging_directory, params):
        self.log_dir = logging_directory
        self.model_directory = None
        self.tensorboard_directory = None
        self.params = params

    def __create_dir(self, dir):
        os.makedirs(dir)

    def __create_main_directory(self):
        '''
        :return: create main log dir if not allready created
        '''
        if not os.path.isdir(self.log_dir):
            print("main logging dir does not exist, creating main logging dir ./logs")
            os.makedirs(self.log_dir)
        else:
            pass

    def __create_tensorboard_dir(self, model_dir):

        # set abs path to new dir
        new_dir = os.path.join(model_dir, "tensorboard_dir")

        # create new dir
        self.__create_dir(new_dir)

        # set object instance to new path
        self.tensorboard_directory = new_dir

    def __remove_empty_directories(self):

        # get current directories
        current_directories = glob.glob(self.log_dir + "/*")

        # check for each dir, if weight.hdf5 file is contained
        for current_directory in current_directories:
            if not os.path.isfile(os.path.join(current_directory, "weights.hdf5")):
                # remove directory
                shutil.rmtree(current_directory)

    def create_model_directory(self):
        '''
        :param logging_directory: string, gen directory for logging
        :return: None
        '''

        # create main dir if not exist
        self.__create_main_directory()

        # remove emtpy directories
        self.__remove_empty_directories()

        # get allready created directories
        existing_ = os.listdir(self.log_dir)

        # if first model iteration, set to zero
        if existing_ == []:
            new = 0
            # save abs path of created dir
            created_dir = os.path.join(self.log_dir, str(new))

            # make new directory
            self.__create_dir(created_dir)

            # create subdir for tensorboard logs
            self.__create_tensorboard_dir(created_dir)

        else:
            # determine the new model directory
            last_ = max(list(map(int, existing_)))
            new = int(last_) + 1

            # save abs path of created dir
            created_dir = os.path.join(self.log_dir, str(new))

            # make new directory
            self.__create_dir(created_dir)

            # create subdir for tensorboard logs
            self.__create_tensorboard_dir(created_dir)

        # set class instancy to hold abs path
        self.model_directory = created_dir

    def save_dict_to_json(self, json_path):
        """Saves dict of floats in json file
        Args:
            d: (dict) of float-castable values (np.float, int, float, etc.)
            json_path: (string) path to json file
        """
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: str(v) for k, v in self.params.dict.items()}
            json.dump(d, f, indent = 4)


class Evaluation():

    def __init__(self, params, filename, model, mode):
        self.params = params
        self.model_dir = params.model_directory
        self.mode = mode
        self.model = model
        self.model_input_shape = (1, params.img_shape, params.img_shape, 3)
        self.filename = filename
        self.image, self.label = self.__load_test_image()
        self.prediction = self.__predict_image()
        self.seg_cmap, self.seg_norm, self.bounds = self.color_mappings()
        self.jaccard = jaccard_score(self.label.flatten(), self.prediction.flatten(), average = None)

    def resize(self, im):
        desired_size = self.params.img_shape
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

    def __load_test_image(self):
        # load samples
        im = Image.open(os.path.join(self.params.data_path, "images", self.filename))
        lbl = Image.open(os.path.join(self.params.data_path, "masks", self.filename))

        im = np.array(im)
        lbl = np.array(lbl)

        # resize samples
        im_resized = self.resize(im)

        # if image grey scale, make 3 channel
        if len(im_resized.shape) == 2:
            im_resized = np.stack((im_resized,) * 3, axis = -1)

        lbl_resized = self.resize(lbl)

        # convert choroid to background
        # lbl_resized[lbl_resized == 10] = 0

        im_scaled = np.divide(im_resized, 255., dtype = np.float32)

        return im_scaled, lbl_resized.astype(int)

    def __predict_image(self):
        # get probability map
        pred = self.model.predict(self.image.reshape(self.model_input_shape))

        return np.argmax(pred, -1)[0, :, :].astype(int)


class TrainOps():
    def __init__(self, params):
        self.params = params
        self.dice = dice_loss
        self.gen_dice = gen_dice

    def lr_schedule(self, epoch):
        """Learning Rate Schedule
    
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
    
        # Arguments
            epoch (int): The number of epochs
    
        # Returns
            lr (float32): learning rate
        """
        lr = self.params.learning_rate

        if epoch > int(self.params.num_epochs * 0.8):
            lr *= 1e-3
        elif epoch > int(self.params.num_epochs * 0.6):
            lr *= 1e-2
        elif epoch > int(self.params.num_epochs * 0.4):
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def callbacks_(self):
        '''callbacks'''
        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        checkpoint = ModelCheckpoint(filepath = self.params.model_directory + "/weights.hdf5",
                                     monitor = 'val_accuracy',
                                     save_best_only = True,
                                     verbose = 1,
                                     mode = 'max',
                                     save_weights_only = False)

        tb = TensorBoard(log_dir = self.params.model_directory + "/tensorboard")

        csv_logger = CSVLogger(filename = self.params.model_directory + '/history.csv',
                               append = True,
                               separator = ",")

        es = EarlyStopping(monitor = 'val_accuracy', patience = 300)

        return [lr_scheduler, checkpoint, tb, csv_logger, es]


class EvalVolume():

    def __init__(self, params, path, model, mode):
        self.params = params
        self.model_dir = params.model_directory
        self.mode = mode
        self.model = model
        self.path = path
        self.filename = path.split("/")[-1]
        self.model_input_shape = (1, params.img_shape, params.img_shape, 3)
        self.image = self.load_volume()
        self.segmented_volume = self.segment_volume()
        self.seg_cmap, self.seg_norm, self.bounds = color_mappings()
        self.feature_dict = {"id": [], "0": [], "1": [], "2": [], "3": [], "4": [],
                             "5": [], "6": [], "7": [], "8": [], "9": [], "10": [],
                             "11": [], "12": [], "13": [], "14": [], "15": []}

    def resize(self, im):
        desired_size = self.params.img_shape
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

    def load_volume(self):

        vol = read_file(self.path).pixel_array
        resized_vol = np.zeros(shape = (vol.shape[0], 256, 256, 3), dtype = np.float32)
        for i in range(vol.shape[0]):
            # resize samples
            im_resized = self.resize(vol[i, :, :])

            # if image grey scale, make 3 channel
            if len(im_resized.shape) == 2:
                im_resized = np.stack((im_resized,) * 3, axis = -1)

            im_scaled = np.divide(im_resized, 255., dtype = np.float32)
            resized_vol[i, :, :, :] = im_scaled
        return resized_vol

    def __predict_image(self, img):
        # get probability map
        pred = self.model.predict(img.reshape(self.model_input_shape))
        return np.argmax(pred, -1)[0, :, :].astype(int)

    def segment_volume(self):
        predictions = np.zeros(shape = (49, 256, 256), dtype = np.uint8)
        for i in range(self.image.shape[0]):
            predictions[i, :, :] = self.__predict_image(self.image[i, :, :, :])
        self.plot_record(self.image[25, :, :, :], predictions[25, :, :])
        return predictions

    def feature_statistics(self):
        for i in range(self.segmented_volume.shape[0]):
            map_ = self.segmented_volume[i, :, :]

            # count features
            map_counts = np.unique(map_, return_counts = True)

            # add do dict
            for k, feature in enumerate(self.feature_dict.keys()):
                if feature == 'id':
                    self.feature_dict[feature].append(self.filename + "_{}".format(i))
                else:
                    if int(feature) in map_counts[0]:
                        self.feature_dict[feature].append(map_counts[1][map_counts[0].tolist().index(int(feature))])
                    else:
                        self.feature_dict[feature].append(0)
        return self.feature_dict

def data_split(ids):
    """
    @param ids: list of image names
    @type ids: list
    @return: three lists divided into train, validation and test split
    @rtype: list
    """
    shuffle(ids)
    n_records = len(ids)

    test_ids = ids[int(n_records * 0.9):-1]
    validation_ids = ids[int(len(ids) * 0.8):int(len(ids) * 0.9)]
    train_ids = ids[0:int(len(ids) * 0.8)]
    return train_ids, validation_ids, test_ids