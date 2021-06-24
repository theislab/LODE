"""General utility functions"""
import os
import sys
from pathlib import Path
import pandas as pd
import glob
import math
import keras
from keras.optimizers.schedules import *

root_dir = "/home/icb/olle.holmberg/projects/LODE/feature_segmentation"
search_paths = [i for i in glob.glob(root_dir + "/*/*") if os.path.isdir(i)]

for sp in search_paths:
    sys.path.append(sp)

path_variable = Path(os.path.dirname(__file__))
sys.path.insert(0, str(path_variable))
sys.path.insert(0, str(path_variable.parent))

import json
import glob
import shutil
import os

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard, EarlyStopping
from config import DATA_SPLIT_PATH


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
        self.params = params

    def __create_dir(self, dir):
        os.makedirs(dir, exist_ok = True)

    def __create_main_directory(self):
        '''
        :return: create main log dir if not allready created
        '''
        if not os.path.isdir(self.log_dir):
            print("main logging dir does not exist, creating main logging dir ./logs")
            os.makedirs(self.log_dir)
        else:
            pass

    def __remove_empty_directories(self):

        # get current directories
        current_directories = glob.glob(self.log_dir + "/*")

        # check for each dir, if weight.hdf5 file is contained
        for current_directory in current_directories:
            if not os.path.isfile(os.path.join(current_directory, "model.h5")):
                # remove directory
                shutil.rmtree(current_directory)

    def create_model_directory(self, model_dir):
        '''
        :param logging_directory: string, gen directory for logging
        :return: None
        '''

        # create main dir if not exist
        self.__create_main_directory()

        # make new directory
        self.__create_dir(model_dir)

        # set class instancy to hold abs path
        self.model_directory = model_dir

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


def cast_params_types(params, model_path):
    """
    function takes params object and casts all numeric types to float. Then save the config file again
    Parameters
    ----------
    params : loaded config file
    model_path : directory of model

    Returns
    -------
    None
    """
    # cast data types to numeric
    params = params.dict
    for k in params.keys():
        try:
            int(params[k])
            params[k] = int(params[k])
        except ValueError:
            try:
                float(params[k])
                params[k] = float(params[k])
            except ValueError:
                print("Not an int or  float")

    with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
        json.dump(params, json_file)


class TrainOps:
    def __init__(self, params, num_records):
        self.params = params

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

    def step_decay(self, epoch):
        """
        Parameters
        ----------
        epoch :

        Returns
        -------

        """
        initial_lrate = self.params.learning_rate
        drop = 0.5
        epochs_drop = self.params.num_epochs // 8
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def exp_decay(self, epoch):
        if epoch <  self.params.num_epochs // 10:
            return self.params.learning_rate
        else:
            lr = self.params.learning_rate * math.exp(-0.05 * (epoch // 2))
            return max(lr, self.params.learning_rate*10e-3)

    def callbacks_(self):

        available_decays = ["exponential_decay", "step_decay"]
        print_str = f"learning rate decay not available, choose from {available_decays}"
        assert self.params.learning_rate_scheduel in ["exponential_decay", "step_decay"], print_str

        if self.params.learning_rate_scheduel == "exponential_decay":
            lr_scheduler = ExponentialDecay(
                initial_learning_rate = self.params.learning_rate,
                decay_steps = int(self.steps_per_epoch*self.params.num_epochs),
                decay_rate = 0.1)

        elif self.params.learning_rate_scheduel == "step_decay":
            lr_scheduler = ExponentialDecay(
                initial_learning_rate = self.params.learning_rate,
                decay_steps = int(self.steps_per_epoch*self.params.num_epochs),
                decay_rate = 0.1,
                staircase = True)

        checkpoint = ModelCheckpoint(filepath = self.params.model_directory + "/weights.hdf5",
                                     monitor = 'val_accuracy',
                                     save_best_only = True,
                                     verbose = 1,
                                     mode = 'max',
                                     save_weights_only = False)

        tb = TensorBoard(log_dir = self.params.model_directory + "/tensorboard", write_graph = False)

        csv_logger = CSVLogger(filename = self.params.model_directory + '/history.csv',
                               append = True,
                               separator = ",")

        es = EarlyStopping(monitor = 'val_accuracy', patience = 300)

        return [lr_scheduler, checkpoint, tb, csv_logger]


def data_split(ids, params):
    """
    @param ids: list of image names
    @type ids: list
    @return: three lists divided into train, validation and test split
    @rtype: list
    """
    if not params.load_train_records:
        # shuffle(ids)
        n_records = len(ids)

        test_ids = ids[int(n_records * 0.9):-1]
        validation_ids = ids[int(len(ids) * 0.8):int(len(ids) * 0.9)]
        train_ids = ids[0:int(len(ids) * 0.8)]
    else:
        train_ids = pd.read_csv(os.path.join(params.pretrained_model, "train_ids.csv"))["0"].tolist()
        validation_ids = pd.read_csv(os.path.join(params.pretrained_model, "validation_ids.csv"))["0"].tolist()
        test_ids = pd.read_csv(os.path.join(params.pretrained_model, "test_ids.csv"))["0"].tolist()

    if params.load_prepared_split:
        train_ids = pd.read_csv(os.path.join(DATA_SPLIT_PATH, "train_ids.csv"))["0"].tolist()
        validation_ids = pd.read_csv(os.path.join(DATA_SPLIT_PATH, "validation_ids.csv"))["0"].tolist()
        test_ids = pd.read_csv(os.path.join(DATA_SPLIT_PATH, "test_ids.csv"))["0"].tolist()

    return train_ids, validation_ids, test_ids


if __name__ == "__main__":
    print("successfully run")
