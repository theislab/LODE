import random
from utils.utils import Params, TrainOps, Logging, data_split
import os

import keras.backend as K
from models.model import get_model
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
from segmentation_config import TRAIN_DATA_PATH
import cv2


# load utils classes
from utils.plotting import plot_image_label_prediction

params = Params("params.json")
logging = Logging("./logs", params)
trainops = TrainOps(params)

params.data_path = TRAIN_DATA_PATH

ids = os.listdir(os.path.join(params.data_path, "images"))
train_ids, validation_ids, test_ids = data_split(ids, params)

print("number of training examples are: ", len(train_ids))

if TRAIN_DATA_PATH.split("/")[-1] == "first_examples":
    train_ids = train_ids + test_ids
    pretraining = True

upsampling_factors, label_repr = get_class_distribution(self.label_path, list_IDs)
                    train_ids = deepcopy(list_IDs)
            for label in [5, 8, 13]:
                new_ids = upsample(train_ids, label, label_repr, upsampling_factors)
                train_ids = deepcopy(new_ids)
                upsampling_factors, label_repr = get_class_distribution(self.label_path, train_ids)

            self.list_IDs = new_ids

print("number of train and test image are: ", len(train_ids), len(validation_ids))

if params.model == "volumeNet":
    from generators.generator_3d import DataGenerator
else:
    from generators.generator_2d import DataGenerator

# Generators
train_generator = DataGenerator(train_ids, params=params, is_training=True,
                                pretraining=False, choroid_latest=params.choroid_latest)
test_generator = DataGenerator(validation_ids, params=params, is_training=False,
                               pretraining=False, choroid_latest=params.choroid_latest)

# set model tries
model_configs = ["deep_unet"]

for model_config in model_configs:
    # set this iterations model
    # params.model = model_config

    # create logging directory
    logging.create_model_directory()
    params.model_directory = logging.model_directory

    # saving model config file to model output dir
    logging.save_dict_to_json(logging.model_directory + "/config.json")

    pd.DataFrame(validation_ids).to_csv(os.path.join(logging.model_directory + "/validation_ids.csv"))
    pd.DataFrame(train_ids).to_csv(os.path.join(logging.model_directory + "/train_ids.csv"))
    pd.DataFrame(test_ids).to_csv(os.path.join(logging.model_directory + "/test_ids.csv"))

    # plot examples
    for k in range(100):
        record, name = train_generator.example_record()
        cv2.imwrite(logging.model_directory + f"train_image_{k}.png", record[0]*255)
        plt.imsave(logging.model_directory + f"train_label_{k}.png", record[1]*(255//15))

    # for k in range(len(test_ids)):
    #    test_generator.example_record()

    # get model
    model = get_model(params)
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train_generator),
                                  epochs=params.num_epochs,
                                  validation_data=test_generator,
                                  validation_steps=int(len(validation_ids) / 1),
                                  callbacks=trainops.callbacks_(),
                                  use_multiprocessing=False,
                                  workers=4)

    if K.backend() == 'tensorflow':
        K.clear_session()
