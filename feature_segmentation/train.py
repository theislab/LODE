import random
import os
import pandas as pd
import keras.backend as K

from pathlib import Path
import sys

path = Path(os.getcwd())
sys.path.append(str(path.parent))

# add children paths
for child_dir in [p for p in path.glob("**/*") if p.is_dir()]:
    sys.path.append(str(child_dir))

from models.model import get_model
from segmentation_config import TRAIN_DATA_PATH
from utils.utils import Params, TrainOps, Logging, data_split
from generator_2d import DataGenerator

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

print("number of train and test image are: ", len(train_ids), len(validation_ids))

from generators.generator_2d import DataGenerator

# Generators
train_generator = DataGenerator(train_ids[0:5], params=params, is_training=True,
                                pretraining=False, choroid_latest=params.choroid_latest)
test_generator = DataGenerator(validation_ids[0:5], params=params, is_training=False,
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
