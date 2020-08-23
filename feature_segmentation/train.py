import random
from feature_segmentation.utils.utils import Params, TrainOps, Logging, data_split
import os
from generators.generator_3d import DataGenerator
import keras.backend as K
from feature_segmentation.models.model import get_model
import pandas as pd
from random import shuffle
from segmentation.config import TRAIN_DATA_PATH

# load utils classes
from feature_segmentation.utils.plotting import plot_model_run_images

params = Params("params.json")
logging = Logging("./logs", params)
trainops = TrainOps(params)

params.data_path = TRAIN_DATA_PATH

ids = os.listdir(os.path.join(params.data_path, "images"))
train_ids, validation_ids, test_ids = data_split(ids)

print("number of train and test image are: ", len(train_ids), len(validation_ids))

# Generators
train_generator = DataGenerator(train_ids, params = params, is_training = True)
test_generator = DataGenerator(validation_ids, params = params, is_training = False)

for i in range(len(ids)):
    _, _ = train_generator.__getitem__(i)

# set model tries
model_configs = ["deep_unet"]

for model_config in model_configs:
    # set this iterations model
    params.model = model_config

    # create logging directory
    logging.create_model_directory()
    params.model_directory = logging.model_directory

    # saving model config file to model output dir
    logging.save_dict_to_json(logging.model_directory + "/config.json")

    pd.DataFrame(validation_ids).to_csv(os.path.join(logging.model_directory + "/validation_ids.csv"))
    pd.DataFrame(train_ids).to_csv(os.path.join(logging.model_directory + "/train_ids.csv"))
    pd.DataFrame(test_ids).to_csv(os.path.join(logging.model_directory + "/test_ids.csv"))

    # plot examples
    for k in range(2):
        record, name = train_generator.example_record()
        plot_model_run_images(record, model_dir = logging.model_directory, mode = "train", filename = name)

    for k in range(2):
        record, name = test_generator.example_record()
        plot_model_run_images(record, model_dir = logging.model_directory, mode = "test",
                              filename = "test_1_{}".format(name))

    # get model
    model = get_model(params)

    history = model.fit_generator(generator = train_generator,
                                  steps_per_epoch = int(len(train_ids) / (params.batch_size * 1)),
                                  epochs = params.num_epochs,
                                  validation_data = test_generator,
                                  validation_steps = int(len(validation_ids) / 1),
                                  callbacks = trainops.callbacks_(),
                                  use_multiprocessing = True,
                                  workers = 4)

    if K.backend() == 'tensorflow':
        K.clear_session()
