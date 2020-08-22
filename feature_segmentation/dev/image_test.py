from keras.layers import Input
import pandas as pd
from keras.optimizers import SGD
from utils import Params
from utils import Logging
from utils import Evaluation
import model as m
import os
from PIL import Image
import numpy as np
import keras as K
import tensorflow as tf
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
model_dir = "./model"
params = Params("params.json")
logging = Logging("./model",params)

train_file_names = "./filenames/train_records.csv"
validation_file_names = "./filenames/validation_records.csv"
test_file_names = "./filenames/test_records.csv"

train_ids = pd.read_csv(train_file_names)["0"]
validation_ids = pd.read_csv(validation_file_names)["0"]
test_ids = pd.read_csv(test_file_names)["0"]

num_train_examples = train_ids.shape[0]
num_val_examples = validation_ids.shape[0]
num_test_examples = test_ids.shape[0]

image_path = "/home/olle/PycharmProjects/clinical_feature_segmentation/data/images"
label_path = "/home/olle/PycharmProjects/clinical_feature_segmentation/data/labels"


def resize(im):
    desired_size = 512
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

#
filename = test_ids[0]
im_path = os.path.join(image_path,filename)

im = Image.open(im_path.replace(".png",".jpg"))
im = np.array(im)
im_resized = resize(im)