import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

from feature_segmentation.segmentation_config import TRAIN_DATA_PATH
from feature_segmentation.utils.utils import data_split, Params, Logging, TrainOps

params = Params("/home/olle/PycharmProjects/LODE/feature_segmentation/params.json")
logging = Logging("./logs", params)
trainops = TrainOps(params)

params.data_path = TRAIN_DATA_PATH

ids = os.listdir(os.path.join(params.data_path, "images"))

train_ids, validation_ids, test_ids = data_split(ids, params)


list_ds = tf.data.TextLineDataset([os.path.join(params.data_path, "images", id) for id in test_ids])
list_ds = list_ds.shuffle(len(test_ids), reshuffle_each_iteration=False)

def get_label():
    pass

def decode_image():
    pass

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

