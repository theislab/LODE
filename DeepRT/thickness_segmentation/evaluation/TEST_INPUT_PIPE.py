import os
import glob
import zipfile
import functools
from model import *
#from evaluation import *
from train_eval_ops import *
import numpy as np
import matplotlib.pyplot as plt
from input import batch_data_sets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.tensorflow.keras import optimizers
from tensorflow.python.tensorflow.keras import layers
from tensorflow.python.tensorflow.keras import models
from tensorflow.python.tensorflow.keras import callbacks
import loading_numpy_functions as lnf
from params import params
sess = tf.InteractiveSession()

'''create save dir if does not exist'''
try:
    os.stat(params["save_path"])
except:
    os.makedirs(params["save_path"])

'''load data files'''
img_dir = os.path.join(params["data_dir"], "train_images")
label_dir = os.path.join(params["data_dir"], "train_labels")
img_val_dir = os.path.join(params["data_dir"], "test_images")
label_val_dir = os.path.join(params["data_dir"], "test_labels")
ids_train = [i for i in os.listdir(img_dir)]
ids_val = [i for i in os.listdir(img_val_dir)]
num_training_examples = len(ids_train)

im_batch, labels_batch, im_displayed \
    = lnf.get_clinic_train_data(im_dir=img_dir, seg_dir=label_dir, img_shape=params["img_shape"],
                                batch_size=params["batch_size"])

# Running next element in our graph will produce a batch of images
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(im_batch[0,:,:,:])
plt.title(str(im_displayed[0]))
print(str(im_displayed[0]))

plt.subplot(2, 2, 2)
plt.imshow(labels_batch[0, :, :, 0])

plt.subplot(2, 2, 3)
plt.imshow(im_batch[1,:,:,:])
plt.title(str(im_displayed[1]))
print(str(im_displayed[1]))
plt.subplot(2, 2, 4)
plt.imshow(labels_batch[1, :, :, 0])

plt.show()



for i in range(0,params["epochs"]*len(ids_train)):
    im_batch_val, labels_batch_val, im_displayed_val \
        = lnf.get_clinic_train_data(im_dir=img_val_dir, seg_dir=label_val_dir, img_shape=params["img_shape"],
                                    batch_size=params["batch_size"])
    im_batch, labels_batch, im_displayed \
        = lnf.get_clinic_train_data(im_dir=img_dir, seg_dir=label_dir, img_shape=params["img_shape"],
                                    batch_size=params["batch_size"])

