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
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import callbacks
import loading_numpy_functions as lnf
from params import params
sess = tf.InteractiveSession()
def main(params):
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

    save_model_path = os.path.join(params["save_path"],"weights.hdf5")
    '''get model'''
    inputs, outputs = model_fn(params["img_shape"])
    model = models.Model(inputs=[inputs], outputs=[outputs])
    adam = optimizers.Adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss=dice_loss, metrics=[dice_loss])
    model.load_weights(save_model_path)
    plt.figure(figsize=(10, 20))
    test_losses = []
    for i in range(0,16):
        im_batch_val, labels_batch_val, im_displayed_val \
            = lnf.get_clinic_train_data(im_dir=img_val_dir, seg_dir=label_val_dir, img_shape=params["img_shape"],
                                        batch_size=1)
        prediction = model.predict(im_batch_val)
        loss = model.evaluate(x=im_batch_val, y=labels_batch_val, batch_size=1, verbose=1, sample_weight=None, steps=None)
        test_losses.append(loss)
        plt.subplot(1, 3, 3 * i + 1)
        plt.imshow(im_batch_val[0,:,:,0])
        plt.title("Input image")
        plt.subplot(1, 3, 3 * i + 2)
        plt.imshow(labels_batch_val[0, :, :, 0])
        plt.title("Actual Mask")
        plt.subplot(1, 3, 3 * i + 3)
        plt.imshow(np.matrix.round(prediction[0,:, :, 0]))
        plt.title("Predicted Mask, dice score:{}".format(loss[0]))
        print("average dice score over all samples are:{}".format(np.mean(test_losses)))
        plt.suptitle("Examples of Input Image, Label, and Prediction")
        plt.show()
main(params)