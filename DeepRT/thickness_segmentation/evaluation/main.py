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
    '''
    im_batch, labels_batch, im_displayed \
        = lnf.get_clinic_train_data(im_dir=img_dir, seg_dir=label_dir, img_shape=params["img_shape"],
                                    batch_size=params["batch_size"])

    # Running next element in our graph will produce a batch of images
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(im_batch[0,:,:,:])
    plt.title(str(im_displayed[0]))

    plt.subplot(2, 2, 2)
    plt.imshow(labels_batch[0, :, :, 0])

    plt.subplot(2, 2, 3)
    plt.imshow(im_batch[1,:,:,:])
    plt.title(str(im_displayed[1]))

    plt.subplot(2, 2, 4)
    plt.imshow(labels_batch[1, :, :, 0])

    plt.show()
    '''

    '''get model'''
    inputs, outputs = model_fn(params["img_shape"])
    model = models.Model(inputs=[inputs], outputs=[outputs])

    '''Compile model'''
    adam = optimizers.Adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss=dice_loss, metrics=[iou])
    model.summary()

    '''train and save model'''
    save_model_path = os.path.join(params["save_path"],"weights.hdf5")
    cp = tf.tensorflow.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_loss',
                                            save_best_only=True, verbose=1,save_weights_only=False)

    learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                patience=5,
                                                verbose=0,
                                                factor=0.5,
                                                min_lr=0.0001)

    if params["continuing_training"] == True:
        '''Load models trained weights'''
        model = models.load_model(save_model_path, custom_objects={'dice_loss': dice_loss,"iou":iou})

    for i in range(0,params["epochs"]*len(ids_train)):
        im_batch_val, labels_batch_val, im_displayed_val \
            = lnf.get_clinic_train_data(im_dir=img_val_dir, seg_dir=label_val_dir, img_shape=params["img_shape"],
                                        batch_size=params["batch_size"])
        im_batch, labels_batch, im_displayed \
            = lnf.get_clinic_train_data(im_dir=img_dir, seg_dir=label_dir, img_shape=params["img_shape"],
                                        batch_size=params["batch_size"])

        print(np.unique(labels_batch_val, return_counts=True))
        history = model.fit(x=im_batch,
                            y=labels_batch,
                           steps_per_epoch=params["batch_size"]*100,
                           validation_data=(im_batch_val,labels_batch_val),
                           validation_steps=1,
                           callbacks=[cp,learning_rate_reduction])


    '''Visualize the training process'''
    dice = history.history['dice_loss']
    val_dice = history.history['val_dice_loss']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    np.save(os.path.join(params["save_path"],"train_loss"),np.array(loss))
    np.save(os.path.join(params["save_path"],"validation_loss"),np.array(val_loss))
    np.save(os.path.join(params["save_path"], "train_dice"), np.array(dice))
    np.save(os.path.join(params["save_path"], "validation_dice"), np.array(val_dice))


'''Set configurable parameters'''
learning_rates = [0.1,0.01]
loss_functions = ["dice_loss"]
for lr in learning_rates:
    for losses in loss_functions:
        print("Now training with lr:{} and loss:{}".format(lr,losses))
        params["learning_rate"] = lr
        params["loss_functions"] = losses

        main(params)