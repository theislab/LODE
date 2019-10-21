from __future__ import print_function
from keras.optimizers import adam
import cv2
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
import model as mt
from keras.callbacks import ReduceLROnPlateau
from train_eval_ops import *
from keras.regularizers import l2
from keras.models import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import os

def crop_image(img,cond, tol=0):
    # img is image data
    # tol  is tolerance
    mask = cond>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def resnet_layer(inputs,
                 num_filters=32,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=1):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 32
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = AveragePooling2D(pool_size=8)(x)

    return outputs, inputs

tm_dir = "/home/olle/PycharmProjects/thickness_prediction/thickness_maps"
f_dir = "/home/olle/PycharmProjects/thickness_prediction/fundus"

files_ = pd.read_csv("./full_export_file_names/y_test_filenames.csv")['0'].values.tolist()#[i.replace(".npy","") for i in os.listdir(tm_dir)]

n = 3
# Computed depth from supplied model parameter n
depth = n * 9 + 2
#read model
res_output, inputs = resnet_v2(input_shape=(128,128,3), depth=depth)
outputs = mt.decoder(res_output, n_filters=32, dropout=0.05, batchnorm=True)
model = Model(inputs=inputs, outputs=[outputs])
'''Compile model'''
adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam, loss=custom_mae, metrics=[custom_mae,percentual_deviance])
model.summary()

'''train and save model'''
save_model_path = os.path.join("./_20_imagenet",
                               "weights.hdf5")

'''Load models trained weights'''
model.load_weights(save_model_path)
mae_list = []

print("number of test files:",len(files_))
for i in range(0,1000):
    print(i)
    #load samples
    im = cv2.imread(os.path.join(f_dir, files_[i].replace(".npy",".png")))
    lbl = np.load(os.path.join(tm_dir, files_[i]))

    # convert to three channel
    if im.shape[-1] != 3:
        im = np.stack((im,) * 3, axis=-1)

    # remove non labeled part of image
    im[lbl == 0] = 0

    # crop images
    c_im = crop_image(im, lbl, tol=0)
    c_lbl = crop_image(lbl, lbl, tol=0)

    if (c_im.shape[0] > 0 and c_im.shape[1] > 0):
        # adjust light
        c_im = cv2.addWeighted(c_im, 4, cv2.GaussianBlur(c_im, (0, 0), 10), -4, 128)

        # resize samples
        im_resized = cv2.resize(c_im, (128,128)).reshape(params["img_shape"])
        lbl_resized = cv2.resize(c_lbl, (128,128)).reshape(params["img_shape"][0], params["img_shape"][1], 1)

        # scaling
        label_im = np.divide(lbl_resized, 500., dtype=np.float32)
        train_im = np.divide(im_resized, 255., dtype=np.float32)

        # Train model on dataset
        prediction = model.predict(train_im.reshape(1,128,128,3))

        predicted_thickness_map = prediction[0,:,:,0]
        label = label_im[:,:,0]

        mae = np.abs(np.mean(predicted_thickness_map * 500. - label * 500.))
        mae_list.append(mae)
        #print("mae is:",np.abs(np.mean(predicted_thickness_map*500.-label*500.)))

        plt.imsave("./predictions/"+files_[i]+"_label.png",label*500.,cmap=plt.cm.jet)
        plt.imsave("./predictions/"+files_[i]+"_prediction.png",predicted_thickness_map*500.,cmap=plt.cm.jet)


print("test mae is:", np.mean(mae_list))