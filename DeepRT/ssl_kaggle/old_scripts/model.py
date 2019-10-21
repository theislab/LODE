import tensorflow as tf
from params import *
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate
from keras import regularizers


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_bunet(input_img, n_filters=16, dropout=0.5, batchnorm=True, training =True):

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1, training=training)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2, training=training)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3, training=training)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4, training=training)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)
    p5 = Dropout(dropout)(p5, training=training)

    c6 = conv2d_block(p5, n_filters=n_filters * 32, kernel_size=3, batchnorm=batchnorm)
    p6 = MaxPooling2D(pool_size=(2, 2))(c6)
    p6 = Dropout(dropout)(p6, training=training)

    c7 = conv2d_block(p6, n_filters=n_filters * 64, kernel_size=3, batchnorm=batchnorm)

    c8 = GlobalAveragePooling2D(data_format=None)(c7)

    d4 = Dense(5,kernel_initializer='normal',activation="softmax")(c8)

    outputs = d4

    model = Model(inputs=input_img, outputs=[outputs])
    return model
