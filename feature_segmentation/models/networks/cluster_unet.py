import os
import sys
from pathlib import Path

path_variable = Path(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(path_variable, "layers"))
sys.path.insert(0, str(path_variable.parent))
sys.path.insert(0, str(path_variable.parent.parent))

from keras.models import Model
from keras.layers import BatchNormalization, Activation, UpSampling2D, \
    Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input
from custom_layers import *
from utils.utils import Params
from config import PROJ_DIR


def unet(params):
    inputs = Input(shape = (params.img_shape, params.img_shape, 3))

    # contracting path
    c1 = conv2d_block(inputs, n_filters = params.n_filters * 1, kernel_size = 3)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv2d_block(p1, n_filters = params.n_filters * 2, kernel_size = 3)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, n_filters = params.n_filters * 4, kernel_size = 3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv2d_block(p3, n_filters = params.n_filters * 8, kernel_size = 3)
    p4 = MaxPooling2D(pool_size = (2, 2))(c4)

    c5 = conv2d_block(p4, n_filters = params.n_filters * 8, kernel_size = 3)
    p5 = MaxPooling2D(pool_size = (2, 2))(c5)

    c6 = conv2d_block(p5, n_filters = params.n_filters * 16, kernel_size = 3)
    p6 = MaxPooling2D(pool_size = (2, 2))(c6)
    p6 = Dropout(params.dropout)(p6)

    c7 = conv2d_block(p6, n_filters = params.n_filters * 8, kernel_size = 3)
    c7 = conv2d_block(c7, n_filters = params.n_filters * 2, kernel_size = 3)
    c7 = conv2d_block(c7, n_filters = params.n_filters * 1, kernel_size = 3)
    c7 = conv2d_block(c7, n_filters = params.n_filters // 2, kernel_size = 3)
    c7 = conv2d_block(c7, n_filters = params.n_filters // 4, kernel_size = 3)

    bottle_neck = conv2d_block(c7, n_filters = params.n_filters // 16, kernel_size = 3)

    c7 = conv2d_block(bottle_neck, n_filters = params.n_filters // 4, kernel_size = 3)
    c7 = conv2d_block(c7, n_filters = params.n_filters // 2, kernel_size = 3)
    c7 = conv2d_block(c7, n_filters = params.n_filters * 1, kernel_size = 3)
    c7 = conv2d_block(c7, n_filters = params.n_filters * 2, kernel_size = 3)
    c7 = conv2d_block(c7, n_filters = params.n_filters * 8, kernel_size = 3)

    u7 = Conv2DTranspose(params.n_filters * 16, (3, 3), strides = (2, 2), padding = 'same')(c7)
    c7 = conv2d_block(u7, n_filters = params.n_filters * 8, kernel_size = 3)

    u7 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c7)
    c8 = conv2d_block(u7, n_filters = params.n_filters * 8, kernel_size = 3)

    u8 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c8)
    c9 = conv2d_block(u8, n_filters = params.n_filters * 8, kernel_size = 3)

    u9 = Conv2DTranspose(params.n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c9)
    c10 = conv2d_block(u9, n_filters = params.n_filters * 4, kernel_size = 3)

    u10 = Conv2DTranspose(params.n_filters * 3, (3, 3), strides = (2, 2), padding = 'same')(c10)
    c11 = conv2d_block(u10, n_filters = params.n_filters * 3, kernel_size = 3)

    u11 = Conv2DTranspose(params.n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c11)
    c12 = conv2d_block(u11, n_filters = params.n_filters * 2, kernel_size = 3)

    outputs = Conv2D(params.num_classes, (1, 1), activation = 'softmax')(c12)

    model = Model(inputs = inputs, outputs = [outputs])
    return model

if __name__ == "__main__":
    params = Params(os.path.join(PROJ_DIR, "params.json"))
    unet(params)
