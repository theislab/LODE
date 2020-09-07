import os
from feature_segmentation.utils.utils import Params
from keras.backend import int_shape
from keras.models import Model
from keras.layers import Reshape, Dropout, Conv2DTranspose, MaxPooling2D, \
    concatenate, Input, MaxPooling3D, AveragePooling3D
from config import PROJ_DIR
from feature_segmentation.models.networks.layers.custom_layers import *


def unet(params):
    inputs = Input(shape = (params.n_scans, params.img_shape, params.img_shape, 3))

    # contracting path
    c1 = conv3d_block(inputs, n_filters = params.n_filters * 1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = conv3d_block(p1, n_filters = params.n_filters * 2)
    p2 = MaxPooling3D((4, 2, 2))(c2)

    # reshape 3d image to 2d
    p2 = Reshape(target_shape = [int_shape(p2)[-3], int_shape(p2)[-2], int_shape(p2)[-1]])(p2)
    c3 = conv2d_block(p2, n_filters = params.n_filters * 4, kernel_size = 3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv2d_block(p3, n_filters = params.n_filters * 8, kernel_size = 3)
    p4 = MaxPooling2D(pool_size = (2, 2))(c4)

    c5 = conv2d_block(p4, n_filters = params.n_filters * 8, kernel_size = 3)
    p5 = MaxPooling2D(pool_size = (2, 2))(c5)
    p5 = Dropout(params.dropout)(p5)

    c7 = conv2d_block(p5, n_filters = params.n_filters * 8, kernel_size = 3)

    u7 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u7 = concatenate([u7, c5])
    c8 = conv2d_block(u7, n_filters = params.n_filters * 8, kernel_size = 3)

    u8 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u8 = concatenate([u8, c4])
    c9 = conv2d_block(u8, n_filters = params.n_filters * 8, kernel_size = 3)

    u9 = Conv2DTranspose(params.n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c9)
    u9 = concatenate([u9, c3])
    c10 = conv2d_block(u9, n_filters = params.n_filters * 4, kernel_size = 3)

    u10 = Conv2DTranspose(params.n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c10)

    # average pooling for concatenation
    c2_concat = AveragePooling3D(pool_size = (int_shape(c2)[-4], 1, 1))(c2)
    concat_shape = [int_shape(c2_concat)[-3], int_shape(c2_concat)[-2], int_shape(c2_concat)[-1]]
    c2_concat = Reshape(target_shape = concat_shape)(c2_concat)

    u10 = concatenate([u10, c2_concat])
    c11 = conv2d_block(u10, n_filters = params.n_filters * 2, kernel_size = 3)
    u11 = Conv2DTranspose(params.n_filters, (3, 3), strides = (2, 2), padding = 'same')(c11)

    # average pooling for concatenation
    c1_concat = AveragePooling3D(pool_size = (int_shape(c1)[-4], 1, 1))(c1)
    concat_shape = [int_shape(c1_concat)[-3], int_shape(c1_concat)[-2], int_shape(c1_concat)[-1]]
    c1_concat = Reshape(target_shape = concat_shape)(c1_concat)

    u11 = concatenate([u11, c1_concat])
    c12 = conv2d_block(u11, n_filters = params.n_filters * 2, kernel_size = 3)
    outputs = Conv2D(params.num_classes, (1, 1), activation = 'softmax', name = "15_class")(c12)

    model = Model(inputs = inputs, outputs = [outputs])
    return model


if __name__ == "__main__":
    params = Params(os.path.join(PROJ_DIR, "params.json"))
    unet(params)
