from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Conv2DTranspose, MaxPooling2D, concatenate, Input
from models.networks.layers.custom_layers import *


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
    p4 = Dropout(params.dropout)(p4)

    c7 = conv2d_block(p4, n_filters = params.n_filters * 8, kernel_size = 3)

    u8 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c4])
    c9 = conv2d_block(u8, n_filters = params.n_filters * 8, kernel_size = 3)

    u9 = Conv2DTranspose(params.n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c9)
    u9 = concatenate([u9, c3])
    c10 = conv2d_block(u9, n_filters = params.n_filters * 4, kernel_size = 3)

    u10 = Conv2DTranspose(params.n_filters * 3, (3, 3), strides = (2, 2), padding = 'same')(c10)
    u10 = concatenate([u10, c2])
    c11 = conv2d_block(u10, n_filters = params.n_filters * 3, kernel_size = 3)

    u11 = Conv2DTranspose(params.n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c11)
    u11 = concatenate([u11, c1])
    c12 = conv2d_block(u11, n_filters = params.n_filters * 2, kernel_size = 3)

    c13 = Conv2D(params.num_classes, (3, 3), activation='relu', name="15_class_pre", padding="same")(c12)

    c13_aa = squeeze_excite_aline_block(c13, params)

    c14 = Conv2D(params.num_classes, (3, 3), activation='relu', name="15_class", padding="same")(c13_aa)

    c15 = Concatenate()([c12, c14])

    outputs = Conv2D(params.num_classes, (1, 1), activation='softmax', name="15_class_final")(c15)

    model = Model(inputs = inputs, outputs = [outputs])
    return model