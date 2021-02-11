from keras.models import Model
from keras.layers import BatchNormalization, Activation, UpSampling2D, \
    Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input
from models.networks.layers.custom_layers import *

from feature_segmentation.models.networks.layers.attn_augconv import augmented_conv2d
from feature_segmentation.models.networks.layers.custom_layers import squeeze_excite_aline_block, lstm_aline_block

from keras.models import Model
from keras.layers import BatchNormalization, Activation, UpSampling2D, \
    Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input
from models.networks.layers.custom_layers import *


def unet(params):
    inputs = Input(shape = (params.img_shape, params.img_shape, 3))

    # contracting path
    c1 = conv2d_block(inputs, n_filters = params.n_filters * 1, kernel_size = 3, batchnorm = params.batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv2d_block(p1, n_filters = params.n_filters * 2, kernel_size = 3, batchnorm = params.batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)

    if params.attention:
        p2 = augmented_conv2d(p2, params.n_filters * 2, num_heads = 1)

    c3 = conv2d_block(p2, n_filters = params.n_filters * 4, kernel_size = 3, batchnorm = params.batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)

    if params.attention:
        p3 = augmented_conv2d(p3, params.n_filters * 4, num_heads = 1)

    c4 = conv2d_block(p3, n_filters = params.n_filters * 8, kernel_size = 3, batchnorm = params.batchnorm)
    p4 = MaxPooling2D(pool_size = (2, 2))(c4)

    if params.attention:
        p4 = augmented_conv2d(p4, params.n_filters * 8, num_heads = 1)

    c5 = conv2d_block(p4, n_filters = params.n_filters * 8, kernel_size = 3, batchnorm = params.batchnorm)
    p5 = MaxPooling2D(pool_size = (2, 2))(c5)
    p5 = Dropout(params.dropout)(p5)

    if params.attention:
        p5 = augmented_conv2d(p5, params.n_filters * 8, num_heads = 1)

    c7 = conv2d_block(p5, n_filters = params.n_filters * 8, kernel_size = 3, batchnorm = params.batchnorm)

    u7 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u7 = concatenate([u7, c5])
    c8 = conv2d_block(u7, n_filters = params.n_filters * 8, kernel_size = 3, batchnorm = params.batchnorm)

    if params.attention:
        c8 = augmented_conv2d(c8, params.n_filters * 8, num_heads = 1)

    u8 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u8 = concatenate([u8, c4])
    c9 = conv2d_block(u8, n_filters = params.n_filters * 8, kernel_size = 3, batchnorm = params.batchnorm)

    if params.attention:
        c9 = augmented_conv2d(c9, params.n_filters * 8, num_heads = 1)

    u9 = Conv2DTranspose(params.n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c9)
    u9 = concatenate([u9, c3])
    c10 = conv2d_block(u9, n_filters = params.n_filters * 4, kernel_size = 3, batchnorm = params.batchnorm)

    if params.attention:
        c10 = augmented_conv2d(c10, params.n_filters * 4, num_heads = 1)

    u10 = Conv2DTranspose(params.n_filters * 3, (3, 3), strides = (2, 2), padding = 'same')(c10)
    u10 = concatenate([u10, c2])
    c11 = conv2d_block(u10, n_filters = params.n_filters * 3, kernel_size = 3, batchnorm = params.batchnorm)

    u11 = Conv2DTranspose(params.n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c11)
    u11 = concatenate([u11, c1])
    c12 = conv2d_block(u11, n_filters = params.n_filters * 2, kernel_size = 3, batchnorm = params.batchnorm)

    c12 = squeeze_excite_aline_block(c12, params)
    outputs = Conv2D(params.num_classes, (1, 1), activation = 'softmax', name="15_class")(c12)

    model = Model(inputs = inputs, outputs = [outputs])
    return model