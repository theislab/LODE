from keras.models import Model
from keras.layers import BatchNormalization, Activation, UpSampling2D, \
    Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, GlobalAveragePooling2D, Reshape, Convolution2D, \
    AveragePooling2D
from models.networks.layers.custom_layers import *


def pyramid_feature_maps(base):
    # red
    red = GlobalAveragePooling2D(name = 'red_pool')(base)
    red = Reshape((1, 1, base.shape[-1]))(red)
    red = conv2d_block(input_tensor=red, n_filters = 128, kernel_size = 1)
    red = UpSampling2D(size = 32, interpolation = 'bilinear', name = 'red_upsampling')(red)

    # yellow
    yellow = AveragePooling2D(pool_size = (2, 2), name = 'yellow_pool')(base)
    yellow = conv2d_block(input_tensor=yellow, n_filters = 128, kernel_size = 1)
    yellow = UpSampling2D(size = 2, interpolation = 'bilinear', name = 'yellow_upsampling')(yellow)

    # blue
    blue = AveragePooling2D(pool_size = (4, 4), name = 'blue_pool')(base)
    blue = conv2d_block(input_tensor=blue, n_filters = 128, kernel_size = 1)
    blue = UpSampling2D(size = 4, interpolation = 'bilinear', name = 'blue_upsampling')(blue)

    # green
    green = AveragePooling2D(pool_size = (8, 8), name = 'green_pool')(base)
    green = conv2d_block(input_tensor=green, n_filters = 128, kernel_size = 1)
    green = UpSampling2D(size = 8, interpolation = 'bilinear', name = 'green_upsampling')(green)

    # base + red + yellow + blue + green
    return concatenate([red, yellow, blue, green])

def unet(params):
    inputs = Input(shape = (params.img_shape, params.img_shape, 3))

    # contracting path
    c1 = conv2d_block(inputs, n_filters = params.n_filters * 1, kernel_size = 3)
    p1 = AveragePooling2D((2, 2))(c1)

    c2 = conv2d_block(p1, n_filters = params.n_filters * 2, kernel_size = 3)
    p2 = AveragePooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, n_filters = params.n_filters * 4, kernel_size = 3)
    p3 = AveragePooling2D((2, 2))(c3)

    c4 = conv2d_block(p3, n_filters = params.n_filters * 8, kernel_size = 3)

    c7 = pyramid_feature_maps(c4)

    u8 = conv2d_block(c7, n_filters = params.n_filters * 8, kernel_size = 3)

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

    outputs = Conv2D(params.num_classes, (1, 1), activation = 'softmax', name="15_class")(c12)

    model = Model(inputs = inputs, outputs = [outputs])
    return model
