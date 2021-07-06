from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, UpSampling2D, \
    Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input, GlobalAveragePooling2D, Reshape, Convolution2D, \
    AveragePooling2D
from models.networks.layers.custom_layers import *


def pyramid_feature_maps(base, pyramid):
    NUM_FILTERS = 128
    base_shape = base.shape[1]
    # red
    red = GlobalAveragePooling2D(name = f'red_pool_{pyramid}')(base)
    red = Reshape((1, 1, base.shape[-1]))(red)
    red = conv2d_block(input_tensor=red, n_filters = NUM_FILTERS, kernel_size = 1)
    red = Conv2DTranspose(filters = NUM_FILTERS, strides = (1, 1), kernel_size = base_shape, name = f'red_upsampling_{pyramid}')(red)

    # yellow
    yellow = AveragePooling2D(pool_size = (2, 2), name = f'yellow_pool_{pyramid}')(base)
    yellow = conv2d_block(input_tensor=yellow, n_filters = NUM_FILTERS, kernel_size = 1)
    yellow = Conv2DTranspose(filters = NUM_FILTERS, strides = (2, 2), kernel_size = 1, padding = "same", name = f'yellow_upsampling_{pyramid}')(yellow)

    # blue
    blue = AveragePooling2D(pool_size = (4, 4), name = f'blue_pool_{pyramid}')(base)
    blue = conv2d_block(input_tensor=blue, n_filters = NUM_FILTERS, kernel_size = 1)
    blue = Conv2DTranspose(filters = NUM_FILTERS, strides = (4, 4), kernel_size = 1, name = f'blue_upsampling_{pyramid}')(blue)
    # green
    green = AveragePooling2D(pool_size = (8, 8), name = f'green_pool_{pyramid}')(base)
    green = conv2d_block(input_tensor=green, n_filters = NUM_FILTERS, kernel_size = 1)
    green = Conv2DTranspose(filters = NUM_FILTERS, strides = (8, 8), kernel_size = 1, name = f'green_upsampling_{pyramid}')(green)

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
    p4 = AveragePooling2D(pool_size = (2, 2))(c4)

    c5 = conv2d_block(p4, n_filters = params.n_filters * 8, kernel_size = 3, batchnorm = params.batchnorm)
    p5 = AveragePooling2D(pool_size = (2, 2))(c5)
    p5 = Dropout(params.dropout)(p5)

    c6 = pyramid_feature_maps(p5, pyramid="1")
    c7 = pyramid_feature_maps(c4, pyramid="2")

    u7 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c5])
    u8 = conv2d_block(u7, n_filters = params.n_filters * 8, kernel_size = 3)

    u8 = concatenate([u8, c5])
    c9 = conv2d_block(u8, n_filters = params.n_filters * 8, kernel_size = 3)

    u9 = Conv2DTranspose(params.n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c9)
    u9 = concatenate([u9, c7])
    c10 = conv2d_block(u9, n_filters = params.n_filters * 4, kernel_size = 3)

    u10 = Conv2DTranspose(params.n_filters * 3, (3, 3), strides = (2, 2), padding = 'same')(c10)
    u10 = concatenate([u10, c3])
    c11 = conv2d_block(u10, n_filters = params.n_filters * 3, kernel_size = 3)

    u11 = Conv2DTranspose(params.n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c11)
    u11 = concatenate([u11, c2])
    c12 = conv2d_block(u11, n_filters = params.n_filters * 2, kernel_size = 3)

    u12 = Conv2DTranspose(params.n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c12)
    u12 = concatenate([u12, c1])
    c13 = conv2d_block(u12, n_filters = params.n_filters * 2, kernel_size = 3)

    outputs = Conv2D(params.num_classes, (1, 1), activation = 'softmax', name="15_class")(c13)

    model = Model(inputs = inputs, outputs = [outputs])
    return model
