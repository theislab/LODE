from keras.models import Model
from keras.layers import BatchNormalization, Activation, UpSampling2D, \
    Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input
from models.networks.layers.custom_layers import *


def unet(params):
    inputs = Input(shape = (params.img_shape, params.img_shape, 3))

    features = params.n_filters
    x = inputs
    skips = []
    for i in range(params.depth):
        # contracting path
        x = conv2d_block(x, n_filters = features, kernel_size = 3)
        skips.append(x)
        x = squeeze_excite_block(x)
        x = MaxPooling2D((2, 2))(x)
        features = features * 2

    x = conv2d_block(x, n_filters = features, kernel_size = 3)

    for i in reversed(range(params.depth)):
        features = features // 2
        u7 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u7 = concatenate([u7, c5])
        c8 = conv2d_block(u7, n_filters = params.n_filters * 8, kernel_size = 3)

        
        # attention_up_and_concate(x,[skips[i])
        layer = UpSampling2D(size = (2, 2), data_format = data_format)(x)
        x = concatenate([skips[i], x], axis = 1)
        x = Conv2D(features, (3, 3), activation = 'relu', padding = 'same', data_format = data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation = 'relu', padding = 'same', data_format = data_format)(x)

    u7 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u7 = concatenate([u7, c5])
    c8 = conv2d_block(u7, n_filters = params.n_filters * 8, kernel_size = 3)

    u8 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c8)
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
