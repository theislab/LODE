from keras.models import Model
from keras.layers import BatchNormalization, Activation, UpSampling2D, \
    Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Input
from models.networks.layers.custom_layers import *


def unet(params):
    inputs = Input((params.img_shape, params.img_shape, 3))
    x = inputs
    depth =  params.depth
    features = params.n_filters
    skips = []

    for i in range(depth):
        # contracting path
        x = conv2d_block(x, n_filters = features, kernel_size = 3)
        skips.append(x)
        x = squeeze_excite_block(x)
        x = MaxPooling2D((2, 2))(x)

        if i in [0, 1, 2]:
            features = features * 2

    features = features * 2
    x = Dropout(params.dropout)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same')(x)

    for i in reversed(range(depth)):
        if i in [0, 1, 2, 4]:
            features = features // 2

        x = Conv2DTranspose(features, (3, 3), strides = (2, 2), padding = 'same')(x)
        x = conv2d_block(x, n_filters = features, kernel_size = 3)
        x = squeeze_excite_block(x)


    outputs = Conv2D(params.num_classes, (1, 1), activation = 'softmax', name="15_class")(x)
    model = Model(inputs = inputs, outputs = [outputs])
    return model
