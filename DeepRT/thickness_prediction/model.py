import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.utils.data_utils import get_file
from keras.regularizers import l2


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



def rettNet(input_shape, n_filters=16, dropout=1.0, batchnorm=True, training=True):
    inputs = Input(shape=(input_shape, input_shape, 3))
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths

    conv1 = keras.layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, bias_constraint=None)(inputs)

    b1 = BatchNormalization()(conv1)
    r1 = Activation('relu')(b1)

    max1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(r1)

    conv2 = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, bias_constraint=None)(max1)

    b2 = BatchNormalization()(conv2)
    r2 = Activation('relu')(b2)

    conv3 = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, bias_constraint=None)(r2)

    b3 = BatchNormalization()(conv3)
    r3 = Activation('relu')(b3)

    max2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(r3)

    conv4 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, bias_constraint=None)(max2)

    b4 = BatchNormalization()(conv4)
    r4 = Activation('relu')(b4)

    conv5 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, bias_constraint=None)(r4)

    b5 = BatchNormalization()(conv5)
    r5 = Activation('relu')(b5)

    max3 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(r5)

    conv6 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, bias_constraint=None)(max3)

    b6 = BatchNormalization()(conv6)
    r6 = Activation('relu')(b6)

    conv7 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, bias_constraint=None)(r6)

    b7 = BatchNormalization()(conv7)
    r7 = Activation('relu')(b7)

    conv8 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, bias_constraint=None)(r7)

    b8 = BatchNormalization()(conv8)
    r8 = Activation('relu')(b8)

    conv9 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros',
                                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                kernel_constraint=None, bias_constraint=None)(r8)

    b9 = BatchNormalization()(conv9)
    r9 = Activation('relu')(b9)

    max4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(r9)

    conv10 = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1),
                                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)(max4)

    b10 = BatchNormalization()(conv10)
    r10 = Activation('relu')(b10)

    conv11 = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1),
                                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)(r10)

    b11 = BatchNormalization()(conv11)
    r11 = Activation('relu')(b11)

    conv12 = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1),
                                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)(r11)

    b12 = BatchNormalization()(conv12)
    r12 = Activation('relu')(b12)

    conv13 = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1),
                                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)(r12)

    b13 = BatchNormalization()(conv13)
    r13 = Activation('relu')(b13)

    max4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format=None)(r13)

    drop1 = keras.layers.Dropout(0.5, noise_shape=None, seed=None)(max4)

    conv14 = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1),
                                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)(drop1)

    b14 = BatchNormalization()(conv14)
    r14 = Activation('relu')(b14)

    drop2 = keras.layers.Dropout(0.5, noise_shape=None, seed=None)(r14)

    conv15 = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1),
                                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)(drop2)

    b15 = BatchNormalization()(conv15)
    r15 = Activation('relu')(b15)

    drop3 = keras.layers.Dropout(0.5, noise_shape=None, seed=None)(r15)

    conv16 = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)(drop3)

    b16 = BatchNormalization()(conv16)
    r16 = Activation('relu')(b16)

    drop4 = keras.layers.Dropout(0.5, noise_shape=None, seed=None)(r16)

    conv17 = keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)(drop4)

    b17 = BatchNormalization()(conv17)
    r17 = Activation('relu')(b17)

    drop5 = keras.layers.Dropout(0.5, noise_shape=None, seed=None)(r17)

    conv18 = keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1),
                                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                                 bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)(drop5)

    # expansive path
    u7 = Conv2DTranspose(n_filters * 32, (3, 3), strides=(2, 2), padding='same')(conv18)
    u7 = concatenate([u7, conv13])
    u7 = Dropout(dropout)(u7, training=training)
    c8 = conv2d_block(u7, n_filters=n_filters * 32, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 16, (3, 3), strides=(2, 2), padding='same')(c8)
    u8 = concatenate([u8, conv9])
    u8 = Dropout(dropout)(u8, training=training)
    c9 = conv2d_block(u8, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c9)
    u9 = concatenate([u9, conv5])
    u9 = Dropout(dropout)(u9, training=training)
    c10 = conv2d_block(u9, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u10 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c10)
    u10 = concatenate([u10, conv3])
    u10 = Dropout(dropout)(u10, training=training)
    c11 = conv2d_block(u10, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u11 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c11)
    u11 = concatenate([u11, conv1])
    u11 = Dropout(dropout)(u11, training=training)
    c12 = conv2d_block(u11, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u12 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c12)
    u12 = Dropout(dropout)(u12, training=training)
    c13 = conv2d_block(u12, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='linear')(c13)

    return inputs, outputs
def resnet_layer(inputs,
                 num_filters=16,
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

def stage(x, num_filters_in, stage, num_res_blocks):
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
            if stage < 1:
                num_filters_out = num_filters_in * 2
            else:
                num_filters_out = num_filters_in
            if res_block == 0:  # first layer but not first stage
                strides = 2  # downsample

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

    return (x, num_filters_in)


def ressttNet(input_shape, n_filters=16, dropout=1.0, batchnorm=True, training=True):

    depth = 2 * 9 + 2

    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 8
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=(input_shape, input_shape, 3))
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    #for stage in range(6):

    stage1, num_filters_in = stage(x, num_filters_in, stage=0, num_res_blocks=2)
    stage2, num_filters_in = stage(stage1, num_filters_in, stage=1, num_res_blocks=2)
    stage3, num_filters_in = stage(stage2, num_filters_in, stage=2, num_res_blocks=2)
    stage4, num_filters_in = stage(stage3, num_filters_in, stage=3, num_res_blocks=2)
    stage5, num_filters_in = stage(stage4, num_filters_in, stage=4, num_res_blocks=2)
    stage6, num_filters_in = stage(stage5, num_filters_in, stage=4, num_res_blocks=2)
    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(stage6)
    x = Activation('relu')(x)

    # expansive path
    u7 = Conv2DTranspose(n_filters * 32, (3, 3), strides=(2, 2), padding='same')(x)
    u7 = concatenate([u7, stage5])
    u7 = Dropout(dropout)(u7, training=training)
    c8 = conv2d_block(u7, n_filters=n_filters * 32, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 16, (3, 3), strides=(2, 2), padding='same')(c8)
    u8 = concatenate([u8, stage4])
    u8 = Dropout(dropout)(u8, training=training)
    c9 = conv2d_block(u8, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c9)
    u9 = concatenate([u9, stage3])
    u9 = Dropout(dropout)(u9, training=training)
    c10 = conv2d_block(u9, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u10 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c10)
    u10 = concatenate([u10, stage2])
    u10 = Dropout(dropout)(u10, training=training)
    c11 = conv2d_block(u10, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u11 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c11)
    u11 = concatenate([u11, stage1])
    u11 = Dropout(dropout)(u11, training=training)
    c12 = conv2d_block(u11, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='linear')(c12)

    model = Model(inputs=inputs, outputs=[outputs])
    return model
