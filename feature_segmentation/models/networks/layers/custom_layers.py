from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Conv3D, GlobalAveragePooling2D, Reshape, Dense, \
    Permute, multiply, Multiply, GlobalAveragePooling1D, Concatenate, add

import tensorflow.keras.backend as K

from feature_segmentation.models.networks.layers.attn_augconv import augmented_conv2d


def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides = [1, 1], data_format = data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides = [1, 1], data_format = data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides = [1, 1], data_format = data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x


def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.get_shape().as_list()[-1]  # init._tensorflow.keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False)(se)
    se = Dense(filters, activation = 'sigmoid', kernel_initializer = 'he_normal', use_bias = False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def squeeze_excite_aline_block(tensor, params, ratio=16):
    init = tensor
    se = K.mean(init, [-2, -1])

    avec = Dense(se.get_shape()[-1] // 8, activation = 'relu', kernel_initializer = 'he_normal', use_bias = False)(se)
    avec = Dense(se.get_shape()[-1], activation = 'sigmoid', kernel_initializer = 'he_normal',
                 use_bias = False)(avec)
    init_se = Permute((3, 1, 2))(init)
    aline_scaled = multiply([init_se, avec])
    init_out = Permute((2, 3, 1))(aline_scaled)
    return init_out


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = "he_normal",
               padding = "same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = "he_normal",
               padding = "same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def lstm_aline_block(tensor, params):
    init = tensor
    filters = init.get_shape().as_list()[1]  # init._tensorflow.keras_shape[channel_axis]

    se = K.mean(init, -1)
    class_attention_vectors = []
    for i in range(filters):
        avec = LSTM(params.img_shape)(se[:, i, :])

        # avec = Reshape((1, -1))(avec)
        class_attention_vectors.append(Reshape((256, 256, 1))(avec))
    return Concatenate(-1)(class_attention_vectors)


def conv3d_block(input_tensor, n_filters, batchnorm=True):
    # first layer
    x = Conv3D(n_filters, 2, padding = "same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv3D(n_filters, 2, padding = "same")(x)

    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
