import matplotlib as mpl
from params import params
import keras.backend as K
import tensorflow as tf
from tensorflow.python.keras import losses
from keras.layers import Input
import model as mt
import os

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def instantiate_bunet(params,adam, training):
    '''
    :param params: params stating config info
    :param opt: an optimizer for the network
    :return: model object for prediction
    '''

    '''get model'''
    input_img = Input(params["img_shape"], name='img')
    model = mt.get_bunet(input_img, n_filters=16, dropout=0.5, batchnorm=True, training=training)

    adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    '''Compile model'''
    model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=[iou])

    model.summary()

    '''train and save model'''
    save_model_path = os.path.join(params["save_path"], "weights.hdf5")
    model.load_weights(save_model_path)

    return model
