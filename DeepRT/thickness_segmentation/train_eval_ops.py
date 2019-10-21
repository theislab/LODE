import tensorflow as tf
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


from sklearn.metrics import confusion_matrix
import numpy as np

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)
