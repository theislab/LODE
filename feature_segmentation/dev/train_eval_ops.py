
from params import params


from keras.layers import Input
import model_test as mt
from train_eval_ops import *
from keras.optimizers import adam

import numpy as np
import os
import keras.backend as K
import tensorflow as tf
from tensorflow.python.keras import losses
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

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def multidice(y_true, y_pred):
    one_hot = tf.one_hot(tf.cast(y_true, tf.int32), 3)[:, :, :, 0, :]
    return (dice_loss(one_hot[:,:,:,0], y_pred[:,:,:,0]) + \
           dice_loss(one_hot[:,:,:,1], y_pred[:,:,:,1]) + \
           dice_loss(one_hot[:,:,:,2], y_pred[:,:,:,2]))/3.


def generalized_dice_loss(y_true, y_pred):
    one_hot = tf.one_hot(tf.cast(y_true, tf.int32), params["number_of_classes"])[:,:,:,0,:]

    ref_vol = tf.reduce_sum(one_hot, 0)
    intersect = tf.reduce_sum(one_hot * y_pred,
                                     0)
    seg_vol = tf.reduce_sum(y_pred, 0)

    weights = tf.reciprocal(tf.square(ref_vol))

    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))

    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator

    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)

    del seg_vol, ref_vol, intersect, weights
    return 1 - generalised_dice_score

def two_dice(y_true, y_pred):
    y_flat = K.argmax(K.reshape(y_pred, [-1, 3]), axis=1)
    y_true_flat = K.reshape(y_true, [-1])

    one_hot_pred = tf.one_hot(tf.cast(y_flat, tf.int32), 3)
    one_hot = tf.one_hot(tf.cast(y_true_flat, tf.int32), 3)

    y_pred_two = one_hot_pred[:, 2]
    y_true_two = one_hot[:, 2]

    two_dice = dice_coeff(y_true_two, y_pred_two)

    return two_dice

def one_dice(y_true, y_pred):
    y_flat = K.argmax(K.reshape(y_pred, [-1, 3]), axis=1)
    y_true_flat = K.reshape(y_true, [-1])

    one_hot_pred = tf.one_hot(tf.cast(y_flat, tf.int32), 3)
    one_hot = tf.one_hot(tf.cast(y_true_flat, tf.int32), 3)

    y_pred_one = one_hot_pred[:, 1]

    y_true_one = one_hot[:, 1]

    one_dice = dice_coeff(y_true_one, y_pred_one)

    return one_dice

def iou(y_true,y_pred):
    #y_true = K.print_tensor(y_true, message='y_true = ')
    #y_pred = K.print_tensor(y_pred, message='y_pred = ')
    num_labels = K.int_shape(y_pred)[-1]
    y_flat = K.argmax(K.reshape(y_pred, [-1, num_labels]), axis=1)
    y_true_flat = K.reshape(y_true, [-1])
    predictions = tf.one_hot(tf.cast(y_flat, tf.int32), num_labels)
    labels = tf.one_hot(tf.cast(y_true_flat,tf.int32), num_labels)
    class_scores = []
    for i in range(num_labels):
        intersection = tf.reduce_sum(labels[:,i] * predictions[:,i])
        union = tf.math.count_nonzero(labels[:, i] + predictions[:, i])
        iou = tf.divide(tf.cast(intersection,tf.float32),tf.cast(union,tf.float32)+1.0)
        class_scores.append(iou)
    return tf.divide(tf.reduce_sum(class_scores),num_labels)


def segmentation_loss(logits, labels, class_weights=None):
    """
    Segmentation loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size * height * width, num_classes]
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None
    Returns:
        segment_loss: Segmentation loss
    """

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='segment_cross_entropy_per_example')

    if class_weights is not None:
        weights = tf.matmul(labels, class_weights, a_is_sparse=True)
        weights = tf.reshape(weights, [-1])
        cross_entropy = tf.multiply(cross_entropy, weights)

    segment_loss = tf.reduce_mean(cross_entropy, name='segment_cross_entropy')

    tf.summary.scalar("loss/segmentation", segment_loss)

    return segment_loss


def l2_loss():
    """
    L2 loss:
    -------
    Returns:
        l2_loss: L2 loss for all weights
    """

    weights = [var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

    tf.summary.scalar("loss/weights", l2_loss)

    return l2_loss


def loss(y_pred, y_true, weight_decay_factor=0.99, class_weights=None):
    """
    Total loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size * height * width,  num_classes]
        labels: Tensor, ground truth [batch_size, height, width, 1]
        weight_decay_factor: float, factor with which weights are decayed
        class_weights: Tensor, weighting of class for loss [num_classes, 1] or None
    Returns:
        total_loss: Segmentation + Classification losses + WeightDecayFactor * L2 loss
    """
    logits = tf.reshape(y_pred, (-1,params["number_of_classes"]))
    labels = tf.reshape(y_true, [-1])
    labels = tf.one_hot(tf.cast(labels,tf.int32), params["number_of_classes"])

    segment_loss = segmentation_loss(logits, labels, class_weights)
    total_loss = segment_loss #+ weight_decay_factor * l2_loss()

    tf.summary.scalar("loss/total", total_loss)

    return total_loss

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
