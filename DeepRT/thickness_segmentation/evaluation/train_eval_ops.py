import matplotlib as mpl
from params import *
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.tensorflow.keras import losses
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
        iou = tf.divide(tf.cast(intersection,tf.float32),tf.cast(union,tf.float32))
        class_scores.append(iou)
    return tf.divide(tf.reduce_sum(class_scores),num_labels)

def generalized_dl(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, params["number_of_classes"]])
    smooth = 1e-10
    y_true = tf.one_hot(tf.cast(y_true,tf.int32), params["number_of_classes"], dtype=tf.float32)
    #weights = 1.0 / (tf.reduce_sum(y_true, axis=[0, 1, 2])**2)

    numerator = tf.reduce_sum(y_true * y_pred)
    #numerator = tf.reduce_sum(numerator)

    denominator = tf.reduce_sum(y_true + y_pred)
    #denominator = tf.reduce_sum(denominator)

    loss = 1.0 - 2.0*(numerator + smooth)/(denominator + smooth)
    return loss
