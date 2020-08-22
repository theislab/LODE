import tensorflow as tf


def dice_loss(y_true, y_pred, num_labels):

    probabilities = tf.keras.backend.reshape(y_pred, [-1, num_labels])
    y_true_flat = tf.keras.backend.reshape(y_true, [-1])

    onehots_true = tf.one_hot(tf.cast(y_true_flat, tf.int32), num_labels)

    numerator = tf.reduce_sum(onehots_true * probabilities, axis = -1)
    denominator = tf.reduce_sum(onehots_true + probabilities, axis = -1)

    loss = 1.0 - 2.0 * (numerator / denominator)
    return tf.keras.backend.mean(loss)


def gen_dice(y_true, y_pred, eps=1e-6):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""

    # [b, h, w, classes]
    pred_tensor = y_pred
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    # count how many of each class are present in
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    # counts = tf.reduce_sum(y_true, axis=1)
    # weights = 1. / (counts ** 2)
    # weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis = 1)
    summed = tf.reduce_sum(y_true + y_pred, axis = 1)

    # [b]
    numerators = tf.reduce_sum(multed, axis = -1)
    denom = tf.reduce_sum(summed, axis = -1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)