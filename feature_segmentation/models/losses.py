import tensorflow as tf
import keras
from keras import backend as K


class DiceLoss(keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """

    def __init__(self, num_classes, smooth=1e-7, name='dice_loss'):
        super().__init__(name = name)
        self.num_classes = num_classes
        self.smooth = smooth
        self.focal = self.add_weight(name='focal', initializer='zeros')


    def dice_coef_cat(self, y_true, y_pred):
        '''
        Dice coefficient for 10 categories. Ignores background pixel label 0
        Pass to model as metric during compile statement
        '''
        y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes = self.num_classes)[..., 1:])
        y_pred_f = K.flatten(y_pred[..., 1:])
        intersect = K.sum(y_true_f * y_pred_f, axis = -1)
        denom = K.sum(y_true_f + y_pred_f, axis = -1)
        return K.mean((2. * intersect / (denom + self.smooth)))

    def dice_coef_cat_loss(self, y_true, y_pred):
        '''
        Dice loss to minimize. Pass to model as loss during compile statement
        '''
        return 1 - self.dice_coef_cat(y_true, y_pred)

    def call(self, y_true, y_pred):
        return self.dice_coef_cat_loss(y_true, y_pred)


class FocalLoss(keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """

    def __init__(self, num_classes, name='focal_loss'):
        super().__init__(name = name)
        self.num_classes = num_classes
        self.alpha = 0.25
        self.gamma = 2
        self.current_value = tf.constant(0)

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        targets_f = K.one_hot(K.cast(targets, 'int32'), num_classes = self.num_classes + 1)[..., 1:]

        weight_a = alpha * (1 - y_pred) ** gamma * targets_f
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets_f)

        return (tf.math.log1p(keras.backend.exp(-keras.backend.abs(logits))) + tf.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(),
                                  1 - keras.backend.epsilon())
        logits = keras.backend.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=self.alpha, gamma=self.gamma, y_pred=y_pred)

        self.current_value = tf.reduce_mean(loss)
        return tf.reduce_mean(loss)

    def call(self, y_true, y_pred):
        return self.focal_loss(y_true, y_pred)


def get_loss(params):
    """
    Parameters
    ----------
    params :

    Returns
    -------

    """

    if params.loss == "categorical_crossentropy":
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = False)

    if params.loss == "dice_loss":
        loss_fn = DiceLoss(num_classes = params.num_classes)

    if params.loss == "focal_loss":
        loss_fn = FocalLoss(num_classes = params.num_classes)

    return loss_fn
