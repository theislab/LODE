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


def get_loss(params):
    """
    Parameters
    ----------
    params :

    Returns
    -------

    """

    if params.loss == "categorical_crossentropy":
        # Instantiate a loss function.
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = False)

    if params.loss == "dice_loss":
        loss_fn = DiceLoss(num_classes = params.num_classes)
    return loss_fn
