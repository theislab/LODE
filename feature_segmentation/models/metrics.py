# yapf: disable
import tensorflow as tf
import keras
import tensorflow
from tensorflow.keras import backend as K


class DiceCoefficient(tensorflow.keras.metrics.Metric):

    def __init__(self, name='dice', num_classes=None, **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dc = self.add_weight(name='dc', initializer='zeros')
        self.iterations = 0
        self.num_classes = num_classes

    def dice_coef_cat(self, y_true, y_pred, num_classes, smooth=1e-7):
        '''
        Dice coefficient for 10 categories. Ignores background pixel label 0
        Pass to model as metric during compile statement
        '''
        y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes = num_classes)[..., 1:])
        y_pred_f = K.flatten(y_pred[..., 1:])
        intersect = K.sum(y_true_f * y_pred_f, axis = -1)
        denom = K.sum(y_true_f + y_pred_f, axis = -1)
        return K.mean((K.constant(2.) * intersect / (denom + smooth)))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.dc.assign_add(self.dice_coef_cat(y_true, y_pred, self.num_classes))
        self.iterations += 1

    def reset_states(self):
        self.dc = self.add_weight(name='dc', initializer='zeros')
        self.iterations = 0

    def result(self):
        return tf.divide(self.dc, self.iterations)


