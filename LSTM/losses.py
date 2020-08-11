import tensorflow as tf
import numpy as np

# -- tf functions --
## for regression target
def mae_last(y_true, y_pred):
    """mean average error of the final prediction in a sequence"""
    return tf.math.abs(y_pred[:,-1] - y_true[:,-1])

def thresholded_mae_last(threshold=0.15):
    """
    mean average error of the final prediction in a sequence, with value 0 for all absolute errors < threshold.
    This deals with label uncertainty in a certain range
    """
    def thresholded_mae_last_fn(y_true, y_pred):
        err = tf.math.abs(y_true[:,-1] - y_pred[:,-1])
        err = tf.math.reduce_max([tf.zeros_like(err), err-threshold], axis=0)
        return err
    return thresholded_mae_last_fn

def mse_last(y_true, y_pred):
    """mean squared error of the final prediction in a sequence"""
    return tf.math.abs(y_pred[:,-1] - y_true[:,-1])**2

def thresholded_mse_last(threshold=0.15):
    """
    mean squared error of the final prediction in a sequence, with value 0 for all absolute errors < threshold.
    This deals with label uncertainty in a certain range
    """
    def thresholded_mse_last_fn(y_true, y_pred):
        err = tf.math.abs(y_true[:,-1] - y_pred[:,-1])
        err = tf.math.reduce_max([tf.zeros_like(err), err-threshold], axis=0)
        mse = err**2
        return mse
    return thresholded_mse_last_fn

def binary_regression_acc_last(threshold=0.15):
    """accuracy of binarized regression values. All absolute errors < threshold are correctly classified, all others incorrectly"""
    def binary_regression_acc_last_fn(y_true, y_pred):
        err = tf.math.abs(y_true[:,-1] - y_pred[:,-1])
        return tf.reduce_mean(tf.cast(err <= threshold, tf.float32))
    return binary_regression_acc_last_fn

## for classification target
def categorical_crossentropy_last(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true[:,-1], y_pred[:,-1])

class PrecisionLast(tf.keras.metrics.Precision):
    # overwrite update_state to fix calculation to last element in sequence only
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true=y_true[:,-1], y_pred=y_pred[:,-1], sample_weight=sample_weight)

class RecallLast(tf.keras.metrics.Recall):
    # overwrite update_state to fix calculation to last element in sequence only
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true=y_true[:,-1], y_pred=y_pred[:,-1], sample_weight=sample_weight)
        
def categorical_accuracy_last(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true[:,-1], y_pred[:,-1])

# -- numpy functions --
## for regression target
def mae_last_numpy(y_true, y_pred, index=-1):
    return np.mean(np.abs(y_pred[:,index] - y_true[:,index]))

def thresholded_mae_last_numpy(y_true, y_pred, threshold=0.15):
    return np.mean(thresholded_mae_last(threshold)(y_true, y_pred).numpy())

def binary_regression_acc_last_numpy(y_true, y_pred, threshold=0.15):
    return binary_regression_acc_last(threshold)(y_pred, y_true).numpy()



    