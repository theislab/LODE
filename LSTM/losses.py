import tensorflow as tf
import numpy as np

def mae_last(y_true, y_pred):
    return tf.math.abs(y_pred[:,-1] - y_true[:,-1])

def mae_last_numpy(y_true, y_pred, index=-1):
    return np.mean(np.abs(y_pred[:,index] - y_true[:,index]))
