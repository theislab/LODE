import matplotlib as mpl
from tensorflow.keras import backend as K
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)
from train_eval_ops import *
import numpy as np
import tensorflow.keras
from params import *
import tensorflow as tf
from tensorflow.python.tensorflow.keras import losses



#tensorflow.keras loss functions

def percentual_deviance(y_true, y_pred):
    return(K.mean(K.abs(y_true[:,:,:,0] - y_pred[:,:,:,0])) / K.mean(y_true))

def custom_mse(y_true, y_pred):
    return(K.mean((y_true[:,:,:,0] - y_pred[:, :, :, 0])**2))

def custom_mae(y_true, y_pred):
    return(K.mean(K.abs((y_true[:,:,:,0] - y_pred[:, :, :, 0]))))

def uncertainty_loss_mse(y_true, y_pred):
    return K.mean((y_true[:,:,:,0]-y_pred[:,:,:,0])**2 * K.exp(-y_pred[:,:,:,1]) + y_pred[:,:,:,1])

def uncertainty_loss_mae(y_true, y_pred):
    return K.mean(K.abs((y_true[:,:,:,0]-y_pred[:,:,:,0])) * K.exp(-y_pred[:,:,:,1]) + y_pred[:,:,:,1])

def weighted_loss(y_true,y_pred):
    return(1.0 * custom_mse(y_true,y_pred) + 0.2 * uncertainty_loss(y_true, y_pred))

def weighted_loss_mae(y_true,y_pred):
    return(1.0 * custom_mae(y_true,y_pred) + 0.2 * uncertainty_loss_mae(y_true, y_pred))

def scale_invariant_term(y_true, y_pred):
    n = K.shape(K.flatten(y_true))**2

    denominator = tf.cast(tf.math.divide(1,n), tf.float32)
    nominator = K.sum(y_true[:,:,:,0] - y_pred[:, :, :, 0])
    nominator_squared = nominator **2

    return nominator_squared * denominator

def scale_invariant_loss(y_true, y_pred):
    l2 = custom_mse(K.log(y_true), K.log(y_pred))
    scale_inv = scale_invariant_term(K.log(y_true), K.log(y_pred))
    return l2 + 1.0*scale_inv

########################################################################################################################

# NUMPY IMPLEMENTATIONS
def percentual_deviance_np(y_true, y_pred):
    return(np.mean(np.abs(y_true[:,:,:,0] - y_pred[:,:,:,0])) / np.mean(y_true))

def fovea_percentual_deviance_np(y_true, y_pred):
    return(np.mean(np.abs(y_true - y_pred)) / np.mean(y_true))

def perc_deviance_map_np(y_true, y_pred):
    deviance_map = np.divide(np.abs(y_true[:, :, :, 0] - y_pred[:, :, :, 0]), y_true[:,:,:,0])
    return(deviance_map)

def custom_mse_np(y_true, y_pred):
    return(np.mean((y_true[:,:,:,0] - y_pred[:, :, :, 0])**2))

def custom_mae_np(y_true, y_pred):
    return(np.mean(np.abs((y_true[:,:,:,0] - y_pred[:, :, :, 0]))))

def uncertainty_loss_mae_np(y_true, y_pred):
    return np.mean(np.abs((y_true[:,:,:,0]-y_pred[:,:,:,0])) * np.exp(-y_pred[:,:,:,1]) + y_pred[:,:,:,1])




