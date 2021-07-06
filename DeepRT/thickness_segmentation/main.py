import os
import model as mt
from train_eval_ops import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
from tensorflow.keras.layers import Input
from params import params, gen_params
import pandas as pd
from python_generator import DataGenerator
from DeepRT.thickness_segmentation.params import gen_params
from DeepRT.thickness_segmentation.params import params
from DeepRT.thickness_segmentation.train_eval_ops import dice_loss, dice_coeff


def return_data_fromcsv_files(params):
    train_file = os.path.join( params["data_dir"], "file_names_complete", "train_new_old_mapping.csv" )
    val_file = os.path.join( params["data_dir"], "file_names_complete", "validation_new_old_mapping.csv" )

    train_files = pd.read_csv( train_file, index_col = False )["new_id"]
    val_files = pd.read_csv( val_file )["new_id"]

    partition = {"train": train_files.tolist(), "validation": val_files.tolist()}
    return partition


def return_model(params):
    '''get model'''
    input_img = Input(params["img_shape"], name = 'img')
    model = mt.get_unet( input_img, n_filters = 12, dropout = params["drop_out"], batchnorm = True,
                         training = True )

    return model


if not os.path.exists(params["save_path"]):
    os.makedirs( params["save_path"] )

gen_params['dim'] = (params["img_shape"][0], params["img_shape"][1])
# bayesian parameters assigment

# set image shape param
dim = gen_params['dim'][0]
params["img_shape"] = (256, 256, 3)
# set params for optimization
params["drop_out"] = 0.5
# learning rate
params["learning_rate"] = 0.001

opt = adam( lr = params["learning_rate"], beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0,
            amsgrad = False )

# get file names for generator
partition = return_data_fromcsv_files( params )

num_training_examples = len( partition['train'] )
num_val_examples = len( partition['validation'] )

# Generators
training_generator = DataGenerator( partition['train'], is_training = True, **gen_params )
validation_generator = DataGenerator( partition['validation'], is_training = False, **gen_params )

# get model
model = return_model(params)
'''Compile model'''
model.compile(optimizer = opt, loss =dice_loss, metrics=[dice_coeff])
# model.summary()

'''train and save model'''
save_model_path = os.path.join(params["save_path"],
                               "weights.hdf5" )

cp = tf.tensorflow.keras.callbacks.ModelCheckpoint( filepath = save_model_path, monitor = "val_loss",
                                         save_best_only = True, verbose = 1, save_weights_only = True )

if params["continuing_training"] == True:
    '''Load models trained weights'''
    model.load_weights( save_model_path, by_name = True, skip_mismatch = True )

# Train model on dataset
history = model.fit_generator(generator = training_generator,
                               validation_data = validation_generator,
                               use_multiprocessing = False,
                               steps_per_epoch = int( num_training_examples / (params["batch_size"]) ),
                               validation_steps = int( num_val_examples / (params["batch_size"]) ),
                               epochs = params["epochs"],
                               verbose = 1,
                               workers = 5,
                               callbacks = [cp])

