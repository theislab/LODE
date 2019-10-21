from __future__ import print_function
from keras.optimizers import SGD
import model as mt
import multiprocessing
from keras.callbacks import LearningRateScheduler
from keras.utils.data_utils import get_file
from train_eval_ops import *
from keras.models import Model
import pandas as pd
import tensorflow as tf
import os
import input as i
from utils import Params
from utils import Logging

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3

    if epoch > 85:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def load_models(model, weights):
    pre_init = False
    if weights == "imagenet":
        WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                               'releases/download/v0.2/'
                               'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            md5_hash='a7b3fe01876f51b976af0dea6bc144eb')

        # get model weights
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

        print("loaded imagenet model weights")
        pre_init = True

    if weights == "thickness_map":
        weights_path = "./thickness_model_weights/weights.hdf5"
        # get model weights
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

        print("load thickness map weights")
        pre_init = True

    if not pre_init:
        print("init random weights")

def main(logging, params):

    #create logging directory
    logging.create_model_directory(logging.log_dir)

    #create save directory if does not exist
    if not os.path.exists(logging.model_directory):
        os.makedirs(logging.model_directory)


    model = mt.ressttNet(input_shape=params.img_shape, n_filters=params.num_filters)
    model.summary()

    #load model
    '''train and save model'''
    save_model_path = "./thickness_model_weights/weights.hdf5"#os.path.join(logging.model_directory,"weights.hdf5")

    if params.continue_training == 1:
        #load model
        model.load_weights(save_model_path)

    model.compile(loss=custom_mae,
                  optimizer=SGD(lr=0.001,momentum=0.99),
                  metrics=[custom_mae, percentual_deviance])

    '''callbacks'''
    lr_scheduler = LearningRateScheduler(lr_schedule)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_custom_mae',
                                                    save_best_only=True, verbose=1, save_weights_only=True)

    tb = keras.callbacks.TensorBoard(log_dir=logging.tensorboard_directory,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=True,
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None)

    #get input genetarators and statististics
    training_generator, validation_generator, test_generator = i.get_generators(params)
    num_train_examples, num_val_examples, num_test_examples = i.get_data_statistics(params)

    # saving model config file to model output dir
    logging.save_dict_to_json(logging.model_directory + "/config.json")

    history = model.fit_generator(
        generator=training_generator,
        steps_per_epoch=int(num_train_examples / (params.batch_size)),
        epochs=30,
        validation_data=validation_generator,
        validation_steps=int(num_val_examples / (params.batch_size)),
        workers=multiprocessing.cpu_count() -4 ,
        use_multiprocessing=True,
        callbacks=[checkpoint, lr_scheduler,tb])

    pd.DataFrame(history.history).to_csv(logging.model_directory+"/loss_files.csv")


#load utils classes
params = Params("params.json")
logging = Logging("./logs",params)


main(logging=logging,params=params)