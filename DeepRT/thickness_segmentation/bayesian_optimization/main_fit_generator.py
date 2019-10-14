import os
import model_test as mt
from train_eval_ops import *
from keras.optimizers import *
import tensorflow as tf
from keras.layers import Input
from params import *
import pandas as pd
from python_generator import DataGenerator

def return_data_fromcsv_files(params):
    train_file = os.path.join(params["data_dir"],"file_names","train_ids.csv")
    val_file = os.path.join(params["data_dir"],"file_names","validation_ids.csv")

    train_files = pd.read_csv(train_file,index_col=False)["new_id"]
    val_files = pd.read_csv(val_file)["new_id"]

    partition = {"train": train_files.tolist()[0:1], "validation": val_files.tolist()}
    return partition


def return_model(params,model_iter,filters):
    '''get model'''
    input_img = Input(params["img_shape"], name='img')
    if model_iter == 1:
        model = mt.get_bunet(input_img, n_filters=4*filters, dropout=params["drop_out"], batchnorm=True, training=True)

    if model_iter == 2:
        model = mt.get_shallow_bunet(input_img, n_filters=4*filters, dropout=params["drop_out"], batchnorm=True, training=True)

    if model_iter == 3:
        model = mt.get_very_shallow_bunet(input_img, n_filters=4*filters, dropout=params["drop_out"], batchnorm=True, training=True)

    return model

def main(input_shape, verbose, dr, lr,shape,model_it,num_filters,bf,cf):
    '''
    :param input_shape: tuple
    :param verbose: int
    :param dr: float
    :param lr: float
    :param shape: tuple
    :param model_it: float
    :param num_filters: float
    :return: An evaluation of sets of parameters with best config printed at end using bayesina
    optimization.
    '''
    if not os.path.exists( params["save_path"]):
        os.makedirs( params["save_path"])

    gen_params['dim'] = (params["img_shape"][0], params["img_shape"][1])
    #bayesian parameters assigment

    # set image shape param
    dim = gen_params['dim'][0]
    # select one of the three models
    model_iter = max(int(model_it),1)
    # select one of the five filter levels
    filters = max(int(num_filters)*1,1)
    params["img_shape"] = (dim,dim,3)
    # set params for optimization
    params["drop_out"] = dr
    # learning rate
    params["learning_rate"] = lr
    # brightness rate
    gen_params["brightness_factor"] = bf
    # contrast rate
    gen_params["contrast_factor"] = cf

    opt = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


    #get file names for generator
    partition = return_data_fromcsv_files(params)

    num_training_examples = len(partition['train'])
    num_val_examples = len(partition['validation'])

    # Generators
    training_generator = DataGenerator(partition['train'], is_training=True,**gen_params)
    validation_generator = DataGenerator(partition['validation'], is_training=False,**gen_params)

    #get model
    model = return_model(params,model_iter,filters)
    '''Compile model'''
    model.compile(optimizer=opt, loss=dice_loss)
    #model.summary()

    '''train and save model'''
    save_model_path = os.path.join(params["save_path"],"weights.hdf5")
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor="val_loss",
                                            save_best_only=True, verbose=0,save_weights_only=True)

    if params["continuing_training"] == True:
        '''Load models trained weights'''
        model.load_weights(save_model_path,by_name=True, skip_mismatch=True)

    # Train model on dataset
    history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        steps_per_epoch = int(num_training_examples/(params["batch_size"]))*100,
                        validation_steps=int(num_val_examples/(params["batch_size"])),
                        epochs=params["epochs"],
                        verbose = 1,
                        workers=5,
                        callbacks=[cp])

    best_val_metric = max(history.history["val_loss"])

    print('Val iou: ', best_val_metric)
    print("Config: ", )
    return(best_val_metric)



from functools import partial
import numpy as np
verbose = 1
input_shape = (256,256,3)
main_with_partial = partial(main, input_shape, verbose)

from bayes_opt import BayesianOptimization


#configs to evaluate
optimizers_names = ["adam"]
models = ["bunet"]

# Bounded region of parameter space
pbounds = {'dr' : (0.1, 0.5), 'lr' : (0.00001, 0.01),
           "shape" : (0.9, 3.1), "model_it" : (0.9,3.1),
           "num_filters" : (0.9,5.1),
           "bf" : (0,1),
           "cf": (0,1)}

optimizer = BayesianOptimization(
    f=main_with_partial,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(init_points=5, n_iter=20)


for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("Best performing model was:")
print(optimizer.max)