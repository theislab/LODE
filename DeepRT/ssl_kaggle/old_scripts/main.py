from tensorflow.keras.optimizers import adam
import os
from tensorflow.keras.models import Model
import resnet as re
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau
from simple_generator import DataGenerator_simple
from python_generator import DataGenerator
from train_eval_ops import *
from tensorflow.keras.utils.data_utils import get_file
import sys

sys.dont_write_bytecode = True

def balancing(train_):
    if params["sampling"] == "oversampling":
        num_balancing_class = train_["image"][train_.level == 0].shape[0]
    if params["sampling"] == "undersampling":
        num_balancing_class = train_["image"][train_.level == 3].shape[0]

    train_im_level0 = np.random.choice(train_["image"][train_.level == 0],size=num_balancing_class,replace=True)
    train_im_level1 = np.random.choice(train_["image"][train_.level == 1],size=num_balancing_class,replace=True)
    train_im_level2 = np.random.choice(train_["image"][train_.level == 2],size=num_balancing_class,replace=True)
    train_im_level3 = np.random.choice(train_["image"][train_.level == 3],size=num_balancing_class,replace=True)
    train_im_level4 = np.random.choice(train_["image"][train_.level == 4],size=num_balancing_class,replace=True)

    files_ = pd.DataFrame(train_im_level0.tolist()+train_im_level1.tolist()+train_im_level2.tolist()\
    +train_im_level3.tolist()+train_im_level4.tolist())

    return files_

def class_distribution(train,validation):
    labels = pd.read_csv("/media/olle/Seagate/kaggle/trainLabels.csv")
    print("train class distribution is:",np.unique(pd.merge(train, labels, left_on=0,right_on='image',
                                                            how="inner")["level"],return_counts=True))

    print("validation class distribution is:",np.unique(pd.merge(validation, labels, left_on=0,right_on='image',
                                                            how="inner")["level"],return_counts=True))
    return

train = pd.read_csv(os.path.join(params["id_path"],"train.csv"))
validation = pd.read_csv(os.path.join(params["id_path"],"validation.csv"))

num_train_examples = train.shape[0]
num_val_examples = validation.shape[0]

print("Number of training examples: {}".format(num_train_examples))
print("Number of validation examples: {}".format(num_val_examples))

print("balancing the records")
train_ids = balancing(train)
validation_ids = balancing(validation)

num_train_examples = train_ids.shape[0]
num_val_examples = validation_ids.shape[0]

print("Number of training examples, after balancing: {}".format(num_train_examples))
print("Number of validation examples, after balancing: {}".format(num_val_examples))

partition = {'train': train_ids[0].tolist(), 'validation': validation_ids[0].tolist()}

class_distribution(train_ids,validation_ids)

def load_weights(model):
    if params["weights"] == "imagenet":
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

    if params["weights"] == "thickness_map":
        weights_path = "./thickness_model_weights/weights.hdf5"
        # get model weights
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

        print("load thickness map weights")

def get_compile_model():
    # get model
    res_output, img_input = re.ResNet50(params["img_shape"], 5)
    model = Model(inputs=img_input, outputs=[res_output])
    model.summary()

    '''Compile model'''
    Adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=Adam, loss="categorical_crossentropy", metrics=["accuracy", fmeasure])
    return model

def get_generators():
    if params["generator_type"] == "simple":
        training_generator = DataGenerator_simple(partition['train'], is_training=True, **gen_params)
        validation_generator = DataGenerator_simple(partition['validation'], is_training=False, **gen_params)
        print("using kaggle generator pipeline")

    if params["generator_type"] == "kaggle":
        # Generators
        training_generator = DataGenerator(partition['train'], is_training=True, **gen_params)
        validation_generator = DataGenerator(partition['validation'], is_training=False, **gen_params)
        print("using simple generator pipeline")

    return training_generator, validation_generator

def main():
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
    #create save directory if does not exist
    if not os.path.exists(params["save_path"]):
        os.makedirs(params["save_path"])

    #get generators
    training_generator, validation_generator = get_generators()

    #get model
    model = get_compile_model()

    if params["continuing_training"] == True:
        load_weights(model)

    '''train and save model'''
    save_model_path = os.path.join(params["save_path"], "weights.hdf5")
    cp = tf.tensorflow.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_acc',
                                            save_best_only=True, verbose=1, save_weights_only=True)

    learning_rate_reduction = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.000001, verbose=1)

    es = tf.tensorflow.keras.callbacks.EarlyStopping(monitor="val_acc",mode="max",patience=10)

    # Train model on dataset
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=False,
                                  steps_per_epoch=int(num_train_examples / (params["batch_size"])),
                                  validation_steps=int(num_val_examples / (params["batch_size"])),
                                  epochs=params["epochs"],
                                  verbose=1,
                                  workers=4,
                                  callbacks=[learning_rate_reduction,cp,es])


    best_val_metric = max(history.history["val_acc"])
    best_val_loss = history.history["val_loss"][np.argmax(history.history["val_acc"])]

    print('Val loss: ', best_val_loss)
    print('Val f measure: ', best_val_metric)
    print("Config: ", )
    return

main()