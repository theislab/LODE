from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import adam
import os
from tensorflow.keras.models import Model
import resnet as re
import multiprocessing
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau
from python_generator_test import DataGenerator_kaggle
from train_eval_ops import *
from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras.callbacks import LearningRateScheduler
import glob
import math

# get model
params = {}
params["batch_size"] = 100
params["img_shape"] = (256, 256, 3)
params["epochs"] = 10
params["learning_rate"] = 0.001
#set to true if weights should not be randomly inititalized
params["continuing_training"] = False
params["save_path"] = "./output"
#set what weights to init network with
params["weights"] = "thickness_map"
params["generator_type"] = "simple"
#set parameter for under or oversampling
params["sampling"] = "oversampling"
#set path to proportion of data set to be trained on
params["file_prop"] = "hundred"
params["ids_path"] = "./file_splits/size_splits"
params["data_dir"] = "/home/olle/PycharmProjects/ssl_kaggle_drd/data/tensorflow.keras_generator_format"

def load_weights(model):
    pre_init = False
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
        pre_init = True

    if params["weights"] == "thickness_map":
        weights_path = "./thickness_model_weights/weights.hdf5"
        # get model weights
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

        print("load thickness map weights")
        pre_init = True

    if params["weights"] == "model_file":
        model.load_weights(os.path.join(params["save_path"],"weights.hdf5"),by_name=True, skip_mismatch=True)
        print("loading balanced model weights")
        pre_init = True

    if not pre_init:
        print("init random weights")

def get_generators():
    if params["generator_type"] == "simple":
        training_generator = DataGenerator_simple(partition['train'], is_training=True, **gen_params)
        validation_generator = DataGenerator_simple(partition['validation'], is_training=False, **gen_params)
        print("using simple generator pipeline")

    if params["generator_type"] == "kaggle":
        # Generators
        training_generator = DataGenerator_kaggle(partition['train'], is_training=True, **gen_params)
        validation_generator = DataGenerator_kaggle(partition['validation'], is_training=False, **gen_params)
        print("using kaggle generator pipeline")

    return training_generator, validation_generator



'''train and save model'''
# get model
res_output, img_input = re.ResNet50(params["img_shape"], 5)
model = Model(inputs=img_input, outputs=[res_output])
model.summary()

'''Compile model'''
sgd = tensorflow.keras.optimizers.SGD(lr=params["learning_rate"], momentum=0.99, decay=0.0, nesterov=False)
Adam = adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

def main(model):

    # learning rate schedule
    def step_decay(epoch):
        initial_lrate = 0.1
        drop = 0.1
        epochs_drop = 35.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    #create save directory if does not exist
    if not os.path.exists(params["save_path"]):
        os.makedirs(params["save_path"])

    #number of training images
    num_training_images = len(glob.glob(os.path.join(params["train_dir"],"*","*.jpeg")))
    num_validation_images = len(glob.glob(os.path.join(params["validation_dir"],"*","*.jpeg")))

    train_datagen=ImageDataGenerator(
        rescale=1./255-0.5,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range = 0.3
    )

    validation_datagen=ImageDataGenerator(
        rescale=1./255-0.5,
        )


    train_generator=train_datagen.flow_from_directory(
        directory=params["train_dir"],
        target_size=(params["img_shape"][0], params["img_shape"][1]),
        color_mode='rgb',
        batch_size=params["batch_size"],
        shuffle=True,
        class_mode="categorical")

    print('break')

    valid_generator=validation_datagen.flow_from_directory(
        directory=params["validation_dir"],
        target_size=(params["img_shape"][0], params["img_shape"][1]),
        color_mode='rgb',
        batch_size=params["batch_size"],
        class_mode="categorical")

    '''callbacks'''
    save_model_path = os.path.join(params["save_path"], "weights.hdf5")
    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_acc',
                                                    save_best_only=True, verbose=1, save_weights_only=True)
    learning_rate_reduction = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1)
    es = tf.tensorflow.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=75)
    tb = tensorflow.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                write_graph=True, write_images=True)

    learning_rate_ = LearningRateScheduler(step_decay, verbose=2)

    #load weights
    load_weights(model)

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=int(num_training_images / (params["batch_size"])),
        use_multiprocessing=False,
        epochs=150,
        validation_data = valid_generator,
        validation_steps = int(num_validation_images/(params["batch_size"])),
        workers=multiprocessing.cpu_count() - 1,
        callbacks=[checkpoint,es,learning_rate_])

    pd.DataFrame(history.history).to_csv(params["loss_file_name"])

#config iterations
inits = ["random"]
data_distribution = ["unbalanced"]

#generate all configs
for init in inits:
    params["weights"] = init
    # create folder name for saving output
    params["save_path"] = os.path.join("/home/olle/PycharmProjects/ssl_kaggle_drd/kaggle_test",
                                       "2_"  + params["weights"] + "_hundred_64")

    for data_dist in data_distribution:
        params["train_dir"] = os.path.join(params["data_dir"],data_dist,"512/train/")
        params["validation_dir"] = os.path.join(params["data_dir"],data_dist,"512/validation/")
        params["loss_file_name"] = "output_" + params["weights"] + "_" + data_dist + "_hundred_64.csv"

        #when
        if data_dist == "unbalanced":
            params["weights"] = "test"

        #exeecute model iteration
        main(model)
