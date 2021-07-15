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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import glob
import math

# get model
params = {}
params["batch_size"] = 32
params["img_shape"] = (256, 256, 3)
params["epochs"] = 10
params["learning_rate"] = 0.001
# set to true if weights should not be randomly inititalized
params["continuing_training"] = False
params["save_path"] = "./output_random_hundred"
# set what weights to init network with
params["weights"] = "thickness_map"
params["generator_type"] = "simple"
# set parameter for under or oversampling
params["sampling"] = "oversampling"
# set path to proportion of data set to be trained on
params["file_prop"] = "hundred"
params["ids_path"] = "./file_splits/size_splits"
params["data_dir"] = "/home/olle/PycharmProjects/ssl_kaggle_drd/data/tensorflow.keras_generator_format"

params["validation_dir"] = os.path.join(params["data_dir"],"unbalanced","512/validation/")

def load_weights(model):
    pre_init = False

    model.load_weights(os.path.join(params["save_path"], "weights.hdf5"), by_name=True, skip_mismatch=True)
    print("loading balanced model weights")
    pre_init = True

    if not pre_init:
        print("init random weights")


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
    # number of training images
    num_validation_images = len(glob.glob(os.path.join(params["validation_dir"], "*", "*.jpeg")))

    validation_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    valid_generator = validation_datagen.flow_from_directory(
        directory=params["validation_dir"],
        target_size=(params["img_shape"][0], params["img_shape"][1]),
        color_mode='rgb',
        batch_size=1,
        shuffle=True,
        class_mode="categorical")

    '''callbacks'''
    save_model_path = os.path.join(params["save_path"], "weights.hdf5")

    # load weights
    load_weights(model)

    prediction = model.predict_generator(
        generator=valid_generator,
        steps=int(num_validation_images),verbose=1)

    #extract prediction and labels in numpy format
    preds = np.argmax(prediction,1)
    lbls = valid_generator.labels.tolist()

    print("Accuracy score:", accuracy_score(lbls, preds))
    print("Recall score:", recall_score(lbls, preds, average='macro'))
    print("Precision score:", precision_score(lbls, preds, average='macro'))
    print("confusion matrix:", confusion_matrix(lbls, preds))



main(model)
