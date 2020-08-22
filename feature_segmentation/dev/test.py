from keras.layers import Input
import pandas as pd
from keras.optimizers import SGD
from utils import Params
from utils import Logging
from utils import Evaluation
import model as m
import os
from PIL import Image
import numpy as np
import keras as K
import tensorflow as tf

def iou(y_true, y_pred):
    num_labels = tf.compat.v1.keras.backend.int_shape(y_pred)[-1]
    y_flat = tf.compat.v1.keras.backend.argmax(tf.reshape(y_pred, [-1, num_labels]), axis=1)
    y_true_flat = tf.reshape(y_true, [-1])
    predictions = tf.one_hot(tf.cast(y_flat, tf.int32), num_labels)
    labels = tf.one_hot(tf.cast(y_true_flat, tf.int32), num_labels)
    class_scores = []
    for i in range(num_labels):
        intersection = tf.reduce_sum(labels[:, i] * predictions[:, i])
        union = tf.count_nonzero(labels[:, i] + predictions[:, i])
        iou = tf.divide(tf.cast(intersection, tf.float32), tf.cast(union, tf.float32) + 1.0)
        class_scores.append(iou)
    return tf.divide(tf.reduce_sum(class_scores), num_labels)

import matplotlib.pyplot as plt
model_dir = "./model"
params = Params("params.json")
logging = Logging("./model",params)

train_file_names = "./filenames/train_records.csv"
validation_file_names = "./filenames/validation_records.csv"
test_file_names = "./filenames/test_records.csv"

train_ids = pd.read_csv(train_file_names)["0"]
validation_ids = pd.read_csv(validation_file_names)["0"]
test_ids = pd.read_csv(test_file_names)["0"]

num_train_examples = train_ids.shape[0]
num_val_examples = validation_ids.shape[0]
num_test_examples = test_ids.shape[0]

image_path = "/home/olle/PycharmProjects/clinical_feature_segmentation/data/images"
label_path = "/home/olle/PycharmProjects/clinical_feature_segmentation/data/labels"

#
filename = test_ids[0]
'''get model'''
input_img = Input((params.img_shape,params.img_shape,1),
                  name='img')

model = m.unet(input_img,
               n_filters=16,
               dropout=params.drop_out,
               batchnorm=True,
               training=False)

'''Compile model'''
model.compile(optimizer=SGD(lr=params.learning_rate,
              momentum=0.99),
              loss=K.losses.sparse_categorical_crossentropy,
              metrics=[iou])

'''train and save model'''
save_model_path = os.path.join(model_dir, "weights.hdf5")

'''Load models trained weights'''
model.load_weights(save_model_path,by_name=True,skip_mismatch=True)

evaluation = Evaluation(model_dir,params,filename,model)
evaluation.plot_record()
print(evaluation.jaccard)

