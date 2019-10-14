"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
from PIL import ImageEnhance
import gc
import pandas as pd
from PIL import Image
from loading_numpy_functions import *
import tensorflow as tf

save_directory = './tf_records/'
label_directory = './train_labels/'
data_directory = './train_images/'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#@profile
def convert_to(train_images,train_labels,train_name,train_num_exmaples):
    """Converts a dataset to tfrecords."""
    images = train_images
    labels = train_labels
    print("train im shape is {}".format(images.shape))
    num_examples = train_num_exmaples
    print('number of examples this file is {}'.format(num_examples))
    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(save_directory, "train" + '.bin')
    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            if index%100 == 0:
                gc.collect()
            #print(images.shape)
            image_raw = images[index].tostring()
            labels_raw = labels[index].tostring()
            train_name_raw = train_name[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(rows),
                        'width': _int64_feature(cols),
                        'depth': _int64_feature(depth),
                        'label': _bytes_feature(labels_raw),
                        'image_raw': _bytes_feature(image_raw),
                        'image_name': _bytes_feature(train_name_raw),

                    }))
            writer.write(example.SerializeToString())
#@profile
def main():
    data = [[], [], []]
    batch_size = 1
    filenames = os.listdir(data_directory)
    num_examples = len(filenames)
    print("number of exmaples is {}".format(num_examples))
    for i in range(0, num_examples):
        width = 400
        height = 160



        x_batch, y_batch, im_displayed,new_shape,orig_shape = get_clinic_train_data(data_directory, label_directory,
                                                                  width, height, batch_size, i)
        print(im_displayed)
        data[0].append(im_displayed)
        data[1].append(x_batch)
        data[2].append(y_batch)

    train_images = np.vstack(data[1])
    train_labels = np.vstack(data[2]).astype(np.float32)
    train_name = np.vstack(data[0])
    train_num_exmaples = train_labels.shape[0]
    print("convert new batch to tfrecords")
    # Convert to Examples and write the result to TFRecords.
    convert_to(train_images,train_labels,train_name,train_num_exmaples)
    del data
    del train_images
    gc.collect()

main()