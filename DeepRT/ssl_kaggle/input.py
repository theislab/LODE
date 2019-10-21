# from __future__ import print_function
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

from keras.preprocessing.image import ImageDataGenerator
import glob as glob
from utils import Params
import os
import model as m

params = Params("params.json")


def get_data_statistics(data_path):
    # number of training images
    num_training_images = len(glob.glob(os.path.join(data_path + "/train", "*", "*.jpeg")))
    num_validation_images = len(glob.glob(os.path.join(data_path + "/validation", "*", "*.jpeg")))
    num_test_images = len(glob.glob(os.path.join(data_path + "/test", "*", "*.jpeg")))

    print("number of train, validation and test images are:",
          num_training_images,
          num_validation_images,
          num_test_images)

    return num_training_images, num_validation_images, num_test_images


def create_generators(data_path):
    print('Using real-time data augmentation.')

    train_datagen = ImageDataGenerator(
        rescale=(1. / 255),
        rotation_range=45,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    train_generator = train_datagen.flow_from_directory(
        directory=data_path + "/train",
        target_size=(params.img_shape, params.img_shape),
        color_mode='rgb',
        batch_size=params.batch_size,
        shuffle=True,
        seed=1,
        class_mode="categorical")

    print('train_generator created')

    valid_generator = test_datagen.flow_from_directory(
        directory=data_path + "/validation",
        target_size=(params.img_shape, params.img_shape),
        color_mode='rgb',
        batch_size=1,
        shuffle=False,
        class_mode="categorical")

    print('validation_generator created')

    test_generator = test_datagen.flow_from_directory(
        directory=data_path + "/test",
        target_size=(params.img_shape, params.img_shape),
        color_mode='rgb',
        batch_size=1,
        shuffle=False,
        class_mode="categorical")

    print('test_generator created')

    return (train_generator, valid_generator, test_generator)


def create_test_generator(data_path):
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    test_generator = test_datagen.flow_from_directory(
        directory=data_path + "/test",
        target_size=(params.img_shape, params.img_shape),
        color_mode='rgb',
        batch_size=1,
        shuffle=False,
        class_mode="categorical")

    print('test_generator created')

    return test_generator


def get_test_statistics(data_path):
    # number of training images
    num_test_images = len(glob.glob(os.path.join(data_path + "/test", "*", "*.jpeg")))

    print("number of test images are:", num_test_images)

    return (num_test_images)
