from __future__ import print_function
from python_generator import DataGenerator
import pandas as pd
import os


def get_generators(params):

    gen_params = {'dim': (params.img_shape, params.img_shape),
                  'batch_size': params.batch_size,
                  'n_channels': 3,
                  'shuffle': True,
                  'fundus_path': "./data/fundus",
                  'thickness_path': "./data/thickness_maps",
                  'brightness_factor': 0.5,
                  'contrast_factor': 0.7}

    print('Using real-time data augmentation.')
    train_file_names = "./full_export_file_names/diabetes_patients/train_diab_filtered.csv"
    validation_file_names = "./full_export_file_names/diabetes_patients/validation_diab_filtered.csv"
    test_file_names = "./full_export_file_names/diabetes_patients/test_diab_filtered.csv"

    train_ids = pd.read_csv(train_file_names)["0"]
    validation_ids = pd.read_csv(validation_file_names)["0"]
    test_ids = pd.read_csv(test_file_names)["0"]

    partition = {'train': train_ids.values.tolist(),
                 'validation': validation_ids.values.tolist(),
                 'test': test_ids.values.tolist()}

    # Generators
    training_generator = DataGenerator(partition['train'], is_training=True, **gen_params)
    validation_generator = DataGenerator(partition['validation'], is_training=False, **gen_params)
    test_generator = DataGenerator(partition['test'], is_training=False, **gen_params)

    return(training_generator,validation_generator,test_generator)

def get_data_statistics(params):
    print('Using real-time data augmentation.')
    train_file_names = "./full_export_file_names/diabetes_patients/train_diab_filtered.csv"
    validation_file_names = "./full_export_file_names/diabetes_patients/validation_diab_filtered.csv"
    test_file_names = "./full_export_file_names/diabetes_patients/test_diab_filtered.csv"

    train_ids = pd.read_csv(train_file_names)["0"]
    validation_ids = pd.read_csv(validation_file_names)["0"]
    test_ids = pd.read_csv(test_file_names)["0"]

    num_train_examples = train_ids.shape[0]
    num_val_examples = validation_ids.shape[0]
    num_test_examples = test_ids.shape[0]

    print("Number of training examples: {}".format(num_train_examples))
    print("Number of validation examples: {}".format(num_val_examples))
    print("Number of test examples: {}".format(num_test_examples))

    return(num_train_examples, num_val_examples, num_test_examples)
