import os

"""
PROJ_DIR: project dir of project
WORK_SPACE: path to directory storing
- segmentation/path_files
VOL_SAVE_PATH: path to where to save segmented volumes
EMBEDD_SAVE_PATH: path where to save embeddings
"""


PROJ_DIR = '/home/olle/PycharmProjects/LODE/feature_segmentation'
WORK_SPACE = '/home/olle/PycharmProjects/LODE/workspace'
VOL_SAVE_PATH = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/segmented_volumes20200822")

OCT_DIR = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/dicoms")
EMBEDD_DIR = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/embeddings20200728")
TRAIN_DATA_PATH = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/data/train_data/hq_examples_fibrosis")

