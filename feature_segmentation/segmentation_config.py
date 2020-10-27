import os

"""
PROJ_DIR: project dir of project
WORK_SPACE: path to directory storing
- segmentation/path_files
VOL_SAVE_PATH: path to where to save segmented volumes
EMBEDD_SAVE_PATH: path where to save embeddings
"""

PROJ_DIR = '/home/icb/olle.holmberg/projects/LODE/feature_segmentation'
WORK_SPACE = '/storage/groups/ml01/workspace/olle.holmberg/LODE/feature_segmentation'
OCT_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/test_examples"


FEATURE_SAVE_PATH = os.path.join(WORK_SPACE, "segmentation/feature_tables")
VOL_SAVE_PATH = os.path.join(WORK_SPACE, "segmentation/segmented_volumes20201026")
EMBEDD_SAVE_PATH = os.path.join(WORK_SPACE, "segmentation/embeddings20201026")
TRAIN_DATA_PATH = os.path.join(WORK_SPACE, "segmentation/data/train_data/hq_examples")
DATA_SPLIT_PATH = os.path.join(WORK_SPACE, "segmentation/data/train_data/data_split")

