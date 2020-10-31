import os
from pathlib import Path

"""
PROJ_DIR: project dir of project
WORK_SPACE: path to directory storing
- segmentation/path_files
VOL_SAVE_PATH: path to where to save segmented volumes
EMBEDD_SAVE_PATH: path where to save embeddings
"""

PROJ_DIR = Path(os.getcwd()).parent.as_posix()
WORK_SPACE = './workspace'
OCT_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/test_examples"


FEATURE_SAVE_PATH = os.path.join(WORK_SPACE, "/feature_tables")
VOL_SAVE_PATH = os.path.join(WORK_SPACE, "segmentation/segmented_volumes20201026")
EMBEDD_SAVE_PATH = os.path.join(WORK_SPACE, "segmentation/embeddings20201026")
TRAIN_DATA_PATH = os.path.join(WORK_SPACE, "segmentation/data/train_data/hq_examples")
DATA_SPLIT_PATH = os.path.join(WORK_SPACE, "segmentation/data/train_data/data_split")

