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
VOL_SAVE_PATH = os.path.join(WORK_SPACE, "segmentation/segmented_volumes20200822")
EMBEDD_SAVE_PATH = os.path.join(WORK_SPACE, "segmentation/embeddings20200822")
TRAIN_DATA_PATH = os.path.join(WORK_SPACE, "segmentation/train_data/hq_examples_fibrosis")

