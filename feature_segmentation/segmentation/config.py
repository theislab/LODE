import os

"""
PROJ_DIR: project dir of project
WORK_SPACE: path to directory storing
- segmentation/path_files
VOL_SAVE_PATH: path to where to save segmented volumes
EMBEDD_SAVE_PATH: path where to save embeddings
"""

PROJ_DIR = '/home/olle/PycharmProjects/LODE/feature_segmentation'
WORK_SPACE = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation"
VOL_SAVE_PATH = os.path.join(WORK_SPACE, "segmentation/segmented_volumes20200728")
EMBEDD_SAVE_PATH = os.path.join(WORK_SPACE, "segmentation/embeddings20200728")

