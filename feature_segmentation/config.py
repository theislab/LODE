import os

"""
PROJ_DIR: project dir of project
WORK_SPACE: path to directory storing
- segmentation/path_files
VOL_SAVE_PATH: path to where to save segmented volumes
EMBEDD_SAVE_PATH: path where to save embeddings
"""


PROJ_DIR = '/home/icb/olle.holmberg/projects/LODE'
WORK_SPACE = '/storage/groups/ml01/workspace/olle.holmberg/LODE'
VOL_SAVE_PATH = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/segmented_volumes20200822")
EMBEDD_DIR = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/embeddings20200822")
TRAIN_DATA_PATH = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/data/train_data/hq_examples")
#TRAIN_DATA_PATH = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/data/train_data/first_examples")

OCT_DIR = "/storage/groups/ml01/datasets/raw/2018_LMUAugenklinik_niklas.koehler/Studies/Optical Coherence Tomography Scanner"

