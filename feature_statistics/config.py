import sys
import os
from pathlib import Path

PROJ_DIR = Path(os.getcwd()).parent
WORK_SPACE = '/storage/groups/ml01/workspace/olle.holmberg/LODE/'
OCT_DIR = "/storage/groups/ml01/datasets/raw/2018_LMUAugenklinik_niklas.koehler/Studies/Optical Coherence Tomography Scanner"
SEG_DIR = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/lmu_data/segmented_volumes20201026")
DATA_DIR = "/storage/groups/ml01/datasets/projects/20181610_eyeclinic_niklas.koehler/joint_export"
DICOM_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/dicoms"
sys.path.append(str(PROJ_DIR))
