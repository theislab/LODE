import sys
import os
from pathlib import Path

PROJ_DIR = Path(os.getcwd()).parent
WORK_SPACE = '/home/olle/PycharmProjects/LODE/workspace'
OCT_DIR = "/storage/groups/ml01/datasets/raw/2018_LMUAugenklinik_niklas.koehler/Studies/Optical Coherence Tomography Scanner"
SEG_DIR = os.path.join(WORK_SPACE, "segmentation/segmented_volumes20201026")
DICOM_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/dicoms"
sys.path.append(str(PROJ_DIR))
