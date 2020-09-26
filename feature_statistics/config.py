import sys
import os
from pathlib import Path

PROJ_DIR = Path(os.getcwd()).parent
WORK_SPACE = '/home/olle/PycharmProjects/LODE/workspace'
OCT_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/dicoms"
SEG_DIR = os.path.join(WORK_SPACE, "feature_segmentation/segmentation/segmented_volumes20200728")
DICOM_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/dicoms"
sys.path.append(str(PROJ_DIR))
