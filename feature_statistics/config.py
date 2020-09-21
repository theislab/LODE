import sys
import os
from pathlib import Path

PROJ_DIR = Path(os.getcwd()).parent
SEG_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/segmented_volumes20200728"
DICOM_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/dicoms"
WORK_SPACE = '/home/olle/PycharmProjects/LODE/workspace'
sys.path.append(str(PROJ_DIR))
