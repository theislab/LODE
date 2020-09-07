import sys
import os
from pathlib import Path

PROJ_DIR = Path(os.getcwd()).parent
SEG_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/segmented_volumes20200728"
WORK_SPACE = os.path.join(PROJ_DIR, "workspace")  # '/storage/groups/ml01/workspace/olle.holmberg/LODE'
sys.path.append(str(PROJ_DIR))
