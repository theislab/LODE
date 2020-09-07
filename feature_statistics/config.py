import sys
import os
from pathlib import Path

PROJ_DIR = Path(os.getcwd()).parent
SEG_DIR = "/storage/groups/ml01/workspace/olle.holmberg/LODE/feature_segmentation/segmentation/segmented_volumes20200822"
WORK_SPACE = '/storage/groups/ml01/workspace/olle.holmberg/LODE'
sys.path.append(str(PROJ_DIR))
