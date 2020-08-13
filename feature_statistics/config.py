import sys
import os
from pathlib import Path
PROJ_DIR = Path(os.getcwd()).parent
DATA_DIR = "/storage/groups/ml01/datasets/projects/20181610_eyeclinic_niklas.koehler/clinical_feature_segmentation/segmented_volumes20200728"
WORK_SPACE = os.path.join(PROJ_DIR, "workspace") # '/storage/groups/ml01/workspace/olle.holmberg/LODE'

sys.path.append(str(PROJ_DIR))
