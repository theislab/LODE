import os
import pandas as pd
from tqdm import tqdm
import sys
import glob
import argparse
from pathlib import Path
import cv2


path_variable = Path(os.path.dirname(__file__))
sys.path.insert(0, str(path_variable))
sys.path.insert(0, str(path_variable.parent))

from config import PROJ_DIR

search_paths = [i for i in glob.glob(PROJ_DIR + "/*/*") if os.path.isdir(i)]

for sp in search_paths:
    sys.path.append(sp)

print("path search completed")

from utils.plotting import save_segmentation_plot, plot_image_predictions, \
            plot_uncertainty_heatmaps, plot_uncertainty_statistics
from evaluate.evaluation_utils import check_enseble_test_ids, get_ensemble_dict, \
    SEGMENTED_CLASSES, ensemble_predict, segmentation_to_vector, save_segmentation, initialize_volume_feature_dict, oct_segmentation_to_vector
from generators.generator_utils.image_processing import read_oct_from_dicom, read_resize_image
from config import WORK_SPACE, VOL_SAVE_PATH, EMBEDD_SAVE_PATH

def remove_image_suffix(record_string):
    suffixes = [".tif", ".tiff", ".TIFF", ".TIF", ".jpeg", ".png", ".jpg"]
    for suffix_ in suffixes:
        record_string = record_string.replace(suffix_, "")
    return record_string

if __name__ == "__main__":
    # select model to be evaluated
    ensemble_name = "ensemble_stratified_camera_effect"
    ENSEMBLE_SPACE = "/home/icb/olle.holmberg/projects/LODE/feature_segmentation/segmentation_ensembles"
    ensemble_models = ["83", "82", "81", "80", "79"]
    ensemble_dir = os.path.join(ENSEMBLE_SPACE, f"{ensemble_name}/models")

    # get dictionary holding all models
    ensemble_dict = get_ensemble_dict(ensemble_models, ensemble_dir)

    # assert correct train test split across ensemble
    check_enseble_test_ids(ensemble_dict)

    shape = (256, 256)
    save_oct = True
    uq_maps = {}
    
    data_path = "/storage/groups/ml01/datasets/raw/20210420_olle_holmberg_di/oct_public_datasets/cell_oct_data"
    data_csv = pd.read_csv(os.path.join(data_path, "data.csv"))

    print("starting the segmentation")
    i = 0
    for i, oct_path in enumerate(tqdm(data_csv.oct_path.values)):
        oct_path = os.path.join(data_path, oct_path.replace("./", ""))
        
        o_path = Path(oct_path)
        oct_ = read_resize_image(oct_path, shape)
        model_segmentations, ensemble_prediction, uq_map = ensemble_predict(ensemble_dict, oct_)
        save_segmentation(ensemble_prediction, save_path = data_path + "/segmentation", 
                record_name = remove_image_suffix(oct_path.split('/')[-1]))

