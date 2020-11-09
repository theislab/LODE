import os
import pandas as pd
from tqdm import tqdm
import sys
import glob
import argparse
from pathlib import Path
import cv2

from feature_segmentation.utils.generate_model_jsons import create_model_json, save_model_json
from feature_segmentation.utils.plotting import save_segmentation_plot, plot_image_predictions, \
    plot_uncertainty_heatmaps, plot_uncertainty_statistics

path_variable = Path(os.path.dirname(__file__))
sys.path.insert(0, str(path_variable))
sys.path.insert(0, str(path_variable.parent))

from segmentation_config import PROJ_DIR

search_paths = [i for i in glob.glob(PROJ_DIR + "/*/*") if os.path.isdir(i)]

for sp in search_paths:
    sys.path.append(sp)

from feature_segmentation.evaluate.evaluation_utils import check_enseble_test_ids, get_ensemble_dict, \
    SEGMENTED_CLASSES, ensemble_predict, segmentation_to_vector, save_segmentation, initialize_volume_feature_dict, \
    oct_segmentation_to_vector
from feature_segmentation.generators.generator_utils.image_processing import read_oct_from_dicom, read_resize_image
from feature_segmentation.segmentation_config import WORK_SPACE, VOL_SAVE_PATH, EMBEDD_SAVE_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_paths", help = "which of the different sub .csv files to read from ",
                        type = str, default = "test")
    # args = parser.parse_args()

    # select model to be evaluated
    ensemble_name = "ensemble_stratified_camera_effect"
    ENSEMBLE_SPACE = "/media/olle/Seagate/LODE/workspace/segmentation_ensembles"
    ensemble_models = ["83", "82", "81", "80", "79"]
    ensemble_dir = os.path.join(ENSEMBLE_SPACE, f"{ensemble_name}/models")

    # get dictionary holding all models
    ensemble_dict = get_ensemble_dict(ensemble_models, ensemble_dir)

    # assert correct train test split across ensemble
    check_enseble_test_ids(ensemble_dict)

    META_COLUMNS = ['id']
    feature_statistics = pd.DataFrame(columns=SEGMENTED_CLASSES + META_COLUMNS)

    shape = (256, 256)
    save_oct = True
    save_oct_json = True
    uq_maps = {}
    data_path = "/media/olle/Seagate/datasets/cell_oct_data"
    oct_paths = glob.glob(data_path + "/oct2017/OCT2017/*/*/*.jpeg")
    i = 0
    for i, oct_path in enumerate(tqdm(oct_paths)):
        o_path = Path(oct_path)
        try:
            oct_ = read_resize_image(oct_path, shape)

            if oct_ is not None:
                model_segmentations, ensemble_prediction, uq_map = ensemble_predict(ensemble_dict, oct_)
                feature_dict = oct_segmentation_to_vector(ensemble_prediction)

                feature_dict["id"] = f"{oct_path.split('/')[-1]}"

                uq_maps[feature_dict["id"]] = uq_map

                # append to data frame
                feature_statistics = feature_statistics.append(pd.DataFrame(feature_dict, index=[0]), sort=True)

                if save_oct:
                    save_segmentation(ensemble_prediction, data_path + "/segmentation",
                                      feature_dict["id"].replace(".png", ""))

                if save_oct_json:
                    oct_ = cv2.resize(oct_, (ensemble_prediction.shape[0:2]), interpolation=cv2.INTER_LINEAR)

                    test_json = create_model_json(feature_dict["id"], ensemble_prediction, oct_)
                    save_model_json(test_json, save_dir=data_path + "/json",
                                    name=feature_dict["id"].replace("jpeg", "json"))

                plot_image_predictions([oct_, ensemble_prediction],
                                       data_path + "/segmentation_plots",
                                       "test",
                                       feature_dict["id"].replace(".png", ""))
            else:
                print("oct does not have requested shape, skipping")
                continue

            if i % 1000 == 0:
                # plot all uq maps
                plot_uncertainty_heatmaps(uq_maps, data_path + "/uq_heatmap")

                # reset uq map dict
                uq_maps = {}

                # progress
                feature_statistics.to_csv(data_path + "/feature_statistics.csv")
        except:
            print("record not working, skipping: ", oct_path)



