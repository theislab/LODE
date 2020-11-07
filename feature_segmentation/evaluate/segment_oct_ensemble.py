import os
import pandas as pd
from tqdm import tqdm
import sys
import glob
import argparse
from pathlib import Path

from feature_segmentation.utils.plotting import save_segmentation_plot

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
    ensemble_name = "ensemble_stratified"
    ensemble_models = ["56"]
    ensemble_dir = os.path.join(WORK_SPACE, f"segmentation/{ensemble_name}")

    # get dictionary holding all models
    ensemble_dict = get_ensemble_dict(ensemble_models, ensemble_dir)

    # assert correct train test split across ensemble
    check_enseble_test_ids(ensemble_dict)

    META_COLUMNS = ['id']
    feature_statistics = pd.DataFrame(columns = SEGMENTED_CLASSES + META_COLUMNS)

    shape = (256, 256)
    save_oct = True

    data_path = WORK_SPACE + "/segmentation/data/versions/iteration_10"
    oct_paths = glob.glob(data_path + "/*/*.png")
    i = 0
    for oct_path in tqdm(oct_paths):
        try:
            oct_ = read_resize_image(oct_path, shape)

            if oct_ is not None:
                model_segmentations, ensemble_prediction, uq_map = ensemble_predict(ensemble_dict, oct_)
                feature_dict = oct_segmentation_to_vector(ensemble_prediction)

                feature_dict["id"] = f"{oct_path.split('/')[-1]}"

                # append to data frame
                feature_statistics = feature_statistics.append(pd.DataFrame(feature_dict, index=[0]), sort = True)

                if save_oct:
                    save_segmentation(ensemble_prediction, os.path.dirname(oct_path),
                                      feature_dict["id"].replace(".png", ""))

                save_segmentation_plot(oct_path.replace(".png", "_seg.png"), ensemble_prediction)
            else:
                print("oct does not have requested shape, skipping")
                continue
        except:
            print("Dicom was not able to be processed, skipping")
        i += 1
    # progress
    feature_statistics.to_csv(data_path + "/feature_statistics.csv")
