import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import glob
import json
from tqdm import tqdm

from feature_segmentation.evaluate.evaluation_utils.utils import ensemble_predict
from feature_segmentation.generators.generator_2d import label_mapping
from feature_segmentation.generators.generator_utils.image_processing import read_resize
from feature_segmentation.config import TRAIN_DATA_PATH, DATA_SPLIT_PATH

tf.compat.v1.disable_eager_execution()


cv_model_path = "/media/olle/3DCPC/oct_segmentation/cross_validation_runs"
result_save_dir = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/cv_runs_results"

# get all cv runs
test_ids = pd.read_csv(os.path.join(DATA_SPLIT_PATH, "test_ids.csv"))["0"].tolist()
cv_runs = [id.replace(".png", "") for id in test_ids]


for cv_run in tqdm(cv_runs):
    # get abs paths for models to be evaluated
    model_paths = glob.glob(f"{cv_model_path}/*/{cv_run}/*.*h5")

    if len(model_paths) > 0:
        ensemble_dict = {}

        # get ensemble config dict
        for model_path in tqdm(model_paths):
            # load test configurations
            model = load_model(model_path, compile=False)
            ensemble_dict[model_path] = {"model": model}

        img_path = os.path.join(TRAIN_DATA_PATH, "images", cv_run + ".png")
        label_path = os.path.join(TRAIN_DATA_PATH, "masks", cv_run + ".png")

        img, lbl = read_resize(img_path, label_path, (256, 256))

        # map artifacts and sereous ped to zero or fv-ped
        lbl = label_mapping(lbl)

        # run prediction for each ensemble model
        model_predictions, ensemble_prediction, uq_map = ensemble_predict(ensemble_dict, img)

        json_result_file = {"image": img.tolist(), "label": lbl.tolist(), "prediction": ensemble_prediction.tolist()}

        with open(f'{result_save_dir}/{cv_run}.json', 'w',  encoding='utf-8') as f:
            json.dump(json_result_file, f)
    else:
        print("record has not trained model")
