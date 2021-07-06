import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
import json
from tqdm import tqdm
from utils.image_processing import read_resize
from utils.utils import label_mapping

from evaluate.evaluation_utils.utils import ensemble_predict
tf.compat.v1.disable_eager_execution()

NUMBER_OF_MODELS = 50

idv_path = "/home/olle/PycharmProjects/LODE/workspace/inter_doctor_variation"
result_save_dir = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/opt_ensemble/idv"

images = os.listdir(os.path.join(idv_path, "images"))

# get abs paths for models to be evaluated
model_paths = [
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_1/1745_L_20180706_640469001_46/model.h5',
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_1/338917_L_20160927_492116001_32/model.h5',
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_3/24844_R_20170502/model.h5',
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_2/90356_R_20160122/model.h5',
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_1/42510_L_20130722_281489001_22/model.h5',
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_2/22757_L_20130307_260204001_11/model.h5',
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_1/7561_R_20160202_443734001_17/model.h5',
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_3/61989_R_20170914_569274001_15/model.h5',
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_2/48221_L_20150914_418501001_42/model.h5',
    '/media/olle/3DCPC/oct_segmentation/cv_runs/cross_validation_runs0/logs_1/90356_R_20160122/model.h5'
]

ensemble_dict = {}

# get ensemble config dict
for model_path in tqdm(model_paths):
    # load test configurations
    model = load_model(model_path, compile=False)
    ensemble_dict[model_path] = {"model": model}


if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir, exist_ok=True)


for image_file_name in tqdm(images):
    image_name = image_file_name.replace(".png", "")
    if len(model_paths) > 0:
        img_path = os.path.join(idv_path, "images", image_file_name)

        ben_mask_p = os.path.join(idv_path, "iteration_idv_ben", "masks", image_file_name)
        johannes_mask_p = os.path.join(idv_path, "iteration_idv_johannes", "masks", image_file_name)
        michael_mask_p = os.path.join(idv_path, "iteration_idv_michael", "masks", image_file_name)
        concensus_mask_p = os.path.join(idv_path, "iteration_idv_concensun", "masks", image_file_name)

        img, ben_mask = read_resize(img_path, ben_mask_p, (256, 256))
        _, johannes_mask = read_resize(img_path, johannes_mask_p, (256, 256))
        _, michael_mask = read_resize(img_path, michael_mask_p, (256, 256))
        _, concensus_mask = read_resize(img_path, concensus_mask_p, (256, 256))

        # map artifacts and sereous ped to zero or fv-ped
        ben_mask = label_mapping(ben_mask)
        johannes_mask = label_mapping(johannes_mask)
        michael_mask = label_mapping(michael_mask)
        concensus_mask = label_mapping(concensus_mask)

        # run prediction for each ensemble model
        model_predictions, ensemble_prediction, uq_map = ensemble_predict(ensemble_dict, img)

        json_result_file = {"image": img.tolist(),
                            "ben_mask": ben_mask.tolist(),
                            "johannes_mask": johannes_mask.tolist(),
                            "michael_mask": michael_mask.tolist(),
                            "concensus_mask": concensus_mask.tolist(),
                            "prediction": ensemble_prediction.tolist()}

        with open(f'{result_save_dir}/{image_name}.json', 'w',  encoding='utf-8') as f:
            json.dump(json_result_file, f)
    else:
        print("record has not trained model")
