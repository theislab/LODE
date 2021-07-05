import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np
import tensorflow as tf
import pandas as pd
import glob
import random
from tqdm import tqdm

from keras.models import load_model

from utils.utils import label_mapping
from utils.image_processing import read_resize
from config import TRAIN_DATA_PATH, DATA_SPLIT_PATH

tf.compat.v1.disable_eager_execution()


def main(model_path, validation_ids, jsons_save_dir):
    """
    :param model_path:
    :type model_path:
    :param validation_ids:
    :type validation_ids:
    :return:
    :rtype:
    """
    # load test configurations
    model = load_model(model_path, compile = False)
    save_dir = os.path.join(jsons_save_dir, f"{random.randint(0, 1000000)}")

    for img_id in validation_ids:
        img_path = os.path.join(TRAIN_DATA_PATH, "images", img_id)
        label_path = os.path.join(TRAIN_DATA_PATH, "masks", img_id)

        img, lbl = read_resize(img_path, label_path, (256, 256))

        if len(img.shape) < 4:
            img = img.reshape(1, 256, 256, 3)

        # scale image
        img = img / 255.

        # map artifacts and sereous ped to zero or fv-ped
        lbl = label_mapping(lbl)

        # run prediction for each ensemble model
        prediction = model.predict(img)

        segmentation = np.argmax(prediction, -1)

        json_result_file = {"image": img.tolist(),
                            "label": lbl.tolist(),
                            "prediction": segmentation.tolist()}

        if not os.path.exists(f'{save_dir}'):
            os.makedirs(save_dir, exist_ok = True)

        with open(f'{save_dir}/{img_id.replace("png", "json")}', 'w', encoding = 'utf-8') as f:
            json.dump(json_result_file, f)

    with open(f'{save_dir}/model.json', 'w') as f:
        json.dump({"model_path": model_path}, f)


if __name__ == "__main__":
    model_directory = "/home/olle/PycharmProjects/LODE/workspace/validation_model_selection"
    jsons_save_dir = "/home/olle/PycharmProjects/LODE/workspace/validation_model_results"

    model_paths = glob.glob(model_directory + "/*/*.h5")

    for model_path in tqdm(model_paths):
        validation_ids = pd.read_csv(os.path.join(DATA_SPLIT_PATH, "validation_ids.csv"))["0"].tolist()
        main(model_path, validation_ids, jsons_save_dir)
