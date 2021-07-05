import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np
import tensorflow as tf
import pandas as pd

from keras.models import load_model

from utils.utils import label_mapping
from utils.image_processing import read_resize
from config import TRAIN_DATA_PATH, DATA_SPLIT_PATH

tf.compat.v1.disable_eager_execution()


def main(model_path, validation_ids):
    """
    :param model_path:
    :type model_path:
    :param validation_ids:
    :type validation_ids:
    :return:
    :rtype:
    """
    # load test configurations
    model = load_model(model_path + "/model.h5", compile = False)

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

        if not os.path.exists(f'{model_path}/validation_results'):
            os.makedirs(f'{model_path}/validation_results', exist_ok = True)

        with open(f'{model_path}/validation_results/{img_id.replace("png", "json")}', 'w', encoding = 'utf-8') as f:
            json.dump(json_result_file, f)


if __name__ == "__main__":
    model_path = "/home/olle/PycharmProjects/LODE/workspace/validation_model_selection/135590_L_20171201_586379001_17"

    # get all cv runs
    validation_ids = pd.read_csv(os.path.join(DATA_SPLIT_PATH, "validation_ids.csv"))["0"].tolist()
    main(model_path, validation_ids)
