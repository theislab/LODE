import os
from keras.engine.saving import load_model
from utils import Params, TrainOps, Evaluation, InferEnsemble, Logging
import random
import numpy as np
from sklearn.metrics import jaccard_score
import pandas as pd
from segmentation_models.metrics import iou_score
import json
from tqdm import tqdm
from pprint import pprint
import seaborn as sns
# select model to be evaluated
from plotting import plot_model_run_images
import matplotlib.pyplot as plt

from feature_segmentation.utils.plotting import plot_image_predictions

model_directory = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/ensemble_segmentation"
models = ["44", "47", "48", "49"]

# load utils classes
params = Params(os.path.join(model_directory, models[0], "config.json"))

# cast data types to numeric
params = params.dict
for k in params.keys():
    try:
        int(params[k])
        params[k] = int(params[k])
    except ValueError:
        try:
            print("Not a int")
            float(params[k])
            params[k] = float(params[k])
        except ValueError:
            print("Not an int or  float")

with open(os.path.join(model_directory, models[0], 'params.json'), 'w') as json_file:
    json.dump(params, json_file)

params = Params(os.path.join(model_directory, models[0], "params.json"))
logging = Logging("./logs", params)
trainops = TrainOps(params)

params.data_path = "/media/olle/Seagate/datasets/archive/OCT2017 /test/CNV"

ids = os.listdir(params.data_path)
random.shuffle(ids)

test_ids = ids  # pd.read_csv(os.path.join(model_directory, models[0], "validation_ids.csv"))["0"].tolist()

partition = {'test': test_ids}

params.is_training = False

ensemble = {}
for model_name in tqdm(models):
    save_model_path = os.path.join(model_directory, model_name, "weights.hdf5")
    model = load_model(save_model_path, custom_objects={'iou_score': iou_score})
    ensemble[model_name] = model

from sklearn.metrics import classification_report

target_dict = {0: 'class 0', 1: 'class 1', 2: 'class 2',
               3: 'class 3', 4: 'class 4', 5: 'class 5',
               6: 'class 6', 7: 'class 7', 8: 'class 8',
               9: 'class 9', 10: 'class 10', 11: 'class 11',
               12: 'class 12', 13: 'class 13', 14: 'class 14',
               15: 'class 15'}

all_predictions = []
all_labels = []

ensemble_model_directory = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/ensemble_segmentation" \
                           "/ensemble "

if not os.path.exists(ensemble_model_directory):
    os.makedirs(ensemble_model_directory, exist_ok=True)

image_iter = 0
for i in range(0, len(test_ids) - 1):
    evaluation = InferEnsemble(params=params,
                               filename=test_ids[i],
                               ensemble=ensemble,
                               mode="test",
                               choroid=params.choroid_latest)

    # evaluation.model_segmentations, evaluation.ensemble_predictions, evaluation.prediction
    all_predictions.extend(evaluation.prediction.flatten().tolist())

    records = [evaluation.image, evaluation.prediction]
    plot_image_predictions(records, ensemble_model_directory, mode="test",
                           filename=test_ids[i].replace(".png", "ensemble.png"))

    # save individual model predictions
    for model_ in evaluation.model_segmentations.keys():
        records = [evaluation.image, evaluation.model_segmentations[model_]]
        plot_image_predictions(records, ensemble_model_directory, mode="test",
                               filename=test_ids[i].replace(".png", f"{model_}.png"))

target_names = []
labels_present = np.unique(np.unique(all_labels).tolist() + np.unique(all_predictions).tolist())
for lp in labels_present:
    target_names.append(target_dict[lp])

ious = jaccard_score(all_labels, all_predictions, average=None)
clf_report = classification_report(all_labels, all_predictions, target_names=target_names, output_dict=1)

print(classification_report(all_labels, all_predictions, target_names=target_names))

pprint(f"The class ious are: {ious}")
pprint(f"The mIOU is {np.mean(ious)}")

np.savetxt(model_directory + "/ious.txt", ious)
pd.DataFrame(clf_report).to_csv(model_directory + "/clf_report.csv")
