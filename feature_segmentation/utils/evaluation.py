import os
from keras.engine.saving import load_model
from utils import Params, TrainOps, Evaluation, Logging
import random
import numpy as np
from sklearn.metrics import jaccard_score
import pandas as pd
from segmentation_models.metrics import iou_score
import json
from pprint import pprint

# select model to be evaluated
from plotting import plot_model_run_images

model_directory = "./logs/21"

# load utils classes
params = Params(os.path.join(model_directory, "config.json"))

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

with open(os.path.join(model_directory, 'params.json'), 'w') as json_file:
  json.dump(params, json_file)

params = Params(os.path.join(model_directory, "params.json"))
logging = Logging("./logs", params)
trainops = TrainOps(params)

ids = os.listdir(params.data_path + "/images")
random.shuffle(ids)

test_ids = pd.read_csv(os.path.join(model_directory, "test_ids.csv"))["0"].tolist()

partition = {'test': test_ids}

params.is_training = False

save_model_path = os.path.join(model_directory, "weights.hdf5")

model = load_model(save_model_path, custom_objects={'iou_score': iou_score})# model.load_weights(save_model_path, by_name=True)

from sklearn.metrics import classification_report

target_dict = {0:'class 0', 1:'class 1', 2:'class 2',
                3: 'class 3', 4: 'class 4', 5: 'class 5',
                6: 'class 6', 7: 'class 7', 8: 'class 8',
                9: 'class 9', 10: 'class 10', 11: 'class 11',
                12: 'class 12', 13: 'class 13', 14: 'class 14',
                15: 'class 15'}

all_predictions = []
all_labels = []

image_iter = 0
for i in range(0,len(test_ids)-1):
    evaluation = Evaluation(params=params,
                            filename=test_ids[i],
                            model=model,
                            mode="test",
                            choroid = params.choroid_latest)

    all_predictions.extend(evaluation.prediction.flatten().tolist())
    all_labels.extend(evaluation.label.flatten().tolist())

    records = [evaluation.image, evaluation.label, evaluation.prediction]
    plot_model_run_images(records, model_directory, mode="test", filename = test_ids[i])

target_names = []
labels_present = np.unique(np.unique(all_labels).tolist() + np.unique(all_predictions).tolist())
for lp in labels_present:
    target_names.append(target_dict[lp])
    

ious = jaccard_score(all_labels, all_predictions, average=None)
print(classification_report(all_labels, all_predictions, target_names=target_names))

pprint(f"The class ious are: {ious}")
pprint(f"The mIOU is {np.mean(ious)}")
