import os
from keras.engine.saving import load_model
from utils.utils import Params, TrainOps, Evaluation, Logging
import random
import numpy as np
from sklearn.metrics import jaccard_score
import pandas as pd
from segmentation_models.metrics import iou_score
import json

# select model to be evaluated
model_directory = "./logs/44"
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

target_names = ['class 0', 'class 1', 'class 2',
                'class 3', 'class 4', 'class 5',
                'class 6', 'class 7', 'class 8',
                'class 9', 'class 10', 'class 11', 'class 12', 'class 13', 'class 14',
                'class 15']

all_predictions = []
all_labels = []

image_iter = 0
for i in range(0,len(test_ids)-1):
    evaluation = Evaluation(params=params,
                            filename=test_ids[i],
                            model=model,
                            mode="test")
    print(test_ids[i], np.unique(evaluation.prediction), np.unique(evaluation.label))
    all_predictions.extend(evaluation.prediction.flatten().tolist())
    all_labels.extend(evaluation.label.flatten().tolist())



    # plot result
    evaluation.plot_record()

print(classification_report(all_labels, all_predictions, target_names=target_names))
print(jaccard_score(all_labels, all_predictions, average=None))
