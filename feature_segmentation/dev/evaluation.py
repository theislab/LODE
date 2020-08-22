import os
from utils import Params, TrainOps, Evaluation, Logging
import random
import numpy as np
from model import unet
from sklearn.metrics import jaccard_score
import pandas as pd

# load utils classes
params = Params("params.json")
logging = Logging("./logs", params)
trainops = TrainOps(params)

ids = os.listdir(params.data_path + "hq_images")
random.shuffle(ids)

# select model to be evaluated
params.model_directory = "./logs/18"

test_ids = pd.read_csv(os.path.join(params.model_directory, "test_images.csv"), header=None)[0].tolist()

partition = {'test': test_ids}

params.is_training = False

'''get model'''
model = unet(params)

'''train and save model'''
save_model_path = os.path.join(params.model_directory, "weights.hdf5")

model.load_weights(save_model_path, by_name = True)

from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1', 'class 2',
                'class 3', 'class 4', 'class 5',
                'class 6', 'class 7', 'class 8',
                'class 9', 'class 10', 'class 11']

all_predictions = []
all_labels = []

image_iter = 0
for i in range(0,len(test_ids)-1):
    evaluation = Evaluation(params=params,
                            filename=test_ids[i],
                            model=model,
                            mode="test")

    all_predictions.extend(evaluation.prediction.flatten().tolist())
    all_labels.extend(evaluation.label.flatten().tolist())

    # plot result
    evaluation.plot_record()

print(classification_report(all_labels, all_predictions, target_names=target_names))
print(np.mean(jaccard_score(all_labels, all_predictions, average=None)))