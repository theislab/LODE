import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import random
from jupyter_helper_functions import *

eval_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/"
data_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/\
data/stratified_and_patient_split/"

label_path = os.path.join(data_path,"export_2_filtered_total_variation/test_labels")
prediction_path = os.path.join(eval_path,"test_predictions_total_var_filtered_clean_split")
image_path = os.path.join(data_path,"export_2_filtered_total_variation/test_images")

label_paths = [os.path.join(label_path,i) for i in os.listdir(prediction_path)]
prediction_paths = [os.path.join(prediction_path,i) for i in os.listdir(prediction_path)]


test_label_foveas = []
test_prediction_foveas = []
label_means = []
pred_means = []
names = []
for lp in label_paths:
    record_name = lp.split("/")[-1]
    label_mu = cv2.resize(np.load(lp),(768,768))
    pred_mu = cv2.resize(np.load(os.path.join(prediction_path,record_name))[0,:,:,0], (768, 768)) * 500.
    names.append(record_name)
    label_fovea_value = get_low_res_depth_grid_values(label_mu)[0]
    pred_fovea_value = get_low_res_depth_grid_values(pred_mu)[0]

    test_label_foveas.append(label_fovea_value)
    test_prediction_foveas.append(pred_fovea_value)

np.savetxt("test_label_fovea_totalvar_clean_split.txt", test_label_foveas)
np.savetxt("test_pred_fovea_totalvar_clean_split.txt", test_prediction_foveas)
np.savetxt("names_fovea_totalvar_clean_split.txt", names, delimiter=" ", fmt="%s")
print(np.mean(test_label_foveas))
