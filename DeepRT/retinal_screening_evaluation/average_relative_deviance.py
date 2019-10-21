import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

label_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/test_labels_export_1_filtered_preproc"
prediction_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/test_predictions_export_1_filtered_preproc"

records = os.listdir(label_path)

deviances = []
for record in records:
    #set paths
    gt_path = os.path.join(label_path, record)
    pred_path = os.path.join(prediction_path, record)

    #load images, resclae label
    gt = np.load(gt_path)
    pred = np.load(pred_path)

    #set nans to zero
    gt = np.nan_to_num(gt)
    pred = np.nan_to_num(pred)

    #resize thickness maps
    gt = cv2.resize(gt, (128,128))
    pred = cv2.resize(pred, (128,128)) * 500.

    #calculate deviance
    mean_relative_deviance = np.mean(np.abs(np.subtract(gt,pred))) / np.mean(gt)
    deviances.append(mean_relative_deviance)

print("the mean relative deviance is: {} with standard deviance of: {}".format(np.mean(deviances), np.std(deviances)))