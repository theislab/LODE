import os
import pandas as pd
import numpy as np

export_path = "/home/olle/PycharmProjects/thickness_map_prediction/project_evaluation/export_2_foveas_total_var_clean_split"
af = os.path.join(export_path,"test_label_fovea_totalvar_clean_split.txt")
pf = os.path.join(export_path,"test_pred_fovea_totalvar_clean_split.txt")
imn = os.path.join(export_path,"names_fovea_totalvar_clean_split.txt")

#load fovea statistics
actual_f = pd.read_csv(af, header=None,names=["actual_fovea"])
predicted_f = pd.read_csv(pf, header=None,names=["predicted_fovea"])
im_names = pd.read_csv(imn, header=None,names=["image_names"])
#merge
fovea_table = pd.concat([im_names,actual_f,predicted_f],axis=1)
#calculate difference
fovea_table["fovea_difference"] = np.abs(fovea_table.actual_fovea - fovea_table.predicted_fovea) / fovea_table.actual_fovea

#derive in range booleans
fovea_table["is_actual_fovea_normal"] = (fovea_table.actual_fovea > 250) & (fovea_table.actual_fovea < 291)
fovea_table["is_predicted_fovea_normal"] = (fovea_table.predicted_fovea > 250) & (fovea_table.predicted_fovea < 291)
fovea_table["is_actual_fovea_abnormal"] = (fovea_table.actual_fovea < 250) | (fovea_table.actual_fovea > 291)
fovea_table["is_predicted_fovea_abnormal"] = (fovea_table.predicted_fovea < 250) | (fovea_table.predicted_fovea > 291)

#get final boolean evaluation column
fovea_table["ground_truth_prediction_normal_range_alignment"] = ((fovea_table["is_actual_fovea_normal"]) & (fovea_table["is_predicted_fovea_normal"])) | \
       ((fovea_table["is_actual_fovea_normal"]) & (fovea_table.fovea_difference < 0.2))

#get final boolean evaluation column
fovea_table["ground_truth_prediction_abnormal_range_alignment"] = ((fovea_table["is_actual_fovea_abnormal"]) & (fovea_table["is_predicted_fovea_abnormal"])) | \
       ((fovea_table["is_actual_fovea_abnormal"]) & (fovea_table.fovea_difference < 0.2))

#print relevant statistics
num_normal_alignment = np.sum(fovea_table["ground_truth_prediction_normal_range_alignment"])
num_normal = np.sum(fovea_table["is_actual_fovea_normal"])
print("Percentage of normal range alignment is: {}".format(float(num_normal_alignment) / num_normal))

num_abnormal_alignment = np.sum(fovea_table["ground_truth_prediction_abnormal_range_alignment"])
num_abnormal = np.sum(fovea_table["is_actual_fovea_abnormal"])
print("Percentage of abnormal range alignment is: {}".format(float(num_abnormal_alignment) / num_abnormal))

'''
#bin data and categorize it
bins = np.histogram(fovea_table.actual_fovea, bins=4)[1]
fovea_table["fovea_bins"] = pd.cut(fovea_table.actual_fovea, bins).cat.codes
train_files = fovea_table[["image_names","fovea_bins"]]
train_files.to_csv("/home/olle/PycharmProjects/thickness_map_prediction/fundus_to_HDdepth_map_prediction/train_files.csv")
'''

