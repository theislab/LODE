import os
import numpy as np
import cv2

im_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/\
stratified_and_patient_split/output_export_1_filtered_preproc/test_predictions_export_1_unfiltered"

lbl_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/\
stratified_and_patient_split/export_1_unfiltered/test_predictions_niklas_pipeline"

im_paths = [os.path.join(im_path, i) for i in os.listdir(im_path)]
lbl_paths = [os.path.join(lbl_path,i) for i in os.listdir(lbl_path)]

for i in im_paths:
    cv2.imwrite(i,cv2.resize(cv2.imread(i), (128,128)))

for i in lbl_paths:
    np.save(i,cv2.resize(np.load(i),(128,128)))