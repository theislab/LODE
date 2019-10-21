import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from jupyter_helper_functions import *
from tqdm import tqdm

gen_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/thickness_map_data_full_export/"
prediction_path = os.path.join(gen_path,"test_predictions")
label_path = os.path.join(gen_path, "thickness_maps")
save_path = os.path.join(gen_path, "test_predictions_low_res")

full_label_paths = [os.path.join(label_path,i) for i in os.listdir(prediction_path)]

low_res_pd = pd.DataFrame(index=os.listdir(prediction_path),
                   columns = ["C0_value", "S1_value", "S2_value", "N1_value", "N2_value", "I1_value", "I2_value", "T1_value", "T2_value"])

for count, pp in tqdm(enumerate(full_label_paths)):
    try:
        save_name = "low_res_" + pp.split("/")[-1]
        prediction = np.load(pp)
        full_size_pred = cv2.resize(prediction, (768,768), cv2.INTER_NEAREST)
        low_grid_values = get_low_res_depth_grid_maxvalues(full_size_pred)
        low_res_pd.iloc[count] = low_grid_values
    except:
        print("record throwing error: {}".format(pp))
low_res_pd.to_csv("label_etdrs_maxvalues.csv")