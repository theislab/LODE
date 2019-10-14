from input_data import *
import numpy as np
import pandas as pd
import depth_vector as dv
import os
from model import *
from train_eval_ops import *
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import models
import cv2

params = {}
params["img_shape"] = (256, 256, 3)
params["batch_size"] = 1
params["data_dir"] = "/media/olle/Seagate Expansion Drive/DRD_master_thesis_olle_holmberg/augen_clinic_data/image_data/"
params["save_path"] = "./output"
params["save_hd_path"] = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/hd_depth_maps"
params["learning_rate"] = 0.001
look_up_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/full_data.csv"
save_model_path = os.path.join(params["save_path"],"weights.hdf5")

look_up_table = pd.read_csv(look_up_path)

#set full paths with actual data dir
look_up_table["oct_path"] = params["data_dir"] + look_up_table["oct_path"]
look_up_table["localizer_path"] = params["data_dir"] + look_up_table["localizer_path"]

for record_id in pd.unique(look_up_table.localizer_path):
    exist = cv2.imread(record_id)
    #exist = os.path.isfile(record_id)
    if exist is None:
        print(record_id)


