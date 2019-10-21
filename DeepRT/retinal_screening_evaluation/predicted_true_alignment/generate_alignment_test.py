import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from random import randrange
import cv2
import pandas as pd
import random

from jupyter_helper_functions import heidelberg_colormap, center_img, \
    get_low_res_depth_grid_values, get_low_res_grid_shape, get_text_coord, write_depthgrid_values

def plot_fundus_label_or_prediction_heidelberg_cs(record_path, image_path, save_path, save_name, laterality, prediction=True):
    if prediction:
        label_mu = cv2.resize(np.load(record_path).reshape(1,256,256,1)[0, :, :, 0], (768, 768)) * 500.
    else:
        label_mu = cv2.resize(np.load(record_path), (768, 768))

    cm_heidelberg = heidelberg_colormap()
    # center image
    label_mu[label_mu < 25] = 0
    label_mu = center_img(label_mu)

    # load image and set margin to zero and center
    fundus_image = cv2.resize(cv2.imread(image_path), (768, 768))
    fundus_image[label_mu == 0] = 0
    fundus_image = center_img(fundus_image)

    # get values for low res grid and coordinates
    label_mu = np.nan_to_num(label_mu)
    low_grid_values_label = get_low_res_depth_grid_values(label_mu)

    # overlay nine area grid
    label_mu = np.nan_to_num(label_mu)
    label = np.copy(label_mu)
    low_res_grid = get_low_res_grid_shape(label_mu)
    label[low_res_grid.astype(np.bool)[:, :, 0]] = 0

    coord_list = get_text_coord(label_mu)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(fundus_image)
    plt.title("fundus")
    plt.subplot(1, 3, 2)
    label_mu = np.ma.masked_where(label_mu < 100, label_mu)
    label_mu[label_mu > 500.0] = 1000
    cmap = cm_heidelberg
    cmap.set_bad(color='black')
    plt.imshow(label_mu, cmap=cmap, vmin=100, vmax=750)
    plt.title("high res:{}, laterality: {}".format(save_name, laterality))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1, 3, 3)
    label = np.ma.masked_where(label < 100, label)
    cmap = cm_heidelberg
    cmap.set_bad(color='black')

    plt.imshow(label, cmap=cmap, vmin=100, vmax=750)
    plt.title("low res:{}, laterality: {}".format(save_name, laterality))
    write_depthgrid_values(coord_list, low_grid_values_label,text_size="large")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig(os.path.join(save_path, str(save_name)))
    plt.close()

path = "mapping.csv"
data_dir = "/media/olle/Seagate/thickness_map_prediction/data"
save_dir = "./records"
records = pd.read_csv(path)[["record_name"]]

prediction_pseudo = random.sample(range(records.shape[0], records.shape[0]*2), records.shape[0])
label_pseudo = random.sample(range(0, records.shape[0]), records.shape[0])

#add to data frame
records["prediction_pseudo"] = pd.DataFrame(prediction_pseudo)
records["label_pseudo"] = pd.DataFrame(label_pseudo)

#save mapping
records.to_csv("./mapping.csv")

for i in range(records.shape[0]):

    lbl = np.load(os.path.join(data_dir,"thickness_maps",records["record_name"].iloc[i]+".npy"))
    prd = np.load(os.path.join(data_dir,    "test_predictions_gold_standard",records["record_name"].iloc[i]+".npy"))
    im = cv2.imread(os.path.join(data_dir,"fundus",records["record_name"].iloc[i]+".png"))

    try:
        #create all plots
        plot_fundus_label_or_prediction_heidelberg_cs(record_path=os.path.join(data_dir,"thickness_maps",records["record_name"].iloc[i]+".npy"),
                                                      image_path=os.path.join(data_dir,"fundus",records["record_name"].iloc[i]+".png"),
                                                      save_path="./records",
                                                      save_name=str(records["label_pseudo"].iloc[i])+".png",
                                                      laterality=records["record_name"].iloc[i].split("_")[1],
                                                      prediction=False)

        plot_fundus_label_or_prediction_heidelberg_cs(record_path=os.path.join(data_dir,"test_predictions_gold_standard",records["record_name"].iloc[i]+".npy"),
                                                      image_path=os.path.join(data_dir,"fundus",records["record_name"].iloc[i]+".png"),
                                                      save_path="./records",
                                                      save_name=str(records["prediction_pseudo"].iloc[i])+".png",
                                                      laterality=records["record_name"].iloc[i].split("_")[1],
                                                      prediction=True)

    except:
        print("record not working:",records["record_name"].iloc[i])
        continue