import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/thickness_map_data_full_export/thickness_maps"
thickness_maps = [os.path.join(os.path.join(path, i)) for i in os.listdir(path)]

def total_variation(lbl):

    total_variation = np.nanmean(np.abs(np.ediff1d(lbl[30:128 - 30,
                                                                 30:128 - 30].flatten()))) / (np.nanmax(lbl) - np.nanmin(lbl))
    total_variation_t = np.nanmean(np.abs(np.ediff1d(np.transpose(lbl[30:128 - 30,
                                                                                30:128 - 30]).flatten()))) / (np.nanmax(lbl) - np.nanmin(lbl))
    tv = (total_variation + total_variation_t) / 2
    return(tv)

tv_log = [[],[]]
for map_path in thickness_maps:
    record = map_path.split("/")[-1]
    im_record = record.replace(".npy",".png")

    map = np.load(map_path)
    tv = total_variation(map)
    tv_log[0].append(record)
    tv_log[1].append(tv)
    #filter crit is 0.014

tv_pd = pd.DataFrame(tv_log).T.rename(columns={0:"id",1:"tv"})

tv_pd.to_csv("./tv_pd.csv")

