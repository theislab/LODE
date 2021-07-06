from input_data import *
import numpy as np
import pandas as pd
import depth_vector as dv
from pydicom import read_file
import fnmatch
import os
from model import *
from train_eval_ops import *
from tensorflow.python.tensorflow.keras import optimizers
from tensorflow.python.tensorflow.keras import models
import cv2
from load_dicom_data import *
import logging
import matplotlib.pyplot as plt

from thickness_map_calculation.load_dicom_data import get_dicom_look_up


def get_iterable_dimension(record_lookup):
    y_iter = None
    x_iter = None

    # determine y or x iterable
    y = np.unique(record_lookup.y_starts).shape[0]
    x = np.unique(record_lookup.x_starts).shape[0]
    if y > x:
        y_iter = "iterable"
    else:
        x_iter = "iterable"
    return y_iter, x_iter

def oct_pixel_to_mu_m(record_lookup, depth_vector,iter):
    y_scale = float(record_lookup.y_scales.iloc[iter])
    return (np.multiply(depth_vector, y_scale * 1000))

def get_position_series(record_lookup):
    startx_pos = record_lookup.x_starts.reset_index(drop=True).fillna(0)
    endx_pos = record_lookup.x_ends.reset_index(drop=True).fillna(0)
    starty_pos = record_lookup.y_starts.reset_index(drop=True).fillna(0)
    endy_pos = record_lookup.y_ends.reset_index(drop=True).fillna(0)
    return (startx_pos, endx_pos, starty_pos, endy_pos)

def get_oct_and_segmentation(record_lookup):
    #read in oct
    dc = read_file(record_lookup.dicom_paths.iloc[0][0])
    oct_images = dc.pixel_array

    # predict segmentation masks for all oct images
    oct_segmentations = []
    for i in range(0,oct_images.shape[0]):
        predict_shape = (1, 256, 256, 3)
        orig_height = 496
        orig_width = 512

        #stack oct image
        stacked_img = np.stack((oct_images[i,:,:],) * 3, axis=-1)
        #resize and scale stacked image
        resized_image = cv2.resize(stacked_img,(256,256)) / 255.
        #reshape image for prediction
        reshaped_image = resized_image.reshape(1,256,256,3)
        oct_segmentations.append(cv2.resize(model.predict(reshaped_image)[0,:,:,0],
                                            (orig_height,orig_width)).astype(np.uint8))

    oct_segmentations = np.array(oct_segmentations)*255.
    return oct_images, oct_segmentations

def get_depth_grid(record_lookup, oct_segmentations):
    # get fundus dimension
    depth_grid_dim = (768,768)
    y_cord, x_cord = get_iterable_dimension(record_lookup)
    grid = np.zeros(depth_grid_dim)
    for i in range(0, len(oct_segmentations)):
        d_v = dv.get_depth_vector(oct_segmentations[i])
        #scale dv to mm
        d_v = oct_pixel_to_mu_m(record_lookup, d_v, i)
        #get starting and ending x,y series with new indices
        startx_pos, endx_pos, starty_pos, endy_pos = get_position_series(record_lookup)
        try:
            if y_cord == "iterable":
                # set assert indices are ints
                x_start = int(startx_pos[i])
                x_end = int(endx_pos[i])
                y_start = int(starty_pos[i])

                # assert d_v has same width as x_end -x_start
                if d_v.shape[0] > (x_end - x_start):
                    d_v = d_v[0:x_end - x_start]
                if d_v.shape[0] < (x_end - x_start):
                    difference = (x_end - x_start) - d_v.shape[0]
                    d_v = np.append(d_v, np.zeros(int(difference)))
                # shift indices when laterilty changes to "L"
                # in case x_start is negative in xml it is set to zero and x_end
                # reduced correspondingly to fit the depth vector
                #scale depth vector to mm

                grid[y_start, x_start:x_end] = d_v
            if x_cord == "iterable":
                # assert indices are ints
                # set assert indices are ints
                y_start = int(starty_pos[i])
                y_end = int(endy_pos[i])
                x_start = int(startx_pos[i])

                # assert d_v has same width as x_end -x_start
                if d_v.shape[0] > (y_start - y_end):
                    d_v = d_v[0:y_start - y_end]
                if d_v.shape[0] < (y_start - y_end):
                    difference = (y_start - y_end) - d_v.shape[0]
                    d_v = np.append(d_v, np.zeros(difference))
                # shift indices when laterilty changes to "L"
                grid[y_end:y_start, x_start] = d_v
        except:
                print("COULD NOT CALCULATE GRID")
                print(record_lookup.localizer_path.iloc[0])
    # linearly interpolate missing values
    grid[grid == 0] = np.nan
    # interpolate depending on which axis the depth vector is filled in
    if x_cord == "iterable":
        grid_pd_int = pd.DataFrame(grid).interpolate(limit_direction='both',
                                                     axis=1)
        # set all areas outside of measurements to 0
        min_startx = int(min(startx_pos.iloc[1:]))
        max_startx = int(max(startx_pos.iloc[1:]))

        grid_pd_int.loc[:, max_startx:grid_pd_int.shape[1]] = 0
        grid_pd_int.loc[:, 0:min_startx] = 0

    if y_cord == "iterable":
        grid_pd_int = pd.DataFrame(grid).interpolate(limit_direction='both',
                                                     axis=0)
        # set all areas outside of measurements to 0
        min_starty = int(min(starty_pos.iloc[1:]))
        max_starty = int(max(starty_pos.iloc[1:]))

        grid_pd_int[max_starty:grid_pd_int.shape[0]] = 0
        grid_pd_int[0:min_starty] = 0

        grid_pd_int = grid_pd_int.fillna(0)

    #resize grid
    grid_pd_int_resized = cv2.resize(np.array(grid_pd_int),(128,128)).astype(np.int32)
    return grid_pd_int_resized

def get_record_info(look_up_table,record_ird):
    pat_id = look_up_table[look_up_table.series_id == record_id].localizer_path.values[0].split("/")[0]
    study_date = look_up_table[look_up_table.series_id == record_id].localizer_path.values[0].split("/")[1]
    return(pat_id, study_date)

params = {}
params["img_shape"] = (256, 256, 3)
params["batch_size"] = 1
params["save_path"] = "./output/"
params["save_hd_path"] = "./hd_thickness_maps"
params["save_fundus_path"] = "./fundus"

params["learning_rate"] = 0.001
look_up_path = "./full_lookup.csv"
save_model_path = os.path.join(params["save_path"],"weights.hdf5")


#get model

inputs, outputs = model_fn(params["img_shape"])
model = models.Model(inputs=[inputs], outputs=[outputs])
adam = optimizers.Adam(lr=params["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss=dice_loss, metrics=[dice_loss])
model.load_weights(save_model_path)

export_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/meta.csv"
#meta_pd = pd.read_csv(export_path)
#columns_meta = meta_pd.dicom_path.str.split(pat="/", n=-1, expand=True)
#dicom_paths = meta_pd[columns_meta[2] == "Optical Coherence Tomography Scanner"]["dicom_path"]
fundus_path = "/home/olle/PycharmProjects/thickness_map_prediction/calculation_thickness_maps/data/thickness_map_data_full_export/full_data_export_examples"
oct_path = "/home/olle/PycharmProjects/thickness_map_prediction/retinal_thickness_segmentation/evaluation/best_heidelberg_alignements"
study_paths = [os.path.join(oct_path, sub_dir) for sub_dir in os.listdir(oct_path)]

dicom_paths = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(oct_path)
    for f in fnmatch.filter(files, '*.dcm')]

def get_record_id(record_lookup):
    return record_lookup.patient_id[0]+"_"+record_lookup.laterality[0]+"_"+\
           record_lookup.study_date[0]+"_"+ str(record_lookup.series_numbers[0])

def get_fundus_dicom_dir(fundus_path, record_id):
    record_meta = record_id.split("_")
    return os.path.join(fundus_path, record_meta[0],record_meta[2])

def get_save_fundus(fundus_path, record_id):
    fdc = read_file(fundus_path)
    fundus = fdc.pixel_array
    fundus_resized = cv2.resize(fundus,(128,128))
    cv2.imwrite(os.path.join(params["save_fundus_path"],str(record_id)+".png") ,fundus_resized)


def check_fundus_dir(record_id, fundus_path):
    record_meta = record_id.split("_")
    # remove final 1 that signifies the OCT modality
    series_id = str(int(np.floor(int(record_meta[-1]) / 100) * 100))
    fundus_dicom_dir = get_fundus_dicom_dir(fundus_path, record_id)
    try:
        fundus_dicom_paths = [os.path.join(fundus_dicom_dir, sub_dir) for sub_dir in os.listdir(fundus_dicom_dir)]

        for fd in fundus_dicom_paths:
            fdc = read_file(fd)
            if str(fdc.SeriesNumber) == str(series_id):
                return("pass", fd)
    except:
        logging.info('fundus dicom paths does not exist for record: {}'.format(record_id))
        return(None, None)
    return(None,None)

logging.basicConfig(filename='failed_records.log', level=logging.INFO)

def save_octs(oct_images,dicom_path,name_suffix):
    for i in range(oct_images.shape[0]):
        oct = oct_images[i]
        save_path = os.path.join("/".join(dicom_path.split("/")[0:-1]),str(i)+name_suffix+".png")
        cv2.imwrite(save_path,oct)

ids_ =  ["357104"]
for i in dicom_paths:
    #try:
    record_lookup = get_dicom_look_up(i)
    if record_lookup is None:
        logging.info('dicom: {}'.format(i.split("/")[-1]))
        continue
    record_id = get_record_id(record_lookup)
    patient_id = record_lookup.patient_id.iloc[0]
    print(patient_id)
    #if patient_id == ids_ :
    #fundus_exist, fundus_dcm_path = check_fundus_dir(record_id, fundus_path)
    #if fundus_exist == "pass":
    #get all oct images for record
    oct_images, oct_segmentations = get_oct_and_segmentation(record_lookup)
    #get interpolated thickness grid filled with depth vectors
    grid = get_depth_grid(record_lookup, oct_segmentations)
    save_octs(oct_images,i,"")
    save_octs(oct_segmentations, i,"_seg")
    print("The file being printed is:{}".format(record_id))
    #np.save(os.path.join(params["save_hd_path"],str(record_id)+".npy"), grid)
    #cv2.imwrite(os.path.join("/".join(i.split("/")[0:-1]), "thickness_map.png"), grid)
    #get_save_fundus(fundus_dcm_path, record_id)

    #except:
    #    logging.info('File not working, logging record: {}'.format(i.split("/")[-1]))

