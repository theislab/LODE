from keras import Input
import os
import numpy as np
import cv2
from model import get_unet
from segmentations import Segmentations
from dicom_table import DicomTable
import glob
from thickness_map import Map
import matplotlib.pyplot as plt

example_dir = "/media/olle/Seagate/thickness_map_prediction/calculation_thickness_maps/data/thickness_map_examples"
dicom_paths = glob.glob(example_dir + "/*/*.dcm")

params = {}
params["save_path"] = "./output/"
params["save_map_path"] = "./thickness_maps"
params["img_shape"] = (256, 256, 3)
save_model_path = os.path.join(params["save_path"],"weights.hdf5")
input_img = Input( params["img_shape"], name = 'img' )
model = get_unet(input_img, n_filters = 12)
model.load_weights(save_model_path, by_name = True, skip_mismatch=True)

for dicom_path in dicom_paths:

    # get full dicom information
    dicom = DicomTable(dicom_path)

    if dicom.record_lookup is not None:

        # retrieve all segmentations
        segmentations = Segmentations(dicom, model)

        # set path in dicom dir to save octs
        seg_save_path = "/".join(dicom_path.split( "/" )[:-1])
        segmentations.save_segmentations(os.path.join(seg_save_path, "segmentations"))
        segmentations.save_octs(os.path.join(seg_save_path, "octs"))

        # calculate the retinal thickenss map
        map_ = Map(dicom, segmentations.oct_segmentations, dicom_path)

        # initialize calculation of thickness map
        map_.depth_grid()

        # plot thickness map
        map_.plot_thickness_map(os.path.join(seg_save_path,"thickness_map.png"))

        # save thickness map
        np.save(os.path.join(params["save_map_path"],
                             dicom.record_id + ".npy"), cv2.resize(map_.thickness_map,(128, 128)))
