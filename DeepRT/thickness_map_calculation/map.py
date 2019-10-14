from keras import Input
import os
import numpy as np
from model import get_unet
from segmentations import Segmentations
from dicom_table import DicomTable
import glob
from thickness_map import Map
import matplotlib.pyplot as plt

example_dir = "./data/oct"
dicom_paths = glob.glob( example_dir + "/*/*/*.dcm" )

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

        # calculate the retinal thickenss map
        grid = Map(dicom, segmentations.oct_segmentations)

        # save thickness map
        np.save(os.path.join(params["save_map_path"],dicom.record_id + ".npy"), grid)
