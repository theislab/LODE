from keras import Input
import os
import numpy as np
import cv2
import glob
from DeepRT.thickness_map_calculation.model import get_unet
from DeepRT.thickness_map_calculation.segmentations import Segmentations
from DeepRT.thickness_map_calculation.dicom_table import DicomTable
from DeepRT.thickness_map_calculation.map import Map
import matplotlib.pyplot as plt

# retrieve full paths to dicom files
example_dir = "./data"
dicom_paths = glob.glob(example_dir+"/*/*/*/*.dcm")

params = {}
# set path where segmenation algorithm is saved
params["save_path"] = "./output/"
save_model_path = os.path.join(params["save_path"],"weights.hdf5")

# set path where maps should be saved
params["save_map_path"] = "./thickness_maps"

# oct image dimensions, here set
params["img_shape"] = (256, 256, 3)

# load model config and weights
input_img = Input( params["img_shape"], name = 'img' )
model = get_unet(input_img, n_filters = 12)
model.load_weights(save_model_path, by_name = True, skip_mismatch=True)

# iterate through dicom files and generate map for each
for dicom_path in dicom_paths:

    # get full dicom information
    dicom = DicomTable(dicom_path)

    # if arguments in dicom are faulty, None is returned
    if dicom.record_lookup is not None:

        # retrieve all segmentations
        segmentations = Segmentations(dicom, model)

        # set path in dicom dir to save octs
        dicom_dir = "/".join(dicom_path.split( "/" )[:-1])

        # save all segmentations in ./dicom_dir/segmentations/
        segmentations.save_segmentations(os.path.join(dicom_dir, dicom.record_id+"_segmentations"))

        # save octs in ./dicom_dir/octs/
        segmentations.save_octs(save_path = os.path.join(dicom_dir, dicom.record_id+"_octs"))

        # calculate the retinal thickenss map
        map_ = Map(dicom, segmentations.oct_segmentations, dicom_path)

        # initialize calculation of thickness map
        map_.depth_grid()

        # plot thickness map
        map_.plot_thickness_map(os.path.join(dicom_dir, dicom.record_id+"_thickness_map.png"))

        # save thickness map
        np.save(os.path.join(params["save_map_path"],
                             dicom.record_id + ".npy"),
                             cv2.resize(map_.thickness_map, (128, 128)))
