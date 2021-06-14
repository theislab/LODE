from pydicom import read_file
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import cv2

review_path = "/media/olle/Seagate/LODE/workspace/feature_segmentation/segmentation/data/review_volumes"

dicom_paths = glob.glob(review_path + "/*/*.dcm")

for dc_path in dicom_paths:
    dc = read_file(dc_path)

    id_ = str(dc.PatientID).replace("ps:", "") + "_" + dc.ImageLaterality + "_" + str(dc.StudyDate)

    record_path = os.path.join(review_path, id_)

    if not os.path.exists(record_path):
        os.makedirs(record_path)
    else:
        continue

    for b_scan in range(dc.pixel_array.shape[0]):
        oct_ = dc.pixel_array[b_scan, ::]
        cv2.imwrite(os.path.join(record_path, str(b_scan) + ".png"), oct_)
