from pydicom import read_file
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
save_path = "/media/olle/Seagate/eyeclinic_octs/735/oct_R"
dicom_paths = glob.glob('/media/olle/Seagate/eyeclinic_octs/735/Right/**/*')

for dp in dicom_paths:
    dc = read_file(dp)
    octs = dc.pixel_array
    id = dc.PatientID.replace("ps:","")
    study_date = dc.StudyDate
    laterality = dc.ImageLaterality

    for iter_ in range(0,octs.shape[0]):
        save_name = id + "_" + study_date + "_" + laterality + "_" + str(iter_) + ".png"
        oct = octs[iter_,:,:]
        cv2.imwrite(os.path.join(save_path,save_name), oct)
