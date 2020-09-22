import numpy as np
import pandas as pd
from pydicom import read_file
import cv2
import os
from pathlib import Path
import sys

path = Path(os.getcwd())
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

from kcenter_greedy_nalu import kCenterGreedy
from config import OCT_DIR, WORK_SPACE, EMBEDD_DIR

def to_three_channel(img):
    return np.stack((img,) * 3, axis = -1)


def move_selected_octs(selected_pd, dst_dir):
    dicom_paths = []
    for row in selected_pd.itertuples():
        if row.laterality == "L":
            laterality = "Left"
        elif row.laterality == "R":
            laterality = "Right"
        else:
            laterality = row.laterality
        
        print(row.laterality)
        dicom_file_path = os.path.join(OCT_DIR, str(row.patient_id), 
                laterality, str(row.study_date), row.dicom)
        
        print("#"*100)
        print(dicom_file_path)
        # dicom_file_path = os.path.join(OCT_DIR, row.dicom)
        # load dicom file if not empty
        dc = read_file(dicom_file_path)
        vol = dc.pixel_array
        oct_ = vol[int(row.frame), :, :]

        if len(oct_.shape) == 2:
            oct_ = to_three_channel(oct_)

        record_name = f"{row.patient_id}_{row.laterality}_{row.study_date}"
        oct_name = f"{row.patient_id}_{row.laterality}_{row.study_date}_{row.frame}"
        oct_dst_dir = os.path.join(dst_dir, record_name)
        dicom_paths.append(dicom_file_path)

        # create dir for selected record
        os.makedirs(oct_dst_dir, exist_ok = True)

        # copy selected oct
        cv2.imwrite(os.path.join(oct_dst_dir, oct_name + ".png"), oct_)

        # copy oct volume
        os.makedirs(os.path.join(oct_dst_dir, "vol"), exist_ok = True)
        for j in range(vol.shape[0]):
            oct_slice = vol[j, :, :]

            # convert to 3 channel if necessary
            if len(oct_slice.shape) == 2:
                oct_slice = to_three_channel(oct_slice)
            cv2.imwrite(os.path.join(os.path.join(oct_dst_dir, "vol"), f"{record_name}_{j}.png"), oct_slice)

    # save list to data frame
    pd.DataFrame(dicom_paths).to_csv(os.path.join(dst_dir, "dicom_paths.csv"))


class Args():
    def __init__(self, number_to_search):
        self.number_to_search = number_to_search
        self.budget = 10
        self.chunk_size = 1
        self.sampling_rate = 49


args = Args(number_to_search = 11)

if __name__ == "__main__":
    print("import works")
