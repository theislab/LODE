import os
from pydicom import read_file
import glob
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import numpy as np
import shutil


class DicomAnonymizer:
    def __init__(self, base_dir, save_dir):
        self.base_dir = base_dir
        self.save_dir = save_dir

    def process(self, i):
        try:
            dc = read_file(i, stop_before_pixels = False)
            dc.PatientID = i.split("/")[-2].split("_")[-1]
            # if patient id is header by 00, then strip
            if len(dc.PatientID) > len(mapping.patient_id.iloc[0].astype(str)):
                dc.PatientID = dc.PatientID.lstrip("00")

            patient_mapping = mapping[mapping.patient_id.astype(str) == dc.PatientID]
            pseudo_id = patient_mapping.pseudo_id.tolist()[0]

            dicom_name = i.split("/")[-1]

            try:
                laterality = dc.ImageLaterality
            except:
                laterality = "None"

            record_dir = os.path.join(self.save_dir, dc.Modality, str(int(pseudo_id)), laterality, dc.StudyDate)

            if not os.path.exists(record_dir):
                os.makedirs(record_dir)

            # anonymize
            dc.PatientName = ""
            dc.PatientBirthDate = dc.PatientBirthDate[0:-2] + "01"
            dc.PatientID = str(int(pseudo_id))

            # save anonymized dicom
            dc.save_as(os.path.join(record_dir, dicom_name))
            return []
        except:
            self.write_log(record = i)
            print(f"record not working: {i}")

    def write_log(self, record):
        log_file_name = "dicom_anonymizing_error_log.txt"

        if not os.path.exists("./logs"):
            os.makedirs("./logs")

        if not os.path.exists(f'logs/{log_file_name}'):
            with open(f'logs/{log_file_name}', 'w') as f:
                f.write("%s\n" % record)
        else:
            with open(f'logs/{log_file_name}', 'a') as f:
                f.write("%s\n" % record)


if __name__ == "__main__":
    BASE_DIR = "/media/basani/Seagate Expansion Drive"
    studies_dir = "ANONYMIZED_DICOMS"
    dicom_anonymizer = DicomAnonymizer(base_dir = BASE_DIR, save_dir=os.path.join(BASE_DIR, studies_dir))

    dicom_files = glob.glob("/media/basani/Seagate Expansion Drive/ForumBulkExport/*/*.dcm")
    start_time = time.time()

    mapping = pd.read_csv(os.path.join(BASE_DIR, "temp_tables", "pseudo_real_id_mapping.csv"))

    num_cores = 1
    processed_list = Parallel(n_jobs = num_cores)(delayed(dicom_anonymizer.process)(i.replace("olle", "ben")) for i in tqdm(dicom_files))
