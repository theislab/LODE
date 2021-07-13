import os
from pydicom import read_file
import glob
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import numpy as np


class MetaLogging:
    def __init__(self):
        pass

    def process(self, i):
        try:
            dc = read_file(i, stop_before_pixels = True)

            try:
                dicom_log = {"patient_id": dc.PatientID,
                             "patient_name": dc.PatientName,
                             "laterality": dc.ImageLaterality,
                             "study_date": dc.StudyDate,
                             "birthdate": dc.PatientBirthDate,
                             # "series_description": dc.SeriesDescription,
                             # "study_description": dc.StudyDescription,
                             # "image_type": dc.ImageType,
                             "modality": dc.Modality}
            except:
                print("No laterality info available")
                dicom_log = {"patient_id": dc.PatientID,
                             "patient_name": dc.PatientName,
                             "laterality": None,
                             "study_date": dc.StudyDate,
                             "birthdate": dc.PatientBirthDate,
                             # "series_description": dc.SeriesDescription,
                             # "study_description": dc.StudyDescription,
                             # "image_type": dc.ImageType,
                             "modality": dc.Modality}

            return dicom_log
        except:
            self.write_log(record=i)
            print(f"record not working: {i}")

    def write_log(self, record):
        if not os.path.exists("./logs"):
            os.makedirs("./logs")

        if not os.path.exists('logs/meta_error_log.txt'):
            with open('logs/meta_error_log.txt', 'w') as f:
                f.write("%s\n" % record)
        else:
            with open('logs/meta_error_log.txt', 'a') as f:
                f.write("%s\n" % record)


if __name__ == "__main__":
    BASE_DIR = "/media/olle/Seagate Expansion Drive"
    studies_dir = "ForumBulkExport"

    meta_logging = MetaLogging()
    dicom_files = glob.glob(os.path.join(BASE_DIR, studies_dir + "/*/*"))

    start_time = time.time()

    print("number of OCT dicom files to process are: ", len(dicom_files))

    num_cores = 4
    processed_list = Parallel(n_jobs = num_cores)(delayed(meta_logging.process)(i) for i in tqdm(dicom_files))
    processed_list = [pl for pl in processed_list if pl is not None]

    export_pd = pd.DataFrame.from_dict(processed_list)

    export_pd.to_csv(os.path.join(BASE_DIR, "temp_tables", "export.csv"), sep = ",")

    print("Different modalities loaded: ", np.unique(export_pd.modality.tolist()))
    print(f"Processed {len(dicom_files)} files in {time.time() - start_time} time")
