import os
from pydicom import read_file
import glob
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import numpy as np
import shutil


if __name__ == "__main__":
    BASE_DIR = "/media/olle/Seagate Expansion Drive"
    studies_dir = "ANONYMIZED_DICOMS"
    dicom_files = pd.read_csv("./oct_dicom_files.csv")["0"].tolist()[0:2000]

    start_time = time.time()

    mapping = pd.read_csv(os.path.join(BASE_DIR, "temp_tables", "pseudo_real_id_mapping.csv"))


    ###################################################################
    patient = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "patient.csv"))
    patient = patient.drop(axis = 1, columns = ["First_name", "Last_name"])
    patient["Birth_date"] = patient["Birth_date"].str[:-2] + "01"

    patient = pd.merge(patient, mapping, left_on = "Patient_ID", right_on = "patient_id", how="left")
    patient = patient.drop(axis=1, columns = ["Patient_ID", "PATNR"])
    patient = patient.dropna(subset=["pseudo_id"])

    patient = patient.pseudo_id.astype(int)

    patient.to_csv(os.path.join(BASE_DIR, "2_lists", "patient_anonymized.csv"))

    ###################################################################
    diagnosen = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "diagnosen.csv"))

    diagnosen = pd.merge(diagnosen, mapping, left_on = "PATNR", right_on = "patient_id", how="left")
    diagnosen = diagnosen.drop(axis=1, columns = ["PATNR_y", "PATNR_x"])
    diagnosen = diagnosen.dropna(subset=["pseudo_id"])
    diagnosen = diagnosen.pseudo_id.astype(int)

    diagnosen.to_csv(os.path.join(BASE_DIR, "2_lists", "diagnosen_anonymized.csv"))

    ###################################################################
    visus = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "visus.csv"))

    visus = pd.merge(visus, mapping, left_on = "PATNR", right_on = "patient_id", how="left")
    visus = visus.drop(axis=1, columns = ["PATNR_y", "PATNR_x"])

    visus = visus.dropna(subset=["pseudo_id"])

    visus = visus.pseudo_id.astype(int)

    visus.to_csv(os.path.join(BASE_DIR, "2_lists", "visus_anonymized.csv"))

    ###################################################################
    tensio = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "tensio.csv"))

    tensio = pd.merge(tensio, mapping, left_on = "PATNR", right_on = "patient_id", how="left")
    tensio = tensio.drop(axis=1, columns = ["PATNR_y", "PATNR_x"])
    tensio = tensio.dropna(subset=["pseudo_id"])

    tensio = tensio.pseudo_id.astype(int)

    tensio.to_csv(os.path.join(BASE_DIR, "2_lists", "tensio_anonymized.csv"))






