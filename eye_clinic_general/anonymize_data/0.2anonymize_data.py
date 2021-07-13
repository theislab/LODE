import os
from pydicom import read_file
import glob
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import numpy as np
from copy import deepcopy

'''
join the new patient ids with old real - pseudo mapping to match allready existing patient. create new 
pseudo ids for the rest.
'''

if __name__ == "__main__":
    BASE_DIR = "/media/olle/Seagate Expansion Drive"

    new_patients = pd.read_csv(os.path.join(BASE_DIR, "temp_tables", "export.csv"))

    # drop any na patient ids
    new_patients = new_patients.dropna(subset = ["patient_id"])
    new_patients["patient_id"] = new_patients["patient_id"].astype(int)

    old_mapping = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "pseudo_id_key.csv"), sep = ";")

    pseudo_id_mapping = pd.merge(new_patients, old_mapping, left_on = "patient_id", right_on = "PATNR",
                                 how = "left")

    pseudo_id_mapping = pseudo_id_mapping.fillna(-1)

    # get number of new patients
    unique_patient_list = deepcopy(pseudo_id_mapping.drop_duplicates(subset = "patient_id"))
    new_patients_bool = unique_patient_list.pseudo_id == -1
    previous_patients_bool = unique_patient_list.pseudo_id != -1

    prev_patients = unique_patient_list.loc[previous_patients_bool][["patient_id", "pseudo_id"]]
    new_patients = unique_patient_list.loc[new_patients_bool][["patient_id"]]

    n_new_patients = sum(new_patients_bool)

    print("Number of new patients are: ", n_new_patients)
    print("Number of old patients are: ", sum(previous_patients_bool))

    last_pseudo_id = max(old_mapping.pseudo_id.dropna().tolist())

    new_pseudo_ids = np.arange(last_pseudo_id + 1, last_pseudo_id + n_new_patients + 1)
    new_patients["pseudo_id"] = new_pseudo_ids

    # append new and previous patients
    all_patients = prev_patients.append(new_patients)

    assert all_patients.shape[0] == unique_patient_list.shape[
        0], "The new and prev patients added do not sum up too all " \
            "patients "

    pseudo_id_mapping = pseudo_id_mapping.drop(["pseudo_id"], axis = 1)
    pseudo_id_mapping = pd.merge(pseudo_id_mapping, all_patients, left_on = "patient_id", right_on = "patient_id",
                                 how = "inner")

    # pseudo_id_mapping = pseudo_id_mapping.drop(["Unnamed: 0"], axis=1)
    pseudo_id_mapping.to_csv(os.path.join(BASE_DIR, "temp_tables", "pseudo_real_id_mapping.csv"), sep = ",")
