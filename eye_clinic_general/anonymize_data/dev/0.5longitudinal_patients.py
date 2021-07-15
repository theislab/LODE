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
    BASE_DIR = "./"
    studies_dir = "ANONYMIZED_DICOMS"

    start_time = time.time()

    mapping = pd.read_csv(os.path.join(BASE_DIR, "temp_tables", "pseudo_real_id_mapping.csv"))

    ###################################################################
    visus = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "visus.csv"))

    visus = pd.merge(visus, mapping[["patient_id", "pseudo_id"]], left_on="PATNR", right_on="patient_id", how="left")
    visus_anonym = visus.drop(axis=1, columns=["PATNR", "patient_id", 'DOKAR', 'DOKNR', 'LFDNR', 'DOKVR', 'DOKTL',
       'DATINDB', 'LFDNR_2'])

    visus_anonym = visus_anonym.dropna(subset=["pseudo_id"])
    visus_anonym["pseudo_id"] = visus_anonym.pseudo_id.astype(int)
    visus_anonym = visus_anonym.drop_duplicates(visus_anonym.columns)

    visus_anonym.to_csv(os.path.join(BASE_DIR, "2_lists", "visus_anonymized.csv"))

    ###################################################################
    tensio = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "tensio.csv"))

    tensio = pd.merge(tensio, mapping[["patient_id", "pseudo_id"]], left_on="PATNR", right_on="patient_id", how="left")
    tensio_anonym = tensio.drop(axis=1, columns=["PATNR", "patient_id",
                                                 'LFDNR','DOKNR', 'DATINDB', 'LFDNR_2',
                                                 'DOKVR'
                                                 ])

    tensio_anonym = tensio_anonym.dropna(subset=["pseudo_id"])
    tensio_anonym["pseudo_id"] = tensio_anonym.pseudo_id.astype(int)
    tensio_anonym = tensio_anonym.drop_duplicates(tensio_anonym.columns)

    tensio_anonym.to_csv(os.path.join(BASE_DIR, "2_lists", "tensio_anonymized.csv"))

    ###################################################################
    diagnosen = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "diagnosen.csv"))

    diagnosen = pd.merge(diagnosen, mapping[["patient_id", "pseudo_id"]], left_on="PATNR", right_on="patient_id",
                         how="left")
    diagnosen_anonym = diagnosen.drop(axis=1, columns=["PATNR", "patient_id", 'FALNR', 'LFDDIA', 'LFDBEW',
                                                       'DTEXT', 'EWDIA', 'BHDIA', 'AFDIA', 'ENDIA', 'FHDIA', 'KHDIA',
                                                       'OPDIA', 'LFDNR',
                                                       'DIAGW', 'ORGFAL', 'ORGFBW', 'ORGPBW', 'DATINDB'
                                                       ])

    diagnosen_anonym = diagnosen_anonym.dropna(subset=["pseudo_id"])
    diagnosen_anonym['pseudo_id'] = diagnosen_anonym.pseudo_id.astype(int)

    diagnosen_anonym = diagnosen_anonym.drop_duplicates(diagnosen_anonym.columns)

    diagnosen_anonym.to_csv(os.path.join(BASE_DIR, "2_lists", "diagnosen_anonymized.csv"))

    ###################################################################
    patient = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "patient.csv"))
    patient = patient.drop(axis=1, columns=["First_name", "Last_name"])
    patient["Birth_date"] = patient["Birth_date"].str[:-2] + "01"

    patient = pd.merge(patient, mapping[["patient_id", "pseudo_id"]], left_on="Patient_ID", right_on="patient_id",
                       how="left")
    patient_anonym = patient.drop(axis=1, columns=["Patient_ID", "patient_id"])
    patient_anonym = patient_anonym.dropna(subset=["pseudo_id"])

    patient_anonym["pseudo_id"] = patient_anonym.pseudo_id.astype(int)
    patient_anonym = patient_anonym.drop_duplicates(patient_anonym.columns)

    patient_anonym.to_csv(os.path.join(BASE_DIR, "2_lists", "patient_anonymized.csv"))

    ###################################################################
    prozeduren = pd.read_csv(os.path.join(BASE_DIR, "2_lists", "prozeduren.csv"))

    prozeduren = pd.merge(prozeduren, mapping[["patient_id", "pseudo_id"]], left_on="PATNR", right_on="patient_id",
                          how="left")
    prozeduren_anonym = prozeduren.drop(axis=1,
                                        columns=["PATNR", "patient_id", 'FALNR', 'LFDBEW', 'LNRIC', 'TXT', 'DATINDB',
                                                 'LFDNR'])

    prozeduren_anonym = prozeduren_anonym.dropna(subset=["pseudo_id"])
    prozeduren_anonym["pseudo_id"] = prozeduren_anonym.pseudo_id.astype(int)
    prozeduren_anonym = prozeduren_anonym.drop_duplicates(prozeduren_anonym.columns)

    prozeduren_anonym.to_csv(os.path.join(BASE_DIR, "2_lists", "prozeduren_anonymized.csv"))


