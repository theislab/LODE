import os
import glob
import pandas as pd
import random
from pathlib import Path
import sys

path = Path(os.getcwd())
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))
sys.path.append(str(path.parent.parent.parent))
sys.path.append("/mnt/home/icb/olle.holmberg/projects/LODE/feature_segmentation")

from segmentation_config import WORK_SPACE, EMBEDD_SAVE_PATH


def get_annotated_patients(annotated_file):
    embeddings = pd.read_csv(os.path.join(WORK_SPACE, f"active_learning/{annotated_file}"),
                    ).dropna()["0"].tolist()

    # extract patient ids from first column
    ids = list(map(lambda x: str(x).split("_")[0], embeddings))

    # return only valid numerical patient ids
    return list(filter(lambda x: x.isdigit(), ids))


def get_unannotated_records(annotated_file):
    """
    @return: file paths of unannotated files
    @rtype: list
    """
    annotated_patients = get_annotated_patients(annotated_file)
    print("list available embeddings")
    unannotated_ids = os.listdir(EMBEDD_SAVE_PATH)
    print("available embeddings listed")

    uap_pd = EMBEDD_SAVE_PATH + "/" + pd.DataFrame(unannotated_ids)

    # rename columns
    uap_pd = uap_pd.rename(columns = {0: "embedding_path"})
    record_ids = uap_pd.embedding_path.str.split("/", expand = True).iloc[:, -1]

    # extract patients
    patients = record_ids.str.split("_", expand = True).iloc[:, 0]
    study_date = record_ids.str.split("_", expand = True).iloc[:, 2]
    laterality = record_ids.str.split("_", expand = True).iloc[:, 1]

    study_date = study_date.str.replace(".npy", "")

    uap_pd["patient_id"] = patients
    uap_pd["study_date"] = study_date
    uap_pd["laterality"] = laterality

    # filter already selected
    ap_pd = uap_pd[patients.isin(annotated_patients)]
    uap_filtered_pd = uap_pd[~patients.isin(annotated_patients)]
    return uap_filtered_pd, ap_pd