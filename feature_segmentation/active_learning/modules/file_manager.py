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


class FileManager:
    def __init__(self, annotated_file):
        self.annotated_file = annotated_file
        self.cache_dir = os.path.join(WORK_SPACE, "active_learning/cache")

    @property
    def feature_table_paths(self):
        return glob.glob(os.path.join(WORK_SPACE, "segmentation/feature_tables/*.csv"))

    @property
    def annotated_patients(self):
        embeddings = \
        pd.read_csv(os.path.join(WORK_SPACE, f"active_learning/{self.annotated_file}"),
                    ).dropna()["0"].tolist()

        # extract patient ids from first column
        ids = list(map(lambda x: str(x).split("_")[0], embeddings))

        # return only valid numerical patient ids
        return list(filter(lambda x: x.isdigit(), ids))

    def unannotated_records(self, use_cache=False):
        """
        @param use_cache: boolean to indicate whether to use chached files or not
        @type use_cache: bool
        @return: file paths of unannotated files
        @rtype: list
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if use_cache:
            ua_paths = pd.read_csv(os.path.join(self.cache_dir, "unannotated_paths.csv"))["0"].tolist()
            return ua_paths
        else:
            print("list available embeddings")
            unannotated_ids = os.listdir(EMBEDD_SAVE_PATH)
            print("available embeddings listed")

            uap_pd = EMBEDD_SAVE_PATH + "/" + pd.DataFrame(unannotated_ids)
            
            # rename columns
            uap_pd = uap_pd.rename(columns={0: "embedding_path"})
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
            ap_pd = uap_pd[patients.isin(self.annotated_patients)]
            uap_filtered_pd = uap_pd[~patients.isin(self.annotated_patients)]
        return uap_filtered_pd, ap_pd


if __name__ == "__main__":
    file_manager = FileManager("annotated_files.csv")
    import time

    class Args():
        def __init__(self, number_to_search):
            self.number_to_search = number_to_search


    args = Args(number_to_search = 10)
    
    start_ = time.time()
    # get record paths
    unannotated_pd, annotated_pd = file_manager.unannotated_records(use_cache = False)
    unannotated_pd = unannotated_pd.sample(args.number_to_search)
    
    print("The file procesing took: ", time.time() - start_)
    print("number of embedded volumes", unannotated_pd.shape[0])
    print("number of annotated embedded volumes", annotated_pd.shape[0])

    print("#"*30)
    print(annotated_pd.head())
