import os
import glob
import pandas as pd
import random
from pathlib import Path
import sys

path = Path(os.getcwd())
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

from config import WORK_SPACE, EMBEDD_DIR


class FileManager:
    def __init__(self, annotated_file):
        self.annotated_file = annotated_file
        self.cache_dir = os.path.join(WORK_SPACE, "feature_segmentation/active_learning/cache")

    @property
    def feature_table_paths(self):
        return glob.glob(os.path.join(WORK_SPACE, "feature_segmentation/segmentation/feature_tables/*"))

    @property
    def annotated_patients(self):
        embeddings = \
        pd.read_csv(os.path.join(WORK_SPACE, f"feature_segmentation/active_learning/{self.annotated_file}"),
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
            unannotated_ids = os.listdir(EMBEDD_DIR)
            uap_pd = EMBEDD_DIR + "/" + pd.DataFrame(unannotated_ids)
<<<<<<< HEAD
            
            print(uap_pd.shape)
            # extract patients
            patients = uap_pd[0].str.split("/", expand = True).iloc[:, -1].str.split("_", expand = True).iloc[:, 0]
            
=======

            # rename columns
            uap_pd = uap_pd.rename(columns={0: "path"})
            record_ids = uap_pd.path.str.split("/", expand = True).iloc[:, -1]

            # extract patients
            patients = record_ids.str.split("_", expand = True).iloc[:, 0]
            study_date = record_ids.str.split("_", expand = True).iloc[:, 1]
            laterality = record_ids.str.split("_", expand = True).iloc[:, 2]

            uap_pd["patient_id"] = patients
            uap_pd["study_date"] = study_date
            uap_pd["laterality"] = laterality

>>>>>>> 225d6cbeadb57906e8fd508f3726d02a8d611d92
            # filter already selected
            ap_pd = uap_pd[patients.isin(self.annotated_patients)]
            uap_filtered_pd = uap_pd[~patients.isin(self.annotated_patients)]
        return uap_filtered_pd, ap_pd


if __name__ == "__main__":
    file_manager = FileManager("annotated_files.csv")


    class Args():
        def __init__(self, number_to_search):
            self.number_to_search = number_to_search


    args = Args(number_to_search = 10)

    # get record paths
    unannotated_pd, annotated_pd = file_manager.unannotated_records(use_cache = False)
    unannotated_pd = unannotated_pd.sample(args.number_to_search)

    print("number of embedded volumes", unannotated_pd.shape[0])
    print("number of annotated embedded volumes", annotated_pd.shape[0])
