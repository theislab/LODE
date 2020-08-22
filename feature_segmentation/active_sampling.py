import argparse
import os
import time
import random
import pandas as pd
from active_learning.config import WORK_SPACE
from active_learning.utils import Select, move_selected_octs, FileManager

'''
Program to perform active learning on a pool of unannotated OCT files.

Assumes ./active_learning/config.py file with

PROJ_DIR: project dir of project
WORK_SPACE: path to directory storing
- active_learning/path_files
OCT_DIR: path where OCT are stored
EMBEDD_DIR: path where embeddings are stored

workspace directory expected to contain:

workspace
└── active_learning
    ├── annotated_files.csv - record names of all annotated octs so far
   
Description: 
This module uses the path files in the workspace directory to read, segment and gather feature statistics for each 
B-scan in every OCT volume available in path files *.csvs. The program assumes that each csv in path files contains 
absolute paths to dicom files with corresponding OCT images under column name "path". 

The output of the program is one data table (.csv file) with logging for every record and its belonging feature 
statistics. Beside the calculated statistics the .csv output file contains frame, laterality, study date, 
patient pseudo id and dicom file name for record identification.

The outfile file is saved in the WORK_SPACE directory under ./feature_tables  
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help = "which model version to use for segmenting",
                        type = str, default = "test")
    parser.add_argument("budget", help = "which of the different sub .csv files to read from ",
                        type = int, default = 200)
    parser.add_argument("chunk_size", help = "which of the different sub .csv files to read from ",
                        type = int, default = 100)
    parser.add_argument("sampling_rate", help = "which of the different sub .csv files to read from ",
                        type = int, default = 49)
    args = parser.parse_args()

    file_manager = FileManager("annotated_files_test.csv")

    # get record paths
    unannotated_embeddings_paths = file_manager.unannotated_records
    annotated_embeddings_paths = file_manager.get_annotated_embedding_paths()
    # unannotated_embeddings_paths = random.sample(unannotated_embeddings_paths, 10000)

    print("number of embedded volumes", len(unannotated_embeddings_paths))
    print("number of annotated embedded volumes", len(annotated_embeddings_paths))

    selection = Select(args.budget,
                       ft_path = file_manager.feature_table_paths,
                       ae_paths = annotated_embeddings_paths,
                       uae_paths = unannotated_embeddings_paths)

    print("filter unannotated paths for features of interest (foi)")
    features_filtered_pd = selection.filter_paths(selection.uae_paths, args.sampling_rate, filter_ = True)
    features_unfiltered_pd = selection.filter_paths(selection.uae_paths, args.sampling_rate, filter_ = False)

    features_filtered_pd.to_csv(os.path.join(WORK_SPACE, "active_learning", "features_filtered_pd.csv"))

    print("number of filtered octs", features_filtered_pd.shape[0])

    start = time.time()
    print("finished with annotated embeddings, proceeding with unannotated embeddings")
    unannotated_embeddings = selection.reduce_dim_unannotated(features_filtered_pd, chunk = args.chunk_size)

    print("time required for {} octs were {} seconds".format(unannotated_embeddings.shape[0],
                                                             str(time.time() - start)))

    [ind_to_label, min_dist] = selection.select_batch(unannotated_embeddings, budget = args.budget)
    selected_scans = unannotated_embeddings.iloc[ind_to_label]

    print("format csv")
    selected_scans_pd = selected_scans.id.str.split("_", expand = True).rename(
        columns = {0: "patient_id", 1: "laterality", 2: "study_date", 3: "frame"})

    # assign id
    selected_scans_pd["id"] = selected_scans["id"]

    # add dicom name
    selected_scans_pd = pd.merge(selected_scans_pd, features_filtered_pd[["dicom", "id"]],
                                 how = "left", left_on = "id", right_on = "id")

    print("records to select for annotations are: ", selected_scans)
    DST_DIR = os.path.join(WORK_SPACE, "active_learning", f"selected_{args.name}")

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    selected_path = os.path.join(WORK_SPACE, "active_learning", f"records_selected_{args.name}.csv")
    selected_scans_pd.to_csv(os.path.join(DST_DIR, selected_path))

    print("move the selected volumes to selected dir")
    move_selected_octs(selected_scans_pd, DST_DIR)
