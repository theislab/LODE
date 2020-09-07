import argparse
import os
import time
import random
import pandas as pd
from segmentation_config import WORK_SPACE
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
This module uses the the output of segment.py, i.e. the embeddings and feature statistics to sample more images of
features of interest (foi). The foi are currently set manully in the active_learning/utils.py (to be upgraded).

In short, the program uses the core set approach to maximize the information in the new samples while focusing
on foi as well as only sampling from different patients.

The input of this program relies on the output of segment.py in the workspace/segmentation directory as well as
annotated_files.csv (see above) containing the records annotated so far. 

the output is saved in DST_DIR set in active_learning/config.py. It saved the selected scan as well as as the whole 
OCT volume for reference. 
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("name", help = "which model version to use for segmenting",
                        type = str, default = "test")
    parser.add_argument("budget", help = "which of the different sub .csv files to read from ",
                        type = int, default = 200)
    parser.add_argument("chunk_size", help = "which of the different sub .csv files to read from ",
                        type = int, default = 100)
    parser.add_argument("sampling_rate", help = "which of the different sub .csv files to read from", type = int, default = 49)
    parser.add_argument("number_to_search", help = "number of unannotated scans to search",
            type = int, default=1000)
    args = parser.parse_args()

    file_manager = FileManager("annotated_files.csv")
    
    assert_error = f"chunksize: {args.chunk_size} need to be 100 smaller than number_to_search: {args.number_to_search}. Deacrease chunk size or increase number to search argument."
    assert args.chunk_size <= (args.number_to_search // 100), assert_error

    # get record paths
    unannotated_embeddings_paths = file_manager.unannotated_records
    annotated_embeddings_paths = file_manager.get_annotated_embedding_paths()
    unannotated_embeddings_paths = random.sample(unannotated_embeddings_paths, args.number_to_search)
    

    print("number of embedded volumes", len(unannotated_embeddings_paths))
    print("number of annotated embedded volumes", len(annotated_embeddings_paths))

    selection = Select(args.budget,
                       ft_path = file_manager.feature_table_paths,
                       ae_paths = annotated_embeddings_paths,
                       uae_paths = unannotated_embeddings_paths)

    start_filtering = time.time()
    features_filtered_pd = selection.filter_paths(selection.uae_paths, args.sampling_rate, filter_ = True)

    features_filtered_pd.to_csv(os.path.join(WORK_SPACE, "active_learning", "features_filtered_pd.csv"))

    print("number of filtered octs", features_filtered_pd.shape[0])

    start_embedding = time.time()
    unannotated_embeddings = selection.reduce_dim_unannotated(features_filtered_pd, chunk = args.chunk_size)
    print(f"embedding takes {time.time() - start_embedding}")

    start_selection = time.time()
    [ind_to_label, min_dist] = selection.select_batch(unannotated_embeddings, budget = args.budget)
    selected_scans = unannotated_embeddings.iloc[ind_to_label]
    print(f"selection takes {time.time() - start_selection}")

    print("format csv")
    selected_scans_pd = selected_scans.id.str.split("_", expand = True).rename(
        columns = {0: "patient_id", 1: "study_date", 2: "laterality", 3: "frame"})


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
