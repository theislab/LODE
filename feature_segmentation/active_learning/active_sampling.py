import argparse
import os
import pandas as pd
from pathlib import Path
import sys
from pprint import pprint


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
└── feature_segmentation/active_learning
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
    parser.add_argument("budget", help = "number of images to select",
                        type = int, default = 100)
    parser.add_argument("chunk_size", help = "which of the different sub .csv files to read from ",
                        type = int, default = 100)
    
    parser.add_argument("number_to_search", help="number of images to search", type =int, default=10000)
    parser.add_argument("sampling_rate", help = "which of the different sub .csv files to read from", type = int, default = 49)
    parser.add_argument("name", help = "name of iteration to save results under", type = str,
                        default = "test")

    args = parser.parse_args()

    path = Path(os.getcwd())
    sys.path.append(str(path.parent))

    # add children paths
    for child_dir in [p for p in path.glob("**/*") if p.is_dir()]:
        sys.path.append(str(child_dir))

    from file_manager import FileManager
    from filter import Filter
    from embeddings import OCTEmbeddings
    from selection import Select
    from utils import move_selected_octs
    from config import WORK_SPACE, EMBEDD_DIR

    file_manager = FileManager("annotated_files.csv")

    # get record paths
    unannotated_pd, annotated_pd = file_manager.unannotated_records(use_cache = False)

    assert args.number_to_search <= unannotated_pd.shape[0], "searching more images than exist filtered"
    unannotated_pd = unannotated_pd.sample(args.number_to_search)

    filter = Filter(file_manager.feature_table_paths, unannotated_pd)

    features_table = filter.selection_table()

    keys = ["patient_id", "laterality", "study_date"]
    features_table_pd = pd.merge(unannotated_pd, features_table, left_on = keys, right_on = keys, how = "left")
    features_ffiltered_pd = filter.filter_paths(features_table_pd)

    pprint(features_table.head(5))
    pprint(features_ffiltered_pd.head(5))

    print("number of unfiltered samples are:", features_table.shape)
    print("number of filtered samples are:", features_ffiltered_pd.shape)

    assert sum(unannotated_pd.patient_id.isin(annotated_pd.patient_id.values)) == 0, "patient overlap"
    assert sum(features_ffiltered_pd["13"] < 50) == 0, "all record contains feature oi"
    assert features_table is not None, "returning None"
    assert features_ffiltered_pd is not None, "returning None"
    assert (features_ffiltered_pd.embedding_path.drop_duplicates().shape[0] // args.chunk_size) > 5 and not (
            args.chunk_size > 1), "chunk size to large"

    embedding = OCTEmbeddings()

    # embedding
    ua_embeddings = embedding.reduce_dim_unannotated(features_ffiltered_pd, chunk = args.chunk_size)

    assert embedding.reduce_dim_unannotated(pd.DataFrame(columns = features_ffiltered_pd.columns.values.tolist()),
                                            chunk = args.chunk_size).size == 0, "function does not handle empty DF"

    assert ua_embeddings.shape[0] == features_ffiltered_pd.shape[0], "not all filtered oct were embedded"

    selection = Select(args.budget)
    [ind_to_label, min_dist] = selection.select_batch(ua_embeddings)

    selected_scans = ua_embeddings.iloc[ind_to_label]

    print("format csv")
    selected_scans_pd = selected_scans.id.str.split("_", expand = True).rename(
        columns = {0: "patient_id", 1: "study_date", 2: "laterality", 3: "frame"})

    # assign id
    selected_scans_pd["id"] = selected_scans["id"]

    # add dicom name
    selected_scans_pd = pd.merge(selected_scans_pd, features_ffiltered_pd[["dicom", "id"]],
                                 how = "left", left_on = "id", right_on = "id")

    print("records to select for annotations are: ", selected_scans)
    DST_DIR = os.path.join(WORK_SPACE, "active_learning", f"selected_{args.name}")

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    assert selected_scans_pd.patient_id.drop_duplicates().shape[0] == selected_scans_pd.shape[0], \
        "patients selected are not unique"

    selected_path = os.path.join(DST_DIR, f"records_selected_{args.name}.csv")
    selected_scans_pd.to_csv(selected_path)

    print("move the selected volumes to selected dir")
    move_selected_octs(selected_scans_pd, DST_DIR)
