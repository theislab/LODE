import argparse
import os
import pandas as pd
from pathlib import Path
import sys
import glob
from pprint import pprint

path = Path(os.getcwd())
sys.path.append(str(path.parent))

# add children paths
for child_dir in [p for p in path.glob("**/*") if p.is_dir()]:
    sys.path.append(str(child_dir))

from embeddings_utils import reduce_dim_unannotated
from file_utils import get_unannotated_records
from filter_utils import get_feature_table, apply_feature_filter, set_id_columns
from selection_utils import select_batch

from utils import move_selected_octs
from segmentation_config import WORK_SPACE
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

    #args = parser.parse_args()

    class Args():
        budget = 200
        chunk_size = 10
        number_to_search = 10000
        sampling_rate = 49
        name = "srhm_drusen_20201102"

    args = Args()

    feature_table_paths = glob.glob(os.path.join(WORK_SPACE, "segmentation/feature_tables/*.csv"))
    unannotated_pd, annotated_pd = get_unannotated_records("annotated_files.csv")
    
    print("loading feature table paths: ", feature_table_paths)
    assert args.number_to_search <= unannotated_pd.shape[0], "searching more images than exist filtered"
    # unannotated_pd = unannotated_pd.sample(args.number_to_search)
    
    features_table = get_feature_table(feature_table_paths)
    features_table = set_id_columns(features_table)

    keys = ["patient_id", "laterality", "study_date"]

    # get features for unannotated samples
    features_table_pd = pd.merge(unannotated_pd, features_table, left_on = keys, right_on = keys, how = "left")

    # get records for un features of interest
    features_filtered_pd = apply_feature_filter(features_table_pd)
    
    feature_filtered_pd = features_filtered_pd.sample(args.number_to_search)
    pprint(features_table.head(5))
    pprint(features_table_pd.head(5))
    pprint(features_filtered_pd.head(5))

    print("number of unfiltered samples are:", features_table.shape)
    print("number of filtered samples are:", features_filtered_pd.shape)

    assert sum(unannotated_pd.patient_id.isin(annotated_pd.patient_id.values)) == 0, "patient overlap"
    assert features_table is not None, "returning None"
    assert features_filtered_pd is not None, "returning None"

    # embedding
    ua_embeddings = reduce_dim_unannotated(features_filtered_pd, chunk = args.chunk_size)

    assert reduce_dim_unannotated(pd.DataFrame(columns = features_filtered_pd.columns.values.tolist()),
                                            chunk = args.chunk_size).size == 0, "function does not handle empty DF"

    #assert ua_embeddings.shape[0] == features_filtered_pd.shape[0], "not all filtered oct were embedded"

    [ind_to_label, min_dist] = select_batch(ua_embeddings, args.budget)

    selected_scans = ua_embeddings.iloc[ind_to_label]

    print("format csv")
    selected_scans_pd = selected_scans.id.str.split("_", expand = True).rename(
            columns = {0: "patient_id", 1: "laterality", 2: "study_date", 3: "frame"})

    # assign id
    selected_scans_pd["id"] = selected_scans["id"]
    
    print("#"*30)
    print(selected_scans_pd.head(30))

    # add dicom name
    selected_scans_pd = pd.merge(selected_scans_pd, features_filtered_pd[["dicom_path", "id"]],
                                 how = "left", left_on = "id", right_on = "id")
    
    print(selected_scans_pd.head(30))
    print("records to select for annotations are: ", selected_scans)
    DST_DIR = os.path.join(WORK_SPACE, "active_learning", f"selected_{args.name}")

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    selected_path = os.path.join(DST_DIR, f"records_selected_{args.name}.csv")
    selected_scans_pd.to_csv(selected_path)

    print("move the selected volumes to selected dir")
    move_selected_octs(selected_scans_pd, DST_DIR)
