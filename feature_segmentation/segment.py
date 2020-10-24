import os
import pandas as pd
from keras.engine.saving import load_model
from segmentation_models.metrics import iou_score
import time
from keras import Model
from pydicom import read_file
import matplotlib
from tqdm import tqdm
from config import WORK_SPACE, VOL_SAVE_PATH, EMBEDD_SAVE_PATH
from segmentation.utils import EvalVolume, load_config
import argparse
matplotlib.use('Agg')


'''
Program to segment OCT volumes from dicom files.

Assumes ./segmentation/config.py file with

PROJ_DIR: project dir of project
WORK_SPACE: path to directory storing
- segmentation/path_files
VOL_SAVE_PATH: path to where to save segmented volumes
EMBEDD_SAVE_PATH: path where to save embeddings

workspace directory expected to contain:

workspace
└── segmentation
    ├── path_files - paths to all oct dicom files
    ├── model_v* - any models trained for segmentation
    ├── test_examples - examples dicom files (optional)
   
Description: 
This module uses the path files in the workspace directory to read, segment and gather feature statistics for each 
B-scan in every OCT volume available in path files *.csvs. The program assumes that each csv in path files contains 
absolute paths to dicom files with corresponding OCT images under column name "path". 

The output of the program is one data table (.csv file) with logging for every record and its belonging feature 
statistics. Beside the calculated statistics the .csv output file contains frame, laterality, study date, 
patient pseudo id and dicom file name for record identification.

The outfile file is saved in the WORK_SPACE directory under ./feature_tables  
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help = "which model version to use for segmenting",
                        type = str, default = "model_v1")
    parser.add_argument("filename", help = "which of the different sub .csv files to read from ",
                        type = str, default = "test")

    parser.add_argument("save_id", help = "which of the different sub .csv files to read from ",
                                    type = str, default = "test")
    args = parser.parse_args()

    # select model to be evaluated
    model_directory = os.path.join(WORK_SPACE, f"feature_segmentation/segmentation/{args.model}")
    params, logging, trainops = load_config(model_directory)

    file_name = args.filename
    print(os.path.join(WORK_SPACE, "segmentation/path_files", file_name + ".csv"))
    test_ids = pd.read_csv(os.path.join(WORK_SPACE, "feature_segmentation/segmentation/path_files",
                                        file_name + ".csv"))["PATH"].dropna().tolist()
    
    file_name = file_name + "_{}".format(args.save_id)
    # copy remaining ids
    remaining_ids = test_ids.copy()

    save_model_path = os.path.join(params.model_directory, "weights.hdf5")
    print(save_model_path)
    model = load_model(save_model_path, custom_objects = {'iou_score': iou_score})

    # set up inference model
    model_input = model.layers[0]
    model_output = model.layers[-1]
    model_embedding = model.layers[len(model.layers) // 2]

    # inference model
    inference_model = Model(inputs = model_input.output, outputs = [model_output.output, model_embedding.output])

    feature_statistics = pd.DataFrame(
        columns = {'id', 'patient_id', 'laterality', 'study_date', 'frame', 'dicom', '0', '1', '2', '3',
                   '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'})

    # set save paths
    feature_save_path = os.path.join(WORK_SPACE, f"feature_segmentation/segmentation/feature_tables/feature_statistics_{file_name}.csv")
    progress_save_path = os.path.join(WORK_SPACE, f"feature_segmentation/segmentation/to_be_completed_{file_name}.csv")

    image_iter = 0
    start = time.time()
    print("Number of files to segment are: ", len(test_ids))
    for i in tqdm(range(0, len(test_ids))):
        if os.path.exists(test_ids[i]):
            if read_file(test_ids[i]).pixel_array.shape == (49, 496, 512):
                record_name = test_ids[i].split("/")[-1]
                evaluation = EvalVolume(params = params,
                                        path = test_ids[i],
                                        model = inference_model,
                                        mode = "test",
                                        volume_save_path = VOL_SAVE_PATH,
                                        embedd_save_path = EMBEDD_SAVE_PATH,
                                        save_volume = False,
                                        save_embedding = True,
                                        n_scan = 1)

                # get record statistics
                stat = evaluation.feature_statistics(evaluation.selected_bscans)

                # make into data frame
                feature_statistics = feature_statistics.append(pd.DataFrame(stat), sort = True)

                # remove from list
                remaining_ids.remove(test_ids[i])

        # progress
        if i % 100 == 0:
            feature_statistics.to_csv(feature_save_path)
            pd.DataFrame(remaining_ids).to_csv(progress_save_path)
