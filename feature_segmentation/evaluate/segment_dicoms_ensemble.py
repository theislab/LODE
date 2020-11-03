import os
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import glob
import argparse
root_dir = "/home/icb/olle.holmberg/projects/LODE/feature_segmentation"
search_paths = [i for i in glob.glob(root_dir + "/*/*") if os.path.isdir(i)]

for sp in search_paths:
    sys.path.append(sp)

from feature_segmentation.evaluate.evaluation_utils import check_enseble_test_ids, get_ensemble_dict, \
    segment_volume, get_feature_dict, SEGMENTED_CLASSES, save_volume, get_embedding_model, embedd_volume, save_embedding
from feature_segmentation.generators.generator_utils.image_processing import read_oct_from_dicom
from feature_segmentation.segmentation_config import WORK_SPACE, VOL_SAVE_PATH, EMBEDD_SAVE_PATH

# select model to be evaluated
ensemble_models = ["49", "48"]

ensemble_name = "test_ensemble"
ensemble_dir = os.path.join(WORK_SPACE, ensemble_name)

# get dictionary holding all models
ensemble_dict = get_ensemble_dict(ensemble_models, ensemble_dir)

# assert correct train test split across ensemble
check_enseble_test_ids(ensemble_dict)

# get model for AL embedding
embedding_model = get_embedding_model(ensemble_dir + f"/{ensemble_models[0]}")

META_COLUMNS = ['id', 'patient_id', 'laterality', 'study_date', 'frame']
feature_statistics = pd.DataFrame(columns = META_COLUMNS + SEGMENTED_CLASSES)

# get batch of dicoms
file_name = "test"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help = "which of the different sub .csv files to read from ", 
            type = str, default = "test")
    args = parser.parse_args()

    # select model to be evaluated
    ensemble_models = ["63"]

    ensemble_name = "ensemble_stratified"
    ensemble_dir = os.path.join(WORK_SPACE, "segmentation/ensembles/ensembles_stratified")

    # get dictionary holding all models
    ensemble_dict = get_ensemble_dict(ensemble_models, ensemble_dir)

    # assert correct train test split across ensemble
    check_enseble_test_ids(ensemble_dict)

    # get model for AL embedding
    embedding_model = get_embedding_model(ensemble_dir + f"/{ensemble_models[0]}")

    META_COLUMNS = ['id', 'patient_id', 'laterality', 'study_date', 'frame']
    feature_statistics = pd.DataFrame(columns = SEGMENTED_CLASSES + META_COLUMNS)

    # get batch of dicoms
    file_name = args.filename # "paths1"

    csv_path = os.path.join(WORK_SPACE, "segmentation/path_files", file_name + ".csv")
    print(csv_path)

    dicom_paths = pd.read_csv(csv_path)["PATH"].dropna().tolist()
    shape = (256, 256)
    save_vol = True
    save_embedd = True

    feature_save_path = os.path.join(WORK_SPACE, f"segmentation/feature_tables/feature_statistics_{file_name}_ensemble_stratified.csv")
        
    i = 0
    for dicom_path in tqdm(dicom_paths):
        try:
            oct_volume = read_oct_from_dicom(dicom_path, shape)

            if oct_volume is not None:
                segmented_volume = segment_volume(oct_volume, ensemble_dict)
                feature_dict = get_feature_dict(dicom_path, segmented_volume)

                feature_dict["id"] = f"{feature_dict['patient_id']}_{feature_dict['laterality']}_{feature_dict['study_date']}"
                
                # append to data frame
                feature_statistics = feature_statistics.append(pd.DataFrame(feature_dict), sort = True)
                
                if save_vol:
                    save_volume(segmented_volume, save_path = VOL_SAVE_PATH, record_name=feature_dict["id"])

                if save_embedd:
                    embedding = embedd_volume(embedding_model, oct_volume)
                    save_embedding(embedding, save_path = EMBEDD_SAVE_PATH, record_name=feature_dict["id"])

            else:
                print("oct does not have requested shape, skipping")
                continue
        except:
            print("Dicom was not able to be processed, skipping")
        i += 1
        # progress
        if i % 100 == 0:
            feature_statistics.to_csv(feature_save_path)
