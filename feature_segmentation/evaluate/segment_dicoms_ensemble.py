import os
import pandas as pd
from tqdm import tqdm
import time
import sys
import glob
<<<<<<< HEAD
root_dir = "/home/icb/olle.holmberg/projects/LODE/feature_segmentation"
=======

root_dir = "./"
>>>>>>> 29bdcc6939b36ffd6a82b2da2fa1dfa988a63a2b
search_paths = [i for i in glob.glob(root_dir + "/*/*") if os.path.isdir(i)]

for sp in search_paths:
    sys.path.append(sp)

from evaluation_utils import check_enseble_test_ids, get_ensemble_dict, \
    segment_volume, get_feature_dict, SEGMENTED_CLASSES, save_volume, get_embedding_model, embedd_volume, save_embedding
from generators.generator_utils.image_processing import read_oct_from_dicom
from segmentation_config import DATA_DIR, WORK_SPACE, VOL_SAVE_PATH, EMBEDD_SAVE_PATH


if __name__ == "__main__":

    # select model to be evaluated
    ensemble_models = ["63"]
    part = 0
    file_name = f"export2_part_{part}"
    ensemble_name = "ensemble_stratified"
    ensemble_dir = os.path.join(WORK_SPACE, "segmentation/ensembles/ensemble_stratified")

    # get dictionary holding all models
    ensemble_dict = get_ensemble_dict(ensemble_models, ensemble_dir)

    # assert correct train test split across ensemble
    check_enseble_test_ids(ensemble_dict)

    # get model for AL embedding
    embedding_model = get_embedding_model(ensemble_dir + f"/{ensemble_models[0]}")

    META_COLUMNS = ['id', 'patient_id', 'laterality', 'study_date', 'frame']
    feature_statistics = pd.DataFrame(columns = SEGMENTED_CLASSES + META_COLUMNS)

    # get batch of dicoms
    file_name = "dicom_paths"
    file_dir = "Studies2_202012"
    csv_path = os.path.join(DATA_DIR, file_dir, file_name + ".csv")
    print(csv_path)

    dicom_paths = pd.read_csv(csv_path)["0"].dropna().tolist()

    number_of_dicoms = len(dicom_paths)
    dicom_paths = dicom_paths[part*number_of_dicoms//5: int(number_of_dicoms//5 + part*number_of_dicoms//5)]

    shape = (256, 256)
    save_vol = True
    save_embedd = True

    feature_save_path = os.path.join(WORK_SPACE, f"segmentation/feature_tables/feature_statistics_{file_name}_ensemble_stratified.csv")
    
    print("Number of files to process: ", len(dicom_paths))

    i = 0
    for dicom_path in tqdm(dicom_paths):
        try:
            start_read = time.time()
            oct_volume = read_oct_from_dicom(dicom_path, shape)
            
            print("Reading volume took: ", time.time() - start_read)
            if oct_volume is not None:

                start_segment = time.time()
                segmented_volume = segment_volume(oct_volume, ensemble_dict)
                print("Segmenting volume took", time.time() - start_segment)

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
