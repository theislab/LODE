import os
from config import DATA_DIR, PROJ_DIR, SEG_DIR, WORK_SPACE, OCT_DIR
from feature_statistics.utils.etdrs_utils import ETDRSUtils
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

num_cores = 58

print("reading input files")
inputs = os.listdir(SEG_DIR)#[0:200]

print("Number of segmentation to process is: ", len(inputs))

print(f"number of cores {num_cores} set to paralell process")
feature_pd = None

'''
if __name__ == "__main__":
    search_filename = "93552_20131118_L_300449001.npy"
    for i, filename in tqdm(enumerate(os.listdir(SEG_DIR))):
        if filename == search_filename:
            #try:
            etdrs = ETDRSUtils(path=os.path.join(SEG_DIR, filename), dicom_path = OCT_DIR)
            feature_log = etdrs.get_etdrs_stats()

            if feature_pd is None:
                feature_pd = pd.DataFrame(feature_log, index = [i])
            else:
                feature_pd.append(pd.DataFrame(feature_log, index = [i]))
            #except:
            #    print("record not working, skippint record: ", filename)
            #    continue

    feature_pd.to_csv(os.path.join(WORK_SPACE, "segmentation_statistics.csv"))
'''


def process(i):
    try:
        etdrs = ETDRSUtils(path = os.path.join(SEG_DIR, i))
        feature_log = etdrs.get_etdrs_stats()
        return feature_log
    except:
        print("record not working, skippint record: ", i)


if __name__ == "__main__":
    processed_list = Parallel(n_jobs = num_cores)(delayed(process)(i) for i in tqdm(inputs)) 
    processed_list = [l for l in processed_list if l is not None]
    print("Get the processed list", len(processed_list))
    features_pd = pd.DataFrame.from_dict(processed_list)
    
    save_path = os.path.join(DATA_DIR, "segmentation")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    features_pd.to_csv(os.path.join(save_path, "segmentation_statistics.csv"))
