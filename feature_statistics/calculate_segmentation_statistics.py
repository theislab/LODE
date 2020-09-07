import os
from config import PROJ_DIR, SEG_DIR, WORK_SPACE
from feature_statistics.utils.etdrs_utils import ETDRSUtils
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

num_cores = 60 
inputs = os.listdir(SEG_DIR)
# inputs = ["61619_20170110_L_512575001.npy"]*100
print(f"number of cores {num_cores} set to paralell process")
feature_pd = None

'''
if __name__ == "__main__":
    filename = "271148_20160509_L_463059001.npy"
    for i, filename in tqdm(enumerate(os.listdir(SEG_DIR))):
        try:
            etdrs = ETDRSUtils(path=os.path.join(SEG_DIR, filename))
            feature_log = etdrs.get_etdrs_stats()
    
            if feature_pd is None:
                feature_pd = pd.DataFrame(feature_log, index = [i])
            else:
                feature_pd.append(pd.DataFrame(feature_log, index = [i]))
        except:
            print("record not working, skippint record: ", filename)
            continue

'''

def process(i):
    try:
        etdrs = ETDRSUtils(path = os.path.join(SEG_DIR, i))
        feature_log = etdrs.get_etdrs_stats()
        return feature_log
    except:
        print("record not working, skippint record: ", i)


if __name__ == "__main__":
    processed_list = Parallel(n_jobs = num_cores)(delayed(process)(i) for i in inputs)
    features_pd = pd.DataFrame.from_dict(processed_list)
    features_pd.to_csv(os.path.join(WORK_SPACE, "segmentation_statistics.csv"))

