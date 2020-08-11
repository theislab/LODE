import os
from config import PROJ_DIR, DATA_DIR
from utils.etdrs_utils import ETDRSUtils
import pandas as pd
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

num_cores = 60 
inputs = tqdm(os.listdir(DATA_DIR))
# inputs = ["61619_20170110_L_512575001.npy"]*100
print(f"number of cores {num_cores} set to paralell process")


def process(i):
    #try:
    etdrs = ETDRSUtils(path = os.path.join(DATA_DIR, i))
    feature_log = etdrs.get_etdrs_stats()
    return feature_log
    #except:
    #    return 
    #    print("record not working, skippint record: ", i)


if __name__ == "__main__":
    processed_list = Parallel(n_jobs = num_cores)(delayed(process)(i) for i in inputs)    
    features_pd = pd.DataFrame(processed_list)
    features_pd.to_csv("./statistics/feature_statistics.csv")

'''
if __name__ == "__main__":
    filename = "3828_20160502_L_461980001.npy"
    for i, filename in tqdm(enumerate(os.listdir(DATA_DIR))):
        try:
            etdrs = ETDRSUtils(path=os.path.join(DATA_DIR, filename))
            feature_log = etdrs.get_etdrs_stats()

            if feature_pd is None:
                feature_pd = pd.DataFrame(feature_log, index = [i])
            else:
                feature_pd.append(pd.DataFrame(feature_log, index = [i]))
        except:
            print("record not working, skippint record: ", filename)
            continue
'''
