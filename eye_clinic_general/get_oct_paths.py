import os
import glob
import pandas as pd


oct_path = "/storage/groups/ml01/datasets/projects/20181610_eyeclinic_niklas.koehler/oct_images"

oct_paths = glob.glob(os.path.join(oct_path,"*"))

pd.DataFrame(oct_paths).to_csv("oct_paths.csv")
