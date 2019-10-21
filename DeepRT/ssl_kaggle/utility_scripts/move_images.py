import os
import pandas as pd
import shutil
import cv2
from joblib import Parallel, delayed

save_dir = "/media/olle/Seagate/kaggle/keras_format/pre_proc/512_25/validation"

#data dir
gen_data = "/media/olle/Seagate/kaggle/train_512_aspect_ratio"

test_ids = pd.read_csv("/media/olle/Seagate/kaggle/test_zips/retinopathy_solution.csv")
test_data = "/media/olle/Seagate/kaggle/test_zips/test"

test_public = test_ids[test_ids['Usage'] == "Public"]
#labels_validaiton = pd.read_csv(os.path.join(validation_csv,"trainLabels.csv"))

validation_csv = "/media/olle/Seagate/kaggle/id_files"
csv_path = os.path.join("/media/olle/Seagate/kaggle/id_files/unbalanced/twenty_five/validation.csv")



ids = pd.read_csv(csv_path)

ids_to_move = ids["image"].values.tolist()

def myfunc(im):
    label = ids["level"][ids["image"] == im].values[0]
    shutil.copy(src=os.path.join(gen_data, im + ".jpeg"), dst=os.path.join(save_dir, str(label)))
    return

Parallel(n_jobs=-1, verbose=2, backend="threading")(
             map(delayed(myfunc), ids["image"].values.tolist()))

'''
for im in ids["image"].values:
    try:
        label = labels_validaiton["level"][labels_validaiton["image"]== im].values[0]
        shutil.copy(src=os.path.join(gen_data,im+".jpeg"),dst=os.path.join(keras_d,str(label)))
    except:
        print(im)
'''