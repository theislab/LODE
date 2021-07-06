import glob
import os
import numpy as np
import shutil

import random
train_ = "/media/olle/Seagate/kaggle/tensorflow.keras_format/pre_proc/512_3/train"

classes = ["0/*","1/*","2/*","3/*","4/*"]

files_0 = glob.glob(os.path.join(train_,"0/*"))
files_1 = glob.glob(os.path.join(train_,"1/*"))
files_2 = glob.glob(os.path.join(train_,"2/*"))
files_3 = glob.glob(os.path.join(train_,"3/*"))
files_4 = glob.glob(os.path.join(train_,"4/*"))

file_list = [files_0,files_1,files_2,files_3,files_4]

to_rm = []

for fl in file_list:
    to_rm.extend(random.sample(fl,int(len(fl)*0.97)))

for f in to_rm:
    os.remove(f)