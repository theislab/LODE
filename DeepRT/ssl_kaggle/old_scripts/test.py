import pandas as pd
import os
import cv2
fundus_path = "/media/olle/Seagate/kaggle/train"
save_path = "/media/olle/Seagate/kaggle/train_256"

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

fundus_images = [i for i in os.listdir(fundus_path)]
fundus_images_written = [i for i in os.listdir(save_path)]

#get remaining images to write
fundus_images_towrite = [x for x in fundus_images if x not in fundus_images_written]

labels = pd.read_csv("/media/olle/Seagate/kaggle/trainLabels.csv")

for i in fundus_images_towrite:

    im = cv2.imread(os.path.join(fundus_path,i))

    im_128 = cv2.resize(im, (256,256))

    cv2.imwrite(os.path.join(save_path, i),im_128)