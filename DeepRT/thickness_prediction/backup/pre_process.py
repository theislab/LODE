from train_eval_ops import *
import tensorflow as tf
from tensorflow.keras.optimizers import adam
import model as mt
import os
import cv2
from train_eval_ops import *
import matplotlib.pyplot as plt


def crop_image(img,cond, tol=0):
    # img is image data
    # tol  is tolerance
    mask = cond>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

tm_dir = "/media/olle/Seagate/thickness_map_prediction/thickness_maps"
f_dir = "/media/olle/Seagate/thickness_map_prediction/fundus"

files_ = [i.replace(".npy","") for i in os.listdir(tm_dir)]

for i in range(0,1000):
    print(files_[i])
    # load samples
    im = cv2.imread(os.path.join(f_dir, files_[i]+".png"))
    lbl = np.load(os.path.join(tm_dir, files_[i]+".npy"))

    # convert to three channel
    if im.shape[-1] != 3:
        im = np.stack((im,) * 3, axis=-1)

    # crop images
    c_im = crop_image(im, lbl, tol=0)
    c_lbl = crop_image(lbl, lbl, tol=0)

    #adjust light
    c_im = cv2.addWeighted(c_im, 4, cv2.GaussianBlur(c_im, (0, 0), 10), -4, 128)

    # resize samples
    im_resized = cv2.resize(c_im, (128,128))
    lbl_resized = cv2.resize(c_lbl, (128,128))

    plt.imsave("./pre_proc/"+files_[i]+"_label.png",lbl_resized,cmap=plt.cm.jet)
    plt.imsave("./pre_proc/"+files_[i]+"_fundus.png",im_resized)
