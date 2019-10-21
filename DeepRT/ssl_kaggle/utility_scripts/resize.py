import os
import pandas as pd
import shutil
import cv2
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def myfun(i):
    im = resize(i)
    #im = brightness_adjust(im)
    cv2.imwrite(i.replace("/train/","/train_512_aspect_ratio_no_preprocc/"), im)
    return

def crop_image(img, tol=7):
    # img is image data
    # tol  is tolerance
    mask = img > tol
    return img[np.ix_(mask[:,:,0].any(1), mask[:,:,0].any(0))]

def brightness_adjust(image):
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    return(image)

def resize(path):
    desired_size = 512
    im = Image.open(path)
    im = crop_image(np.array(im))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)

    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))


    return np.array(new_im)


img_files = glob.glob("/media/olle/Seagate/kaggle/train/*")
#im = myfun(img_files[0])
Parallel(n_jobs=-1, verbose=2, backend="threading")(
             map(delayed(myfun), img_files))