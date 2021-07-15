#!/usr/bin/env python

from __future__ import print_function
import glob
import json
import os
import os.path as osp
import cv2
import base64
import io
import numpy as np
import PIL.Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from utility_files.postprocessing import post_processing
import shutil
from utility_files.utils import shapes_to_label, iter_one_processing, fibrosis_change, set_outdir, \
    clean_data_path
from utility_files.plotting import plot_examples, create_visualizations


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


def img_data_to_arr(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def labelfiles_to_output(label_file):

    fibrosis_change_log = pd.read_csv(os.path.join(MAIN_DIR, "fibrosis_change_log.csv"),
                                      header = None).rename(columns = {0: "id", 1: "action"})

    with open(label_file) as f:
        base = label_file.split("/")[-1]

        if "12956_R_20140109" in base:
            out_img_file = osp.join(
                OUT_DIR, 'images', base + '.png')
            out_cls_file = osp.join(
                OUT_DIR, 'masks', base + '.npy')
            out_clsv_file_pre = osp.join(
                OUT_DIR, 'visualizations_pre_processing', base + '.png')
            out_clsv_file_post = osp.join(
                OUT_DIR, 'visualizations_post_processing', base + '.png')
            out_clsv_file_complete = osp.join(
                OUT_DIR, 'overview', base + '.png')
            out_json = osp.join(
                OUT_DIR, 'json')

            # copy json to output dir
            shutil.copy(label_file, os.path.join(out_json, base + ".json"))

            # load json file into data
            data = json.load(f)

            # assert image path properties
            img_path = data['imagePath']
            img_path = clean_data_path(img_path)
            data["imagePath"] = img_path

            img = img_b64_to_arr(data["imageData"])

            cls, ins = shapes_to_label(
                img_shape = img.shape,
                shapes = data['shapes'],
                label_name_to_value = CLASS_NAME_TO_ID,
                type = 'instance',
                smoothen = False
            )

            cls_preprocessing = np.copy(cls)

            # visualize before processing
            create_visualizations(out_clsv_file_pre, cls)

            # cls_smooth = postprocessing(cls_smooth)
            cls, img = post_processing(cls, img, True)

            # change according to fibrosis labeling
            fibrosis_change(base, fibrosis_change_log, cls)

            # visualize after processing
            create_visualizations(out_clsv_file_post, cls)

            ins[cls == -1] = 0  # ignore it.

            # class label
            cv2.imwrite(out_cls_file.replace(".npy", ".png").replace(".json", ""), cls)

            # save image last
            PIL.Image.fromarray(img).save(out_img_file.replace(".json", ""))

            record = [img, cls_preprocessing, cls]

            # save overview
            plot_examples(record, out_clsv_file_complete.replace(".json", ""))


OUT_DIR = "revised"
MAIN_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/data/versions/revised_iterations"
CLASS_NAME_TO_ID = set_outdir(OUT_DIR, "labels.txt")it


def main():
    files_to_process = glob.glob(os.path.join(MAIN_DIR, "jsons" + "/*"))

    # create out dir
    print("saving to: ", OUT_DIR)

    from joblib import Parallel, delayed
    Parallel(n_jobs = os.cpu_count())(delayed(labelfiles_to_output)(i) for i in tqdm(files_to_process))

    #for fp in files_to_process:
    #    labelfiles_to_output(fp)


if __name__ == '__main__':
    main()
