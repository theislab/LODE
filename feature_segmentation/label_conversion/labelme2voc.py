#!/usr/bin/env python

from __future__ import print_function
import glob
import json
import os
import os.path as osp
import cv2
import numpy as np
import PIL.Image
import pandas as pd
from utility_files.postprocessing import post_processing
from utility_files.fibrosis_record_changes import fibrosis_swap
import shutil
from utility_files.utils import shapes_to_label, iter_one_processing, fibrosis_change, set_outdir, \
    clean_data_path
from utility_files.plotting import plot_examples, create_visualizations


def labelfiles_to_output(label_files, out_dir, class_name_to_id, lq_records,
                         iteration="first_iteration", with_choroid=True, fibrosis_change_log=None):
    for label_file in label_files:
        print(label_file)

        with open(label_file) as f:
            base = label_file.split("/")[-2]

            # set iteration specific base
            if "7" in iteration:
                base = label_file.split("/")[-1].replace(".json", "").replace(".png", "")
            # if base in "58253_L_20140506":
            if base.replace(".json", "").replace(".png", "") in lq_records:
                print("remove low quality record")
                continue
            out_img_file = osp.join(
                out_dir, 'images', base + '.png')
            out_cls_file = osp.join(
                out_dir, 'masks', base + '.npy')
            out_clsv_file_pre = osp.join(
                out_dir, 'visualizations_pre_processing', base + '.png')
            out_clsv_file_post = osp.join(
                out_dir, 'visualizations_post_processing', base + '.png')
            out_clsv_file_complete = osp.join(
                out_dir, 'overview', base + '.png')
            out_json = osp.join(
                out_dir, 'json')

            # copy json to output dir
            shutil.copy(label_file, os.path.join(out_json, base + ".json"))

            # load json file into data
            data = json.load(f)

            # assert image path properties
            img_path = data['imagePath']
            img_path = clean_data_path(img_path)
            data["imagePath"] = img_path

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(PIL.Image.open(img_file))

            # swap segmentation file
            data = fibrosis_swap(base, fibrosis_change_log, data)

            cls, ins = shapes_to_label(
                img_shape = img.shape,
                shapes = data['shapes'],
                label_name_to_value = class_name_to_id,
                type = 'instance',
                smoothen = False
            )

            # if 5 in cls.flatten().tolist():

            if iteration == "first_iteration":
                cls = iter_one_processing(cls)

            if not with_choroid:
                # set choroid to background
                cls[cls == 10] = 0

            cls_preprocessing = np.copy(cls)

            # visualize before processing
            create_visualizations(out_clsv_file_pre, cls)

            # cls_smooth = postprocessing(cls_smooth)
            cls, img = post_processing(cls, img, with_choroid)

            # change according to fibrosis labeling
            fibrosis_change(base, fibrosis_change_log, cls)

            # visualize after processing
            create_visualizations(out_clsv_file_post, cls)
            # create_visualizations(out_clsv_file_smooth, cls_smooth)

            ins[cls == -1] = 0  # ignore it.
            # class label
            cv2.imwrite(out_cls_file.replace(".npy", ".png").replace(".json", ""), cls)

            # save image last
            PIL.Image.fromarray(img).save(out_img_file.replace(".json", ""))

            record = [img, cls_preprocessing, cls]

            # save overview
            plot_examples(record, out_clsv_file_complete)


def main():
    # test iteration
    PROJ_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation"
    annotatio_file = ".json"
    annotator = "ben"
    iteration = f"iteration_{annotator}"
    out_dir = iteration
    labels_file = "labels.txt"
    fibrosis_log_dir = os.path.join(PROJ_DIR, "data/versions/fibrosis_corrections")
    choroid = True

    # iteration 1
    iter_one_dir = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation/data/versions/iteration_1/json"
    iter_one_json_files = glob.glob(os.path.join(iter_one_dir, "*.json"))

    # iteration 2
    iter_two_dir = "/home/olle/PycharmProjects/feature_segmentation-master/data/versions/iteration_2"
    iter_two_json_files = glob.glob(os.path.join(PROJ_DIR, iter_two_dir, "*/*.json"))

    # iteration 3
    iter_three_dir = "data/versions/iteration_3/iteration_3_3/final_iteration"
    iter_three_json_files = glob.glob(os.path.join(PROJ_DIR, iter_three_dir + f"/*/*{annotatio_file}*"))

    # iteration 4
    iter_four_dir = f"data/versions/iteration_4/{annotator}"
    iter_four_json_files = glob.glob(os.path.join(PROJ_DIR, iter_four_dir + f"/*/*{annotatio_file}*"))

    # iteration 5
    iter_five_dir = f"data/versions/iteration_5/{annotator}"
    iter_five_json_files = glob.glob(os.path.join(PROJ_DIR, iter_five_dir + f"/*/*{annotatio_file}*"))

    # iteration 6
    iter_six_dir = f"data/versions/iteration_6/{annotator}"
    iter_six_json_files = glob.glob(os.path.join(PROJ_DIR, iter_six_dir + f"/*/*{annotatio_file}*"))

    # iteration 7
    iter_six_dir = f"data/versions/iteration_7/{annotator}"
    iter_seven_json_files = glob.glob(os.path.join(PROJ_DIR, iter_six_dir + f"/*/*{annotatio_file}*"))

    # iteration 8
    iter_eight_dir = f"data/versions/iteration_8/{annotator}"
    iter_eight_json_files = glob.glob(os.path.join(PROJ_DIR, iter_eight_dir + f"/*/*{annotatio_file}*"))

    # iteration 9
    iter_nine_dir = f"data/versions/iteration_9/{annotator}"
    iter_nine_json_files = glob.glob(os.path.join(PROJ_DIR, iter_nine_dir + f"/*/*{annotatio_file}*"))

    # test iteration
    iter_test_dir = "data/versions/test_iteration/final_iteration"
    iter_test_json_files = glob.glob(os.path.join(PROJ_DIR, iter_test_dir + f"/*/*{annotatio_file}*"))

    # ad hoc covnersions
    dir_ = "data/train_data/hq_examples_fibrosis/volumes"
    iter_adhoc_json_files = glob.glob(os.path.join(PROJ_DIR, dir_ + f"/*/*{annotatio_file}*"))

    # low quality iteration one files
    lq_records = pd.read_csv(iter_one_dir.replace("json", "loq_quality.txt"),
                             header = None)[0].str.replace(".json", "").str.replace(".png", "").tolist()

    fibrosis_change_log = pd.read_csv(os.path.join(fibrosis_log_dir, "fibrosis_change_log.csv"),
                                      header = None).rename(columns = {0: "id", 1: "action"})

    # create out dir
    class_name_to_id = set_outdir(out_dir, labels_file)

    files_to_process = iter_eight_json_files + iter_six_json_files + iter_five_json_files + iter_four_json_files + iter_seven_json_files
    labelfiles_to_output(files_to_process, out_dir, class_name_to_id, lq_records = lq_records, iteration = iteration,
                         with_choroid = choroid, fibrosis_change_log = fibrosis_change_log)


if __name__ == '__main__':
    main()
