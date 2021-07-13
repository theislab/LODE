import numpy as np
import cv2
import os
import json

PROJ_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation"
MICHAEL_FIB_DIR = os.path.join(PROJ_DIR, "data/versions/fibrosis_corrections/michael")
BEN_FIB_DIR = os.path.join(PROJ_DIR, "data/versions/fibrosis_corrections/ben")


def map_changes(img_id, map_):
    if "14277_R_20180305" in img_id:
        map_[map_ == 7] = 13
        map_[map_ == 5] = 13

    if "57929_L_20170123" in img_id:
        map_[map_ == 8] = 7
        map_[map_ == 5] = 13

    if "239396_R_20151117" in img_id:
        map_[map_ == 5] = 6

    if "92927_R_20171127" in img_id:
        map_[map_ == 5] = 7

    if "181978_R_20160725" in img_id:
        map_[map_ == 5] = 13

    if "175612_L_20180606" in img_id:
        map_[map_ == 8] = 7
        map_[map_ == 5] = 6

    if "176808_R_20160303" in img_id:
        map_[map_ == 5] = 13
        map_[map_ == 8] = 7

    return map_
