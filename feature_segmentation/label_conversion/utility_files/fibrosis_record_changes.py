import numpy as np
import cv2
import os
import json

PROJ_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation"
MICHAEL_FIB_DIR = os.path.join(PROJ_DIR, "data/versions/fibrosis_corrections/michael")
BEN_FIB_DIR = os.path.join(PROJ_DIR, "data/versions/fibrosis_corrections/ben")


def map_changes(img_id, map_):
    if img_id == "14277_R_20180305_607479001_9":
        map_[map_ == 7] = 13
        map_[map_ == 5] = 13

    if img_id == "57929_Left_20170123_515424001":
        map_[map_ == 8] = 7
        map_[map_ == 5] = 13

    if img_id == "239396_R_20151117_429945001_25":
        map_[map_ == 5] = 6

    if img_id == "92927_R_20171127_584948001_11":
        map_[map_ == 5] = 7

    if img_id == "181978_R_20160725_479107001_21":
        map_[map_ == 5] = 13

    if img_id == "175612_L_20180606_631458001_18":
        map_[map_ == 8] = 7
        map_[map_ == 5] = 6

    if img_id == "176808_R_20160303_450205001_29":
        map_[map_ == 5] = 13
        map_[map_ == 8] = 7

    return map_
