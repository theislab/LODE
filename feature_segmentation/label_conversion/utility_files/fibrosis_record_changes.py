import numpy as np
import cv2
import os
import json

PROJ_DIR = "/home/olle/PycharmProjects/LODE/workspace/feature_segmentation/segmentation"
MICHAEL_FIB_DIR = os.path.join(PROJ_DIR, "data/versions/fibrosis_corrections/michael")
BEN_FIB_DIR = os.path.join(PROJ_DIR, "data/versions/fibrosis_corrections/ben")


def fibrosis_swap(img_id, change_log, data):
    record_bool = change_log["id"] == img_id
    if np.sum(record_bool) != 0:
        action = change_log.loc[record_bool]["action"].values[0]
        if action == "new scan":
            if img_id == "25717_R_20151013_423666001_23":
                with open(os.path.join(BEN_FIB_DIR, img_id, "23.json")) as f:
                    data = json.load(f)
            if img_id == "52732_R_20150508_394197001_11":
                with open(os.path.join(BEN_FIB_DIR, img_id, "11.json")) as f:
                    data = json.load(f)
            if img_id == "295655_L_20170822_563504001_40":
                with open(os.path.join(MICHAEL_FIB_DIR,"iteration_4", img_id, "40.json")) as f:
                    data = json.load(f)
            if img_id == "353695_R_20180525_628610001_34":
                with open(os.path.join(MICHAEL_FIB_DIR,"iteration_4", img_id, "34.json")) as f:
                    data = json.load(f)
            if img_id == "365484_L_20180917_659181001_23":
                with open(os.path.join(MICHAEL_FIB_DIR,"iteration_4", img_id, "23.json")) as f:
                    data = json.load(f)
            if img_id == "181978_R_20160725_479107001_21":
                with open(os.path.join(MICHAEL_FIB_DIR,"iteration_5", img_id, "21.json")) as f:
                    data = json.load(f)
            if img_id == "189268_L_20160426_460770001_18":
                with open(os.path.join(MICHAEL_FIB_DIR,"iteration_5", img_id, "18.json")) as f:
                    data = json.load(f)
            if img_id == "356491_R_20170720_556002001_23":
                with open(os.path.join(MICHAEL_FIB_DIR,"iteration_5", img_id, "23.json")) as f:
                    data = json.load(f)
    else:
        return data
    return data


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
