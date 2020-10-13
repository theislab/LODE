import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy

F_BELOW_NEO_INTIMA = [6, 10, 0]


def column_all_background(col):
    col_ = deepcopy(col)
    # ignore any choroid annotation
    col_[col_ == 10] = 0
    if np.sum(col_) == 0:
        return True


def remove_all_background_columns(cls, img):
    cols_to_remove = []
    # cut all background column
    for j in range(0, cls.shape[1]):
        column = cls[:, j]
        all_background = column_all_background(column)
        if all_background:
            cols_to_remove.append(j)

    # remove above selected columns
    cls = np.delete(cls, cols_to_remove, axis = 1)
    img = np.delete(img, cols_to_remove, axis = 1)
    return cls, img


def neu_intima_feature_processing(cls):
    def get_top_feature_left(j, cls):
        minimums = []
        for i in range(5):
            minimums.append(np.min(np.nonzero(cls[:, j + i] == 6)))
        return np.min(minimums)

    def get_top_feature_right(j, cls):
        minimums = []
        for i in range(5):
            minimums.append(np.min(np.nonzero(cls[:, j - i] == 6)))
        return np.min(minimums)

    # find first column idx with 2 annotated
    for j in range(cls.shape[1]):
        column = cls[:, j]
        # is there neo-intima in column
        if np.sum(column == 2) > 0:
            # get top row from left value with Neo-intima
            for k in range(column.shape[0]):
                if column[k] == 2:
                    topLeftRowNeoInitima = k
            break

    # find last top idx with 2 annotated
    for j in range(cls.shape[1] - 1, -1, -1):
        column = cls[:, j]
        # is there neo-intima in column
        if np.sum(column == 2) > 0:
            # get top row from right value with Neo-intima
            for k in range(column.shape[0]):
                if column[k] == 2:
                    topRightRowNeoInitima = k
            break

    '''
    below fills out holes btw neuro intima and rpe layers
    with neuro intima from left
    '''
    # from left post processing
    for j in range(cls.shape[1]):
        column = cls[:, j]
        # is there RPE in column
        if np.sum(column == 6) > 0:
            # if there is neo - intima, then break
            if np.sum(column == 2) > 0:
                break
            # is there no neo intima in column
            if np.sum(column == 2) == 0:
                topRPE = get_top_feature_left(j, cls)
                for k in range(topRPE, topLeftRowNeoInitima, -1):
                    cls[k, j] = 2

    '''
    below fills out holes btw neuro intima and rpe layers
    with neuro intima from right
    '''
    # from right post processing
    for j in range(cls.shape[1] - 1, -1, -1):
        # j is column index
        column = cls[:, j]
        # is there RPE in column
        if np.sum(column == 6) > 0:
            # if there is neo-intima, then break
            if np.sum(column == 2) > 0:
                break
            # is there no neo intima in column
            if np.sum(column == 2) == 0:
                topRPE = get_top_feature_right(j, cls)
                for k in range(topRPE, topRightRowNeoInitima, -1):
                    cls[k, j] = 2
    return cls


def fill_out_vitreous(cls):
    """make all background pixels above 2 into vitreous"""
    # fill out blanks below feature
    for feature in [2]:
        # if feature is in map
        if feature in cls:
            columns = np.sum(cls == feature, axis = 0) > 0
            for column_idx in range(0, cls.shape[1]):
                if columns[column_idx]:
                    # get column of interest
                    column = cls[:, column_idx]
                    # get top idx value of feature in column
                    last = np.max(np.nonzero(column == feature))
                    for row in range(0, last):
                        if column[row] in F_BELOW_NEO_INTIMA:
                            cls[row, column_idx] = 14
    return cls


def post_processing(cls, img, choroid):
    w, h = img.shape[:2]

    cls, img = remove_all_background_columns(cls, img)

    # annotate camera artifact
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = img
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if choroid:
        # put mask into alpha channel of image
        camera_effect_color = 15
    if not choroid:
        camera_effect_color = 14

    # put mask into alpha channel of image
    cls[mask[:, :] == 0] = camera_effect_color

    if 6 in cls.tolist():
        def get_top_rpe_left(j, cls):
            minimums = []
            for i in range(5):
                minimums.append(np.min(np.nonzero(cls[:, j + i] == 6)))
            return np.min(minimums)

        def get_top_rpe_right(j, cls):
            minimums = []
            for i in range(5):
                minimums.append(np.min(np.nonzero(cls[:, j - i] == 6)))
            return np.min(minimums)

    cls = neu_intima_feature_processing(cls)
    '''
    below fills out holes above selected features such that no background labels are
    found in btw of features
    '''
    # fill out blanks above feature
    for feature in [6, 4, 5, 7, 8, 13]:
        # if feature is in map
        if feature in cls:
            columns = np.sum(cls == feature, axis = 0) > 0
            for column_idx in range(0, cls.shape[1]):
                if columns[column_idx]:
                    column = cls[:, column_idx]
                    # get top idx value of feature in column
                    first = np.min(np.nonzero(column == feature))
                    # check if there are any zeros directly above
                    if np.sum(column[:first - 1] == 0) == column[:first - 1].shape[0]:
                        break
                    # break of not neurosensory retina is above
                    if np.sum(column[:first - 1] == 2) == 0:
                        break
                    if column[first - 1] == 0:
                        for i in range(first - 1, -1, -1):
                            # if above row is not zero, stop
                            if column[i] != 0:
                                break
                            # if row is zero, set to feature
                            if column[i] == 0:
                                cls[i, column_idx] = feature

    # change vitrous/camera artifact label w.r.t. presence of choroid 
    if choroid:
        vitrous_label = 14
    if not choroid:
        vitrous_label = 13

    # annotate Vitreous
    for column_idx in range(0, cls.shape[1]):
        column = cls[:, column_idx]

        # if column has nr
        if 2 in column.tolist():
            first_nr = np.min(np.nonzero(column == 2))

            # set all pixels to 10 == Vitreous
            for i in range(0, first_nr):
                if cls[i, column_idx - 1] == 0:
                    cls[i, column_idx - 1] = vitrous_label

        # if column does not have label 2 but feature 6 to separate background from vitreous
        if 2 not in column.tolist():
            if 5 in column.tolist():
                first_nr = np.min(np.nonzero(column == 5))

                # set all pixels to 10 == Vitreous
                for i in range(0, first_nr):
                    if cls[i, column_idx - 1] == 0:
                        cls[i, column_idx - 1] = vitrous_label

    if choroid:
        choroid_color = 10
        # draw choroid to full border - from left
        for col in range(0, cls.shape[1] - 1):
            if np.sum(cls[:, col] == choroid_color) > 0:
                top_left_choroid_row = np.min(np.where(cls[:, col] == choroid_color))
                top_left_choroid_col = col

                # if a less than 5 % margin is left un annotated, fill out row as choroid
                if top_left_choroid_col > int(cls.shape[1] * 0.05):
                    for col_remainder in range(top_left_choroid_col, 0, -1):
                        cls[top_left_choroid_row, col_remainder] = choroid_color
                break

        # draw choroid to full border - from right
        for col in range(cls.shape[1] - 1, 0, -1):
            if np.sum(cls[:, col] == choroid_color) > 0:
                top_right_choroid_row = np.min(np.where(cls[:, col] == choroid_color))
                top_right_choroid_col = col

                # if a less than 5 % margin is left un annotated, fill out row as choroid
                if top_right_choroid_col > int(cls.shape[1] * 0.95):
                    for col_remainder in range(cls.shape[1] - 1, int(cls.shape[1] * 0.95), - 1):
                        if cls[top_right_choroid_row, col_remainder] == 0:
                            cls[top_right_choroid_row, col_remainder] = choroid_color
                break

    if choroid:
        # fill out choroid
        for col in range(0, cls.shape[1]):
            if np.sum(cls[:, col] == choroid_color) > 0:
                topLeftChoiroid = np.max(np.where(cls[:, col] == choroid_color))

                for row_idx in range(topLeftChoiroid, 0, -1):
                    if cls[row_idx, col] == 0:
                        cls[row_idx, col] = choroid_color
                    else:
                        continue

    '''
    below fills out wholes below neo intima, break if all 
    pixels below are background
    '''
    # fill out blanks below feature
    for feature in [2]:
        # if feature is in map
        if feature in cls:
            columns = np.sum(cls == feature, axis = 0) > 0
            for column_idx in range(0, cls.shape[1]):
                if columns[column_idx]:
                    # get column of interest
                    column = cls[:, column_idx]
                    # get top idx value of feature in column
                    last = np.max(np.nonzero(column == feature))

                    # check if no other features are below, then break
                    if np.sum(((column[last + 1:] == 0) | (column[last + 1:] == camera_effect_color))) == \
                            column[last + 1:].shape[0]:
                        break

                    # check if there are any zeros directly above
                    if column[last + 1] == 0:
                        for i in range(last + 1, column.shape[0]):
                            # if above row is not zero, stop
                            if column[i] != 0:
                                break
                            # if row is zero, set to feature
                            if column[i] == 0:
                                cls[i, column_idx] = feature

    # fill out any incorrectly assigned features values above 2 to 14
    cls = fill_out_vitreous(cls)

    # remove 5 pixels from each side as polygon annotations are inprecise there
    cls = cls[:, 5:-5]
    img = img[:, 5:-5]

    # resize images to original size
    img = cv2.resize(img, (h, w))
    cls = cv2.resize(cls, (h, w), interpolation = cv2.INTER_NEAREST)
    return cls, img
