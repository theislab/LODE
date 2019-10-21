#from input_data import *
##from model import *
import os
import numpy as np
import xml.etree.cElementTree as et
import pandas as pd
import xml_data_class as xdc
import depth_vector as dv
from scipy.stats import mode


def if_all_positional_args_nan(im_pd, stop_function):
    # if all position values are Nan#s, then disregard the study
    if im_pd[im_pd["Image_type"] == "OCT"][["startx_pos"]].isnull().all().values[0] == True:
        print("Startx is all empty and cannot be interpolated")
        stop_function = "yes"
    if im_pd[im_pd["Image_type"] == "OCT"][["starty_pos"]].isnull().all().values[0] == True:
        print("Starty is all empty and cannot be interpolated")
        stop_function = "yes"
    if im_pd[im_pd["Image_type"] == "OCT"][["endx_pos"]].isnull().all().values[0] == True:
        print("endx is all empty and cannot be interpolated")
        stop_function = "yes"
    if im_pd[im_pd["Image_type"] == "OCT"][["endy_pos"]].isnull().all().values[0] == True:
        print("endy is all empty and cannot be interpolated")
        stop_function = "yes"

    return stop_function


def scale_positiona_arg(OCT_pos_mm, im_pd, OCT_pos_pixel, x_scale, y_scale):
    '''scaling the positional arguments to the LOCALIZER scale argument'''
    OCT_pos_pixel["startx_pos"] = (OCT_pos_mm["startx_pos"] / x_scale).astype("int32")
    OCT_pos_pixel["endx_pos"] = (OCT_pos_mm["endx_pos"] / x_scale).astype("int32")
    OCT_pos_pixel["starty_pos"] = (OCT_pos_mm["starty_pos"] / y_scale).astype("int32")
    OCT_pos_pixel["endy_pos"] = (OCT_pos_mm["endy_pos"] / y_scale).astype("int32")
    # include lateriality
    OCT_pos_pixel["Laterality"] = im_pd[im_pd["Image_type"] == "OCT"]["Laterality"]
    OCT_pos_pixel["series_id"] = im_pd[im_pd["Image_type"] == "OCT"]["series"]
    OCT_pos_pixel["Image_aq_time"] = im_pd[im_pd["Image_type"] == "OCT"]["Image_aq_time"]
    OCT_pos_pixel["image_id"] = im_pd[im_pd["Image_type"] == "OCT"]["image_id"]
    OCT_pos_pixel["oct_height"] = im_pd[im_pd["Image_type"] == "OCT"]["oct_height"]
    OCT_pos_pixel["oct_width"] = im_pd[im_pd["Image_type"] == "OCT"]["oct_width"]
    OCT_pos_pixel["oct_scale_x"] = im_pd[im_pd["Image_type"] == "OCT"]["scaleX"]
    OCT_pos_pixel["oct_scale_y"] = im_pd[im_pd["Image_type"] == "OCT"]["scaleY"]

    # Set all values greater than 768 (grid dim) to 768
    num_pix = OCT_pos_pixel._get_numeric_data()
    # since it starts with zero any corrupt values over 768 are set to 767
    num_pix[num_pix > 768] = 767

    return (OCT_pos_pixel)

def adjust_neg_coordinates(OCT_pos_mm):
    '''
    :function: This function takes the OCT pos mm dataframe and readjusts the starting and ending coordinates
    for the X-axis. If negative coordinate are present then the negative part is subtracted from the beginning of the
    X-axis starting coordinate.
    :param OCT_pos_mm: DataFrame: contains all the coordinates for the xml files
    :return: return the DataFrame but adjusted if negative indices are present.
    '''
    # subtract the negative x_start coord values from the end coord to exclude future depth vector values
    neg_indices = OCT_pos_mm.startx_pos < 0
    # create a correctly dimensioned vector with zeros and neg values
    column_to_subtract = OCT_pos_mm["startx_pos"].copy()
    column_to_subtract[~neg_indices] = 0

    # Set negative values from corrupt files to 0
    num = OCT_pos_mm._get_numeric_data()
    num[num < 0] = 0
    #here the subtraction is made
    OCT_pos_mm["startx_pos"] = OCT_pos_mm["startx_pos"] - column_to_subtract
    #replace all Nans with zero
    OCT_pos_mm["startx_pos"] = OCT_pos_mm["startx_pos"].replace(np.nan, 0, regex=True)
    return(OCT_pos_mm)

def get_study_xml(study_path, patient):
    '''
    :program info: this function loads the xml of a study and retrieves the OCT and LOCALIZER dimensions, scales and
    positional arguments of the OCT's with respect to the LOCALIZER. The program also asserts that the xml file is not
    corrupt with regards to the LOCALIZER dim, positional arguments. The program makes the following corrections:
    1. if any positional argument is negative, it is set to 0
    2. if the pixel dimension exceeds the allowed dimensions after translation form mm to pixel, the pixel dimension
        is set to 767 (max)
    3. If the xml contains only partial positional or dimensional information, i.e. contains missing values,
        then the missing values are filles using linear interpolation
    :param study_path: Path to the study dir currently in progress
    :return: the grid with dimension as LOCALIZER, and the OCT_pos_pixel, im_pd xml data frame
    '''
    stop_function = None
    columns = ["startx_pos", "starty_pos", "endx_pos", "endy_pos"]
    OCT_pos_pixel = pd.DataFrame(columns=columns)
    grid = None

    #instatiate and tree parser and retrieve the im_pd from the xml class
    tree = et.ElementTree()
    tree.parse(study_path)
    root = tree.getroot()
    data = xdc.xml_data(root, tree)
    im_pd = data.get_image_table
    im_pd = im_pd[im_pd.Image_type.isin(["OCT","LOCALIZER"])]
    # start retrieving the marked depth grid and y, x indices
    LOC_dim = im_pd[im_pd["Image_type"] == "LOCALIZER"][["Width", "Height"]]
    LOC_scale = im_pd[im_pd["Image_type"] == "LOCALIZER"][["scaleX", "scaleY"]]
    OCT_dim = im_pd[im_pd["Image_type"] == "OCT"][["Width", "Height"]]
    OCT_scale = im_pd[im_pd["Image_type"] == "OCT"][["scaleX", "scaleY"]]

    if LOC_dim.empty == True:
        print("The xml files does not contain the Volume data for patient: {}".format(patient))
        stop_function = "yes"


    if stop_function == None:
        #get dim of OCT image
        x_dim = int(LOC_dim.values[0][1])
        y_dim = int(LOC_dim.values[0][0])

        #create grid for depth map
        grid = np.zeros([y_dim, x_dim])

        #get localizer scale
        x_scale = float(LOC_scale.values[0][0])
        y_scale = float(LOC_scale.values[0][1])

        # convert positional arguments from mm to pixels
        OCT_pos_mm = im_pd[im_pd["Image_type"] == "OCT"][["startx_pos", "starty_pos", "endx_pos"
            , "endy_pos"]].astype(float)
        # interpolate possible Nan values
        OCT_pos_mm = OCT_pos_mm.interpolate(limit_direction='both')

        OCT_pos_mm = adjust_neg_coordinates(OCT_pos_mm)

        # check positional args
        stop_function = if_all_positional_args_nan(im_pd, stop_function)

        #if stop function is still None
        if stop_function == None:
            OCT_pos_pixel = scale_positiona_arg(OCT_pos_mm, im_pd, OCT_pos_pixel, x_scale, y_scale)

    return(OCT_pos_pixel, grid,im_pd, stop_function)

def get_xml_indices(Laterality, OCT_pos_pixel, fundus_time_series, im_pd):
    '''
    :param Laterality: String, laterality of interest
    :param OCT_pos_pixel: data frame, containt all positional arguments in pixels
    :return: x_cord, y_cord, x_start, y_start, x_end, y_end, positional arguments
    '''
    # create oct image paths
    y_indices = OCT_pos_pixel.loc[(OCT_pos_pixel.Laterality == Laterality) & \
                                  (OCT_pos_pixel.series_id == fundus_time_series.values[0])]\
                                    [["Laterality", "starty_pos", "endy_pos"]]
    # set indices for later positional args
    y_indices.index = range(1, y_indices.shape[0] + 1)
    x_indices = OCT_pos_pixel.loc[(OCT_pos_pixel.Laterality == Laterality) & \
                                  (OCT_pos_pixel.series_id == fundus_time_series.values[0])]\
                                    [["Laterality", "startx_pos", "endx_pos"]]
    # set indices for later positional args
    x_indices.index = range(1, x_indices.shape[0] + 1)

    # interpolate the indices that are missing
    x_indices = x_indices.replace(0, np.nan)
    x_indices = x_indices.interpolate(limit_direction='both')
    # if all values are missing such that not interpolation has been performed, then set back to 0
    x_indices = x_indices.fillna(0)

    # test which of x or y indices is iterated over
    counts = np.unique(y_indices["starty_pos"], return_counts=True)
    if np.max(counts[1]) > 4:
        x_cord = "iterable"
        y_cord = "not_iterable"
    else:
        y_cord = "iterable"
        x_cord = "not_iterable"

    # get integer value of start and end position of OCT scan (apply mode to get the most common one)
    # filters outliers
    x_start = mode(x_indices["startx_pos"])[0][0]
    x_end = mode(x_indices["endx_pos"])[0][0]

    # get integer value of start and end position of OCT scan
    y_end = mode(y_indices["starty_pos"].values)[0][0]
    y_start = mode(y_indices["endy_pos"])[0][0]
    return(x_cord, y_cord, x_start, y_start, x_end, y_end, y_indices, x_indices)

def get_oct_lookup(Laterality, OCT_pos_pixel, fundus_time_series, im_pd):
    '''
    :param Laterality: String, laterality of interest
    :param OCT_pos_pixel: data frame, containt all positional arguments in pixels
    :return: x_cord, y_cord, x_start, y_start, x_end, y_end, positional arguments
    '''
    # create oct image paths
    y_indices = OCT_pos_pixel.loc[(OCT_pos_pixel.Laterality == Laterality) & \
                                  (OCT_pos_pixel.series_id == fundus_time_series.values[0])]\
                                    [["Laterality", "starty_pos", "endy_pos","Image_aq_time"]]
    # set indices for later positional args
    y_indices.index = range(1, y_indices.shape[0] + 1)
    x_indices = OCT_pos_pixel.loc[(OCT_pos_pixel.Laterality == Laterality) & \
                                  (OCT_pos_pixel.series_id == fundus_time_series.values[0])]\
                                    [["Laterality", "startx_pos", "endx_pos","Image_aq_time"]]
    # set indices for later positional args
    x_indices.index = range(1, x_indices.shape[0] + 1)

    # interpolate the indices that are missing
    x_indices = x_indices.replace(0, np.nan)
    x_indices = x_indices.interpolate(limit_direction='both')
    # if all values are missing such that not interpolation has been performed, then set back to 0
    x_indices = x_indices.fillna(0)

    # test which of x or y indices is iterated over
    counts = np.unique(y_indices["starty_pos"], return_counts=True)
    if np.max(counts[1]) > 4:
        x_cord = "iterable"
        y_cord = "not_iterable"
    else:
        y_cord = "iterable"
        x_cord = "not_iterable"

    # get integer value of start and end position of OCT scan (apply mode to get the most common one)
    # filters outliers
    x_start = mode(x_indices["startx_pos"])[0][0]
    x_end = mode(x_indices["endx_pos"])[0][0]

    # get integer value of start and end position of OCT scan
    y_end = mode(y_indices["starty_pos"].values)[0][0]
    y_start = mode(y_indices["endy_pos"])[0][0]
    return(x_cord, y_cord, x_start, y_start, x_end, y_end, y_indices, x_indices)

