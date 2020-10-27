from __future__ import print_function

import os
import sys
from pathlib import Path
import pandas as pd
import glob


root_dir = "/home/icb/olle.holmberg/projects/LODE/feature_segmentation"
search_paths = [i for i in glob.glob(root_dir + "/*/*") if os.path.isdir(i)]

for sp in search_paths:
        sys.path.append(sp)

path_variable = Path(os.path.dirname(__file__))
sys.path.insert(0, str(path_variable))
sys.path.insert(0, str(path_variable.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec

from segmentation_config import TRAIN_DATA_PATH
from generators.generator_utils.image_processing import read_resize


def color_mappings():
    color_palett = np.array([[148., 158., 167.],
                             [11., 151., 199.],
                             [30., 122., 57.],
                             [135., 191., 234.],
                             [37., 111., 182.],
                             [156., 99., 84.],
                             [226., 148., 60.],
                             [203., 54., 68.],
                             [192., 194., 149.],
                             [105., 194., 185.],
                             [205., 205., 205.],
                             [140., 204., 177.],  # Serous PED
                             [183., 186., 219.],  # other artifact
                             [114, 137, 218],  # fibrosis
                             [209., 227., 239.],
                             [226., 233., 48.]])

    color_palett_norm = color_palett / 255  # (np.max(color_palett)-np.min(color_palett))
    custom_cmap = colors.ListedColormap(
        color_palett_norm
    )

    # set counts and norm
    array_bounds = np.arange(color_palett.shape[0] + 1) - 0.1
    bounds = array_bounds.tolist()
    norm = colors.BoundaryNorm(bounds, custom_cmap.N)
    return custom_cmap, norm, bounds


def save_segmentation_plot(out_clsv_file, cls):
    """
    :param out_clsv_file: str: save path
    :param cls: numpy array integer class map
    :return: save figure to out_clsv_file
    """
    seg_cmap, seg_norm, bounds = color_mappings()
    colorbar_im = plt.imshow(cls, cmap = seg_cmap, norm = seg_norm)

    # set colorbar ticks
    tick_loc_array = np.arange(len(bounds) - 1) + 0.5
    tick_loc_list = tick_loc_array.tolist()

    tick_list = np.arange(len(bounds) - 1).tolist()
    c_bar = plt.colorbar(colorbar_im, cmap = seg_cmap, norm = seg_norm, boundaries = bounds)
    # set ticks
    c_bar.set_ticks(tick_loc_list)
    c_bar.ax.set_yticklabels(tick_list)

    plt.savefig(out_clsv_file.replace(".npy", ".png"))
    plt.close()


def plot_data_processing(record, path):
    """
    :param record: list containing image, seg map before pre processing, after processing
    :param path: str: save path
    :return:
    """
    seg_cmap, seg_norm, bounds = color_mappings()
    fig = plt.figure(figsize = (24, 8))
    columns = 3
    rows = 1
    types = ["image", "pre processing", "post processing"]
    for i in range(1, columns * rows + 1):
        img = record[i - 1]
        fig.add_subplot(rows, columns, i)
        if types[i - 1] != "image":
            colorbar_im = plt.imshow(img, cmap = seg_cmap, norm = seg_norm)

            # set colorbar ticks
            tick_loc_array = np.arange(len(bounds)) + 0.5
            tick_loc_list = tick_loc_array.tolist()

            tick_list = np.arange(len(bounds)).tolist()
            c_bar = plt.colorbar(colorbar_im, cmap = seg_cmap, norm = seg_norm, boundaries = bounds)

            # set ticks
            c_bar.set_ticks(tick_loc_list)
            c_bar.ax.set_yticklabels(tick_list)
        if types[i - 1] == "image":
            if len(img.shape) == 2:
                img = np.stack((img,) * 3, axis = -1)
            plt.imshow(img)
        plt.title(types[i - 1])
    plt.savefig(path)
    plt.close()


def plot_image_label_prediction(records, model_dir, mode, filename):
    """
    :param records: list containing numpy array of image, label and prediction
    :param model_dir: directory of model where to save images
    :param mode: str: train or test
    :param filename: str: filename of image
    :return: save images in directory for inspection
    """

    # set prediction to black if not given
    if len(records) < 3:
        records.append(np.zeros(records[0].shape))

    seg_cmap, seg_norm, bounds = color_mappings()
    fig = plt.figure(figsize=(16, 4))

    gs = gridspec.GridSpec(nrows=1,
                           ncols=3,
                           figure=fig,
                           width_ratios=[1, 1, 1],
                           height_ratios=[1],
                           wspace=0.3,
                           hspace=0.3)

    # turn image to 3 channel
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(records[0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("oct")

    # check label shape
    if len(records[1].shape) == 3:
        records[1] = records[1][:, :, 0]

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(records[1], cmap=seg_cmap, norm=seg_norm)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("ground truth")

    ax3 = fig.add_subplot(gs[0, 2])
    colorbar_im = ax3.imshow(records[2], cmap=seg_cmap, norm=seg_norm)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("prediction")

    # set colorbar ticks
    tick_loc_array = np.arange(len(bounds)) + 0.5
    tick_loc_list = tick_loc_array.tolist()
    tick_list = np.arange(len(bounds)).tolist()
    c_bar = plt.colorbar(colorbar_im, cmap=seg_cmap, norm=seg_norm, boundaries=bounds)

    # set ticks
    c_bar.set_ticks(tick_loc_list)
    c_bar.ax.set_yticklabels(tick_list)

    if not os.path.exists(os.path.join(model_dir, mode + "_records")):
        os.makedirs(os.path.join(model_dir, mode + "_records"))

    plt.savefig(os.path.join(model_dir, mode + "_records", filename))
    plt.close()


def plot_image_predictions(records, model_dir, mode, filename):
    """
    :param records: list containing numpy array of image and prediction
    :param model_dir: directory of model where to save images
    :param mode: str: train or test
    :param filename: str: filename of image
    :return: save images in directory for inspection
    """

    # set prediction to black if not given
    if len(records) < 3:
        records.append(np.zeros(records[0].shape))

    seg_cmap, seg_norm, bounds = color_mappings()
    fig = plt.figure(figsize=(16, 4))

    gs = gridspec.GridSpec(nrows=1,
                           ncols=2,
                           figure=fig,
                           width_ratios=[1, 1],
                           height_ratios=[1],
                           wspace=0.3,
                           hspace=0.3)

    # turn image to 3 channel
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(records[0])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("oct")

    ax3 = fig.add_subplot(gs[0, 1])
    colorbar_im = ax3.imshow(records[1], cmap=seg_cmap, norm=seg_norm)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("prediction")

    # set colorbar ticks
    tick_loc_array = np.arange(len(bounds)) + 0.5
    tick_loc_list = tick_loc_array.tolist()
    tick_list = np.arange(len(bounds)).tolist()
    c_bar = plt.colorbar(colorbar_im, cmap=seg_cmap, norm=seg_norm, boundaries=bounds)

    # set ticks
    c_bar.set_ticks(tick_loc_list)
    c_bar.ax.set_yticklabels(tick_list)

    if not os.path.exists(os.path.join(model_dir, mode + "_img_pred_records")):
        os.makedirs(os.path.join(model_dir, mode + "_img_pred_records"))

    plt.savefig(os.path.join(model_dir, mode + "_img_pred_records", filename))
    plt.close()


def get_max_min_uncertainty(all_uq_maps):
    """
    Parameters
    ----------
    all_uq_maps : dict with file names as keys and uq maps and values

    Returns
    -------
    min and max uncertainty value as floats
    """
    min_value = 0
    max_value = 0

    for record in all_uq_maps.keys():
        max_ = np.max(all_uq_maps[record])
        min_ = np.min(all_uq_maps[record])


        if min_ < min_value:
            min_value = min_
        if max_ > max_value:
            max_value = max_
    return max_value, min_value


def get_feature_uncertainty_distribution(lbl, record_uq_map):
    """
    Parameters
    ----------
    lbl : label map
    record_uq_map :uncertainty map

    Returns
    -------
    dict with mean uncertainty for each label
    """
    labels = np.unique(lbl)

    if len(lbl.shape) > 2:
        lbl = lbl[:, :, 0]

    uncertainty_log = {}
    for label in labels:
        if label not in [0, 14, 15]:
            mask = lbl == label
            uncertainty_log[label] = record_uq_map[mask].flatten()
    return uncertainty_log


def plot_uncertainty_statistics(all_uq_maps, ensemble_dir):
    """
    Plots the uncertainty box plot for the prediction
    Parameters
    ----------
    all_uq_maps : dict with file names as keys and uq maps and values
    ensemble_dir : directory to save output

    Returns
    -------
    None
    """
    mode = "test"

    max_value, min_value = get_max_min_uncertainty(all_uq_maps)

    save_dir = os.path.join(ensemble_dir, mode + "_uncertainty_statistics")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # plot uq maps
    for record in all_uq_maps.keys():
        record_uq_map = all_uq_maps[record]

        img_path = os.path.join(TRAIN_DATA_PATH, "images", record)
        label_path = os.path.join(TRAIN_DATA_PATH, "masks", record)
        _, lbl = read_resize(img_path, label_path, record_uq_map.shape)

        uncertainty_log = get_feature_uncertainty_distribution(lbl, record_uq_map)

        labels, data = uncertainty_log.keys(), uncertainty_log.values()

        plt.style.use('ggplot')
        plt.boxplot(data, showfliers = False, )
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylim([min_value, max_value])
        plt.savefig(os.path.join(save_dir, record))
        plt.close()


def plot_uncertainty_heatmaps(all_uq_maps, ensemble_dir):
    """
    gets min and max values for uncertainties and then makes and saves heatmaps
    Parameters
    ----------
    all_uq_maps : dict with file names as keys and uq maps and values
    ensemble_dir : directory to save output

    Returns
    -------

    """
    mode = "test"
    save_dir = os.path.join(ensemble_dir, mode + "uncertainty_records")
    max_value, min_value = get_max_min_uncertainty(all_uq_maps)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # plot uq maps
    for record in all_uq_maps.keys():
        record_uq_map = all_uq_maps[record]
        plt.imshow(record_uq_map, cmap = 'hot', interpolation = 'nearest', vmin = min_value, vmax = max_value)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, record))
        plt.close()


