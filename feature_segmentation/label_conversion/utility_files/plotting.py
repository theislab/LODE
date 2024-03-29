from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec


def color_mappings():
    color_palett = np.array([[148., 158., 167.],
                             [11., 151., 199.],
                             [30., 122., 57.],
                             [135., 191., 234.], # IRF
                             [37., 111., 182.], # SRF
                             [156., 99., 84.], # SRHM
                             [226., 148., 60.], # RPE
                             [203., 54., 68.],#Fib V PED
                             [192., 194., 149.], # DRUSEN
                             [105., 194., 185.], # Post H Mem
                             [205., 205., 205.], # choroid
                             [140., 204., 177.],  # Serous PED
                             [183., 186., 219.],  # other artifact
                             [114, 137, 218],  # fibrosis
                             [209., 227., 239.], # vitreoud
                             [226., 233., 48.]])

    color_palett_norm = np.clip(color_palett, 0, 255) / 255  # (np.max(color_palett)-np.min(color_palett))
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
    colorbar_im = plt.imshow(cls, interpolation = "nearest", cmap = seg_cmap, norm = seg_norm)

    # set colorbar ticks
    tick_loc_array = np.arange(len(bounds) - 1) + 0.5
    tick_loc_list = tick_loc_array.tolist()

    tick_list = np.arange(len(bounds) - 1).tolist()
    c_bar = plt.colorbar(colorbar_im, cmap = seg_cmap, norm = seg_norm, boundaries = bounds)
    # set ticks
    c_bar.set_ticks(tick_loc_list)
    c_bar.ax.set_yticklabels(tick_list)

    plt.savefig(out_clsv_file.replace(".npy", ".png"), bbox_inches='tight', pad_inches=0)
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
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_model_run_images(records, model_dir, mode, filename):
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

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(records[1], interpolation = "nearest", cmap=seg_cmap, norm=seg_norm)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("ground truth")

    ax3 = fig.add_subplot(gs[0, 2])
    colorbar_im = ax3.imshow(records[2], interpolation = "nearest", cmap=seg_cmap, norm=seg_norm)
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

    plt.savefig(os.path.join(model_dir, mode + "_records", filename), bbox_inches='tight', pad_inches=0)
    plt.close()



def create_visualizations(out_clsv_file, cls):
    seg_cmap, seg_norm, bounds = color_mappings()

    colorbar_im = plt.imshow(cls, interpolation = "nearest", cmap = seg_cmap, norm = seg_norm)
    # set colorbar ticks
    tick_loc_array = np.arange(12) + 0.5
    tick_loc_list = tick_loc_array.tolist()
    fig = plt.figure(figsize = (16, 4))

    gs = gridspec.GridSpec(nrows = 1,
                           ncols = 1,
                           figure = fig,
                           width_ratios = [1],
                           height_ratios = [1],
                           wspace = 0.3,
                           hspace = 0.3)

    ax2 = fig.add_subplot(gs[0, 0])
    ax2.imshow(cls, interpolation = "nearest", cmap = seg_cmap, norm = seg_norm)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.savefig(out_clsv_file.replace(".npy", ".png"), bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_examples(record, path):
    seg_cmap, seg_norm, bounds = color_mappings()
    fig = plt.figure(figsize = (24, 8))
    columns = 3
    rows = 1
    types = ["image", "pre processing", "post processing"]
    for i in range(1, columns * rows + 1):
        img = record[i - 1]
        fig.add_subplot(rows, columns, i)

        #
        if types[i - 1] != "image":
            colorbar_im = plt.imshow(img, interpolation = "nearest", cmap = seg_cmap, norm = seg_norm)

        if types[i - 1] == "image":
            if len(img.shape) < 3:
                img = np.stack((img,) * 3, axis = -1)
            plt.imshow(img)

        plt.title(types[i - 1])

    plt.savefig(path)
    plt.close()

