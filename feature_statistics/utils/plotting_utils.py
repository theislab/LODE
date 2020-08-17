from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec


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


def plot_segmentation_map(cls, show=False, save_path=None, img_name = None):
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

    if show:
        plt.show()

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok = True)
        plt.savefig(os.path.join(save_path, img_name))
    plt.close()
    return colorbar_im