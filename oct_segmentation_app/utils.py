from __future__ import print_function
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

CLASS_MAPPING = {0: "background",
                 1: "epm",
                 2: "neurosensory retina",
                 3: "intraretinal fluid",
                 4: "subretinal fluid",
                 5: "subretinal hyper refl. mat.",
                 6: "rpe",
                 7: "fibro vascular ped",
                 8: "drusen",
                 9: "posterior hyaloid membrane",
                 10: "choroid",
                 11: "serous ped",
                 12: "image artifact",
                 13: "fibrosis",
                 14: "vetrous",
                 15: "camera effect"}


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

    color_palett_norm = color_palett / 255.  # (np.max(color_palett)-np.min(color_palett))
    custom_cmap = colors.ListedColormap(
        color_palett_norm
    )

    # set counts and norm
    array_bounds = np.arange(color_palett.shape[0] + 1) - 0.5
    bounds = array_bounds.tolist()
    norm = colors.BoundaryNorm(bounds, custom_cmap.N)
    return custom_cmap, norm, bounds


def plot_segmentation(segmentation, show_legend=False, show_legend_text=False):
    seg_cmap, seg_norm, bounds = color_mappings()
    seg_plot = plt.imshow(segmentation, interpolation='nearest', cmap=seg_cmap, norm=seg_norm)

    if show_legend:
        # set colorbar ticks
        tick_loc_array = np.arange(seg_cmap.colors.shape[0])
        tick_loc_list = tick_loc_array.tolist()
        tick_list = np.arange(seg_cmap.colors.shape[0]).tolist()

        c_bar = plt.colorbar(seg_plot, cmap=seg_cmap, norm=seg_norm, boundaries=bounds, fraction=0.4)

        # set ticks
        c_bar.set_ticks(tick_loc_list)

        if show_legend_text:
            c_bar.ax.set_yticklabels(CLASS_MAPPING.values())
        else:
            c_bar.ax.set_yticklabels(tick_list)
    return seg_plot
