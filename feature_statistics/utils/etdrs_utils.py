import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from pydicom import read_file
from skimage.measure import moments
from utils.plotting_utils import plot_segmentation_map
import cv2
import scipy


def plot_bool_mask(mask):
    from tvtk.api import tvtk
    from mayavi import mlab
    X, Y, Z = np.mgrid[-10:10:256j, -10:10:256j, -10:10:256j]
    data = mask.astype("float")
    i = tvtk.ImageData(spacing = (1, 1, 1), origin = (0, 0, 0))
    i.point_data.scalars = data.ravel()
    i.point_data.scalars.name = 'scalars'
    i.dimensions = data.shape
    mlab.pipeline.surface(i)
    mlab.colorbar(orientation = 'vertical')
    mlab.show()


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


tissues = {"1": "epiratinal_membrane",
           "2": "neurosensory_retina", "3": "intraretinal_fluid", "4": "subretinal_fluid",
           "5": "subretinal_hyperreflective_material", "6": "rpe", "7": "fibrovascular_ped",
           "8": "drusen", "9": "posterior_hylaoid_membrane", "10": "choroid",
           "13": "fibrosis"}

etdrs_regions = ["C0", "S2", "S1", "N1", "N2", "I1", "I2", "T1", "T2"]

# labels to exclude from thickness
thickness_exclusion = [0, 1,  9, 10, 11, 12, 14, 15]


def volume_spatial_grid(volume):
    scans, height, width = volume.shape
    volume_container = np.zeros([width, height, width])

    # get B-scan locations
    b_locations = np.round(np.linspace(0, 256 - 1, 49)).astype(int)

    # create square sized oct volume
    for k, b_loc in enumerate(b_locations):
        volume_container[b_loc, :, :] = volume[k, :, :]
    return volume_container


def thickness_spatial_grid(map, interpolate=False):
    """
    @param map: thickness vectors
    @type map: arry
    @param interpolate: if to interpolate accross thickness grid
    @type interpolate: bool
    @return: 256,256 sized thickness grid
    @rtype: array
    """
    thickness_map = np.zeros([256, 256])

    # get B-scan locations
    b_locations = np.round(np.linspace(0, 256 - 1, 49)).astype(int)

    # create square sized oct volume
    for k, b_loc in enumerate(b_locations):
        thickness_map[b_loc, :] = map[k, :, 0]

    if interpolate:
        for j in range(thickness_map.shape[1]):
            thickness_map[:, j][thickness_map[:, j] == 0] = np.nan
            nans, x = np.isnan(thickness_map[:, j]), lambda z: z.nonzero()[0]
            thickness_map[:, j][nans] = np.interp(x(nans), x(~nans), thickness_map[:, j][~nans])
    return thickness_map


def thickness_profiler(volume):
    """
    @param vol:
    @type vol:
    @return:
    @rtype:
    """
    # set labels to zero
    volume[np.isin(volume, thickness_exclusion)] = 0

    # set rest to one
    volume[~np.isin(volume, thickness_exclusion)] = 1

    # create thickness map
    thickness_map = np.zeros([volume.shape[0], volume.shape[2], 1])

    # get thickness vectors
    for i in range(volume.shape[0]):
        b_scan = volume[i, :, :]

        for j in range(b_scan.shape[1]):
            top = min(np.argwhere(b_scan[:, j]).reshape(-1), default = 0)
            bottom = max(np.argwhere(b_scan[:, j]).reshape(-1), default = 0)

            # get pixel thickness
            thickness = np.abs(bottom - top)

            # assign to map
            thickness_map[i, j, :] = thickness
    return thickness_map


def rpe_profiler(volume):
    map_positions = np.linspace(0, 255, volume.shape[0], dtype=np.int32)
    rpe_map = np.zeros([volume.shape[1], volume.shape[2]], dtype = np.int32)
    for i in range(volume.shape[0]):
        neurosensory_retina = np.argwhere(np.sum(volume[i, :, :] == 2, axis = 0))
        if not neurosensory_retina.any():
            rpe_map[i, :] = 0
            continue

        start, stop = np.min(neurosensory_retina), np.max(neurosensory_retina)

        rpe_bool = volume[i, :, :] == 6
        rpe_presence = np.sum(rpe_bool, axis = 0)[start:stop]

        # find absence of rpe
        rpe_map[map_positions[i], start:stop] = rpe_presence == 0

    # interpolate atrophy regions
    previous_position = None
    for k, position in enumerate(map_positions):
        if k > 0:
            current_v = rpe_map[position, :]
            previous_v = rpe_map[previous_position, :]

            element_index = 0
            for element_a, element_b in zip(current_v, previous_v):
                if element_a == 1 & element_b == 1:
                    # fill atrophy values
                    rpe_map[previous_position:position, element_index] = 1
                    element_index += 1
                else:
                    element_index += 1
                    continue
        # set prev vector index
        previous_position = position
    return rpe_map


def laterality_string_frormatting(laterality):
    """
    @param laterality: left or right eye
    @type laterality: str
    @return: laterality written as Left ör Right
    @rtype: str
    """
    if laterality == "L":
        laterality = "Left"
    elif laterality == "R":
        laterality = "Right"
    else:
        laterality = laterality
    return laterality


def get_dicom_pixel_measurement(dc):
    """
    @param dc:
    @type dc:
    @return:
    @rtype:
    """
    shared = dc.SharedFunctionalGroupsSequence[0][("0028", "9110")][0]

    y_resolution = shared.PixelSpacing[0]
    x_resolution = shared.PixelSpacing[1]
    slice_thickness = shared.SliceThickness
    return y_resolution, x_resolution, slice_thickness


def get_pixel_2_volume_factors(dicom_path):
    """
    @param dicom_path:
    @type dicom_path:
    @return:
    @rtype:
    """

    # factor to resize networks output size
    B_SCAN_RESIZE_FACTOR = 2
    try:
        # load dicom header
        dicom_file = os.listdir(dicom_path)[0]

        dc = read_file(os.path.join(dicom_path, dicom_file), stop_before_pixels = True)

        # get resolution attribution
        y_resolution, x_resolution, slice_thickness = get_dicom_pixel_measurement(dc)
    
    except:
        print("no dicom or resolution data available, setting to default")
        y_resolution = 0.003872
        x_resolution = 0.012034
        slice_thickness = 0.128197

    pixel_2_volume_mm3 = B_SCAN_RESIZE_FACTOR * y_resolution * x_resolution * slice_thickness
    pixel_height_2_mm = B_SCAN_RESIZE_FACTOR * y_resolution
    return pixel_2_volume_mm3, pixel_height_2_mm


class ETDRSUtils:
    def __init__(self, path, dicom_path):
        self.path = path
        self.record = path.split("/")[-1]
        self.patient = self.record.split("_")[0]
        self.study_date = self.record.split("_")[1]
        self.laterality = laterality_string_frormatting(self.record.split("_")[2])
        self.dicom_path = os.path.join(dicom_path, self.patient, self.laterality, self.study_date)
        self.width = None
        self.height = None
        self.scans = None
        self.volume_scans = None
        self.etdrs_bool_grid = {}
        self.etdrs_stat = None

    def upper_right(self):
        return np.arange(0, self.width)[:, None] <= np.arange(self.height)

    def lower_right(self):
        return self.upper_right()[::-1]

    def upper_left(self):
        return self.lower_left()[::-1]

    def lower_left(self):
        return np.arange(0, self.width)[:, None] > np.arange(self.height)

    def inner_disk(self):
        inner_ring_radius = int(self.height // 6) / 2
        return create_circular_mask(h = self.height, w = self.width, center = None, radius = inner_ring_radius)

    def middle_disk(self):
        middle_ring_radius = int(self.height // 2) / 2
        middle_disk = create_circular_mask(h = self.height, w = self.width, center = None, radius = middle_ring_radius)
        middle_disk[self.inner_disk() == 1] = 0
        return middle_disk

    def outer_disk(self):
        outer_ring_radius = int(self.height // 1) / 2
        outer_disk = create_circular_mask(h = self.height, w = self.width, center = None, radius = outer_ring_radius)
        outer_disk[self.inner_disk() == 1] = 0
        outer_disk[self.middle_disk() == 1] = 0
        return outer_disk

    def make_volume_mask(self, mask):
        return np.stack((mask,) * self.volume_scans, axis = 0)

    def zones(self):
        zones_dict = {"middle": self.middle_disk(), "outer": self.outer_disk(),
                      "upper_right": self.upper_right(), "lower_left": self.lower_left(),
                      "upper_left": self.upper_left(), "lower_right": self.lower_right()}

        # convert masks to 3d
        for key in zones_dict.keys():
            zones_dict[key] = self.make_volume_mask(zones_dict[key])

        # define mask combinations
        rois = [("upper_left", "outer", "upper_right"),
                ("upper_left", "middle", "upper_right"),
                ("lower_right", "middle", "upper_right"),
                ("lower_right", "outer", "upper_right"),
                ("lower_right", "middle", "lower_left"),
                ("lower_right", "outer", "lower_left"),
                ("upper_left", "middle", "lower_left"),
                ("upper_left", "outer", "lower_left")]

        # create 9 depth regions
        self.etdrs_bool_grid["C0"] = self.make_volume_mask(self.inner_disk())
        for k, roi in enumerate(rois, 1):
            self.etdrs_bool_grid[etdrs_regions[k]] = np.asarray(zones_dict[roi[0]] &
                                                                zones_dict[roi[1]] &
                                                                zones_dict[roi[2]])

    def get_etdrs_stats(self):
        record_log = {}

        # load segmentation
        volume = np.load(self.path)

        pixel_2_volume_mm3, pixel_height_2_mm = get_pixel_2_volume_factors(self.dicom_path)

        # get dim
        self.scans, self.height, self.width = volume.shape

        # get atrophy map
        atrophy_map = rpe_profiler(volume)

        # distribute maps across oct grid
        volume_spatial = volume_spatial_grid(volume)
        self.volume_scans = volume_spatial.shape[0]

        # get thickness map
        thickness_map = thickness_profiler(np.copy(volume))

        # distribute maps across oct grid
        thickness_spatial = thickness_spatial_grid(thickness_map)

        # get etdrs bool grid
        self.zones()

        # save record name
        record_log["record"] = self.path.split("/")[-1].replace(".npy", "")

        # for each etdrs region, get statistics
        for etdrs_region in self.etdrs_bool_grid.keys():
            bool_ = self.etdrs_bool_grid[etdrs_region]
            region_segment = volume_spatial[bool_]
            thickness_segment = thickness_spatial[bool_[0, :, :]]
            atrophy_segment = atrophy_map[bool_[0, :, :]]

            # log thickness feature
            thickness_mean = np.mean(thickness_segment[np.nonzero(thickness_segment)] * pixel_height_2_mm)

            # add all tissue types count for etdrs regions
            record_log[etdrs_region + "_" + "thickness_mean"] = thickness_mean
            record_log[etdrs_region + "_" + "atropy_percentage"] = np.mean(atrophy_segment)

            region_total = 0
            # get all tissue counts for all tissues
            for tissue in tissues.keys():
                tissue_pixel_count = np.sum(region_segment == int(tissue))
                tissue_volume = tissue_pixel_count * pixel_2_volume_mm3

                record_log[etdrs_region + "_" + tissue] = tissue_volume

                if tissue not in [9, 10]:
                    region_total += tissue_volume

            # add all tissue types count for etdrs regions
            record_log[etdrs_region + "_" + "total"] = region_total
        return record_log
