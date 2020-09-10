import math
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from utils.time_utils import TimeUtils
from feature_statistics.config import WORK_SPACE, SEG_DIR
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from utils.pandas_utils import sum_etdrs_columns, get_total_number_of_injections, interpolate_numeric_field, \
    avg_etdrs_columns
from utils.util_functions import nan_helper, SeqUtils
from utils.plotting_utils import plot_segmentation_map, color_mappings
import datetime
import itertools
from tqdm import tqdm
import matplotlib.cm as cm

import glob
import matplotlib.patches as mpatches
colors = [[203., 54., 68.],
            [192., 194., 149.],
            [105., 194., 185.],
            [205., 205., 205.],
            [140., 204., 177.],
            [183., 186., 219.],
            [114, 137, 218],
            [209., 227., 239.],
            [226., 233., 48.]]

class MeasureSeqAtrophy(SeqUtils):
    NUMBER_OF_MONTHS = 13
    NUM_SEGMENTATIONS = 3
    DAYS = 30
    REGIONS = ["S", "N", "I", "T"]
    ATROPHY = []
    ATROPHY_COLUMNS = ["avg_atrophy"]

    # ['C0_atropy_percentage', 'S2_atropy_percentage', 'S1_atropy_percentage', 'N1_atropy_percentage',
    #                   'N2_atropy_percentage', 'I1_atropy_percentage', 'I2_atropy_percentage', 'T1_atropy_percentage',
    #                   'T2_atropy_percentage']

    DATA_POINTS = ["study_date", "avg_atrophy", "total_injections", "cur_va_rounded", "next_va"]
    META_DATA = ["patient_id", "laterality", "diagnosis"]
    SEG_PATHS = glob.glob(os.path.join(SEG_DIR, "*"))

    def __init__(self, meta_data, time_line, naive, number_of_injections, treatment_dict):
        self.patient_id = meta_data[MeasureSeqAtrophy.META_DATA[0]]
        self.laterality = meta_data[MeasureSeqAtrophy.META_DATA[1]]
        self.diagnosis = meta_data[MeasureSeqAtrophy.META_DATA[2]]
        self.time_line = time_line
        self.number_of_months = len(time_line)
        self.naive = naive
        self.atrophy_delta = self.get_atrophy_delta
        self.number_of_visits = self.get_number_of_visits()
        self.number_of_injections = number_of_injections
        self.three_month_effect = treatment_dict["three_month"]
        self.six_month_effect = treatment_dict["six_month"]

    @classmethod
    def from_record(cls, record_table):
        time_utils = TimeUtils(record_table = record_table)
        time_line = time_utils.time_line

        # reset index of table
        record_table.reset_index(inplace = True)

        # add total fluid
        record_table = avg_etdrs_columns(record_table, rings = [1, 2], regions = MeasureSeqAtrophy.REGIONS,
                                         features = ["atropy_percentage"], foveal_region = ["C0"],
                                         new_column_name = "avg_atrophy")

        # round va
        record_table.insert(loc = 10, column = "cur_va_rounded", value = record_table.cur_va.round(2),
                            allow_duplicates = True)

        # add total injections
        record_table = get_total_number_of_injections(table = record_table)
        MeasureSeqAtrophy.DATA_POINTS = MeasureSeqAtrophy.DATA_POINTS + MeasureSeqAtrophy.ATROPHY_COLUMNS

        # assign items to time line
        for data_point in MeasureSeqAtrophy.DATA_POINTS:
            time_line = time_utils.assign_to_timeline(time_line = time_line, item = data_point)

        # interpolate time vector
        for item in ["avg_atrophy"] + MeasureSeqAtrophy.ATROPHY_COLUMNS:
            time_line = interpolate_numeric_field(time_line, item = item)

        treatment_dict = {}
        # set treatment effect
        for dist in ["three", "six"]:
            time_line = SeqUtils.set_treatment_effect(time_line, time_dist = dist, item = "avg_atrophy")

            item = f"{dist}_month_effect"
            treatment_dict[f"{dist}_month"] = SeqUtils.get_treatment_effect(item, time_line)

        # check if naive record
        naive = SeqUtils.is_naive(record_table)

        # get number of injections total
        number_of_injections = SeqUtils.get_seq_number_of_injections(time_line)
        return cls(meta_data = record_table.iloc[0], time_line = time_line, naive = naive,
                   number_of_injections = number_of_injections, treatment_dict = treatment_dict)

    def get_number_of_visits(self):
        return len(list(filter(lambda x: x != "nan", self.study_dates)))

    @property
    def get_atrophy_delta(self):
        start_value = self.time_line[min(self.time_line.keys())]["avg_atrophy"]
        end_value = self.time_line[max(self.time_line.keys())]["avg_atrophy"]
        return end_value - start_value

    @property
    def time_series(self):
        time_series = dict.fromkeys(MeasureSeqAtrophy.ATROPHY_COLUMNS)
        # iterate through months and retrieve measurement vector
        months = list(self.time_line.keys())
        for feature in time_series.keys():
            time_series[feature] = []
            for month in months:
                time_series[feature].append(self.time_line[month][feature])
        return time_series

    @property
    def segmentation_paths(self):
        laterality_cond = "_" + self.laterality + "_"
        patient_cond = str(self.patient_id) + "_"
        segmentation_files = list(
            filter(lambda k: (patient_cond in k) & (laterality_cond in k), MeasureSeqAtrophy.SEG_PATHS))
        return segmentation_files

    def record_identifier(self, study_date):
        """
        @param study_date: date of patient visit
        @type study_date: str
        @return: string with *str*, for record search
        @rtype:str
        """
        date_field = study_date.replace("-", "")
        return f"*{self.patient_id}_{date_field}_{self.laterality}_*"

    def dump_segmentation_map(self, month):
        """
        takes all maps from month and writed to disk
        @param month: month
        @type month: int
        @return: None
        @rtype:
        """
        if self.time_line[month]["study_date"] is np.nan:
            print("No segmentation map exists for this month")
            pass
        else:
            date = self.time_line[month]["study_date"].replace("-", "")
            path = list(filter(lambda k: (date.replace("-", "") in k), self.segmentation_paths))[0]
            map = np.load(path)

            indices = np.linspace(0, map.shape[0] - 1, map.shape[0] - 1, dtype = np.int32)
            for idx in indices:
                plot_segmentation_map(map[idx, :, :], show = False,
                                      save_path = os.path.join(WORK_SPACE,
                                                               f"dump/2d_segmentations/{self.patient_id}/{date}"),
                                      img_name = f"{idx}.png")

    def add_segmentation_to_timeline(self):
        """
        Assigns segmented oct volumes to time line
        @return: None
        @rtype:
        """
        months = list(self.time_line.keys())
        for month in months:
            if self.time_line[month]["study_date"] is np.nan:
                self.time_line[month]["segmentation_maps"] = np.nan
                continue
            else:
                date = self.time_line[month]["study_date"].replace("-", "")
                path = list(filter(lambda k: (date.replace("-", "") in k), self.segmentation_paths))[0]
                map = np.load(path)

                indices = np.linspace(0, map.shape[0] - 1, MeasureSeqAtrophy.NUM_SEGMENTATIONS, dtype = np.int32)
                self.time_line[month]["segmentation_maps"] = map[indices, :, :]

    def show_time_series(self, show_segmentations=False, show=False, save_fig=False, colors=None):
        """
        @param show_segmentations: whether to show segmentations
        @type show_segmentations: bool
        @param show: whether to show image
        @type show: bool
        @param save_fig: whether to fave figure
        @type save_fig: bool
        @return: None
        @rtype:
        """
        font = {'family': 'serif',
                'color': 'darkorange',
                'weight': 'normal',
                'size': 10}

        if show_segmentations:
            seg_cmap, seg_norm, bounds = color_mappings()
            self.add_segmentation_to_timeline()

        plt.style.use('ggplot')

        darkred_patch = mpatches.Patch(color = 'darkorange', label = 'injections')
        time_series = self.time_series

        ts_length = len(self.time_series[list(self.time_series.keys())[0]])

        fig, ax = plt.subplots(figsize = (ts_length, 10))
        fig.subplots_adjust(bottom = 0.4)

        y_max = 1.0
        y_min = 0

        xs = np.arange(0, ts_length, 1)
        ys = [0.01]*ts_length

        colors_iter = itertools.cycle([ "b", "r", "g", "y", "orange", "darkblue", "peru", "pink", "purple"])
        for feature in self.time_series.keys():
            # plot points
            ax.plot(xs, time_series[feature], "bo-", label = feature, color=next(colors_iter))

        x_offset = 0
        xy_box = {0: (x_offset, y_max // 5),
                  1: (x_offset, (y_max // 5) * 2),
                  2: (x_offset, (y_max // 5) * 3)}

        # set image zoom by time series length
        zoom = 1.5 / self.number_of_visits

        # zip joins x and y coordinates in pairs
        for x, y in zip(xs, ys):
            if not self.time_line[x + 1]['cur_va_rounded'] is np.nan:
                label = f"Inj: {self.time_line[x + 1]['total_injections']} \n VA: {self.time_line[x + 1]['cur_va_rounded']}"
            else:
                label = ""
            ax.text(x, y, label, fontdict = font)

            if show_segmentations & (self.time_line[x + 1]["study_date"] is not np.nan):
                for i in range(MeasureSeqAtrophy.NUM_SEGMENTATIONS):
                    imagebox = OffsetImage(self.time_line[x + 1]["segmentation_maps"][i, :, :],
                                           zoom = zoom, cmap = seg_cmap, norm = seg_norm)
                    imagebox.image.axes = ax
                    ab = AnnotationBbox(imagebox, (x, 0), xybox = (x + xy_box[i][0], 0 - xy_box[i][1]),
                                        frameon = False)
                    ax.add_artist(ab)

        plt.ylim(y_min, y_max)
        plt.xlabel("month")
        plt.ylabel("atrophy ")
        plt.legend()

        title_ = f"{self.patient_id}_{self.laterality}"
        plt.title(title_)

        if show:
            plt.show()

        if save_fig:
            save_path = os.path.join(WORK_SPACE, "dump")
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok = True)
            ax.figure.savefig(os.path.join(save_path, f"{title_}_timeseries.png"), dpi = ax.figure.dpi)
            plt.close()


if __name__ == "__main__":
    """
    questions:
    # distribution of sequence lengths & sequence time span
    # number of sequences with injections
    # number of sequences with diff diagnosises
    # distribution of time until dry
    # distribution of 3 and 6 month treatment effect
    """

    # load sequences
    seq_pd = pd.read_csv(os.path.join(WORK_SPACE, 'sequence_data/sequences.csv'))

    PATIENT_ID = 709  # 2005
    LATERALITY = "L"

    filter_ = (seq_pd.patient_id == PATIENT_ID) & (seq_pd.laterality == LATERALITY)
    # seq_pd = seq_pd.loc[filter_]

    unique_records = seq_pd[["patient_id", "laterality"]].drop_duplicates() # .iloc[0:10]

    time_until_dry = []
    for patient, lat in tqdm(unique_records.itertuples(index = False)):
        record_pd = seq_pd[(seq_pd.patient_id == patient) & (seq_pd.laterality == lat)]
        time_until_dry.append(MeasureSeqAtrophy.from_record(record_pd))
        time_until_dry[-1].show_time_series(show_segmentations = False, show = False, save_fig = True, colors = colors)
        time_until_dry[-1].dump_segmentation_map(11)

    time_serie_log = {"patient_id": [], "laterality": [], "number_of_injections": [],
                      "three_month_effect": [], "six_month_effect": [], "number_of_months": [],
                      "diagnosis": [], "number_of_visits": [], "naive": [], "atrophy_delta": []}

    for measurem in time_until_dry:
        for key in time_serie_log.keys():
            time_serie_log[key].append(measurem.__dict__[key])

    time_until_dry_pd = pd.DataFrame(time_serie_log)
    time_until_dry_pd.to_csv(os.path.join(WORK_SPACE, "atrophy.csv"))
