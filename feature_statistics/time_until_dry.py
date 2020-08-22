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
from utils.pandas_utils import sum_etdrs_columns, get_total_number_of_injections, interpolate_numeric_field
from utils.util_functions import nan_helper
from utils.plotting_utils import plot_segmentation_map, color_mappings
import datetime
from tqdm import tqdm
import glob
import matplotlib.patches as mpatches


class MeasureSeqTimeUntilDry():
    NUMBER_OF_MONTHS = 13
    NUM_SEGMENTATIONS = 3
    DAYS = 30
    REGIONS = ["S", "N", "I", "T"]
    FLUIDS = ["3", "4"]
    DATA_POINTS = ["study_date", "total_fluid", "total_injections", "cur_va_rounded", "next_va"]
    META_DATA = ["patient_id", "laterality", "diagnosis"]
    SEG_PATHS = glob.glob(os.path.join(SEG_DIR, "*"))

    def __init__(self, meta_data, time_line, naive, time_until_dry, number_of_injections,
                 treatment_dict):

        self.patient_id = meta_data[MeasureSeqTimeUntilDry.META_DATA[0]]
        self.laterality = meta_data[MeasureSeqTimeUntilDry.META_DATA[1]]
        self.diagnosis = meta_data[MeasureSeqTimeUntilDry.META_DATA[2]]

        self.time_line = time_line
        self.number_of_months = len(time_line)
        self.naive = naive
        self.time_until_dry = time_until_dry
        self.number_of_injections = number_of_injections
        self.three_month_effect = treatment_dict["three_month"]
        self.six_month_effect = treatment_dict["six_month"]

    @classmethod
    def from_record(cls, record_table):
        print(record_table.patient_id.iloc[0], record_table.laterality.iloc[0])
        time_utils = TimeUtils(record_table = record_table)
        time_line = time_utils.time_line

        # reset index of table
        record_table.reset_index(inplace = True)

        # add total fluid
        record_table = sum_etdrs_columns(record_table, rings = [1, 2], regions = MeasureSeqTimeUntilDry.REGIONS,
                                         features = [3, 4], foveal_region = ["C0"], new_column_name = "total_fluid")

        # round va
        record_table.insert(loc = 10, column = "cur_va_rounded", value = record_table.cur_va.round(2),
                            allow_duplicates = True)

        # add total injections
        record_table = get_total_number_of_injections(table = record_table)

        # assign items to time line
        for data_point in MeasureSeqTimeUntilDry.DATA_POINTS:
            time_line = time_utils.assign_to_timeline(time_line = time_line, item = data_point)

        # interpolate time vector
        for item in ["total_fluid"]:
            time_line = interpolate_numeric_field(time_line, item = item)

        treatment_dict = {}
        # set treatment effect
        for dist in ["three", "six"]:
            time_line = cls.set_treatment_effect(time_line, time_dist = dist)

            item = f"{dist}_month_effect"
            treatment_dict[f"{dist}_month"] = cls.get_treatment_effect(item, time_line)

        # calculate time until dry, -1 means not dry
        time_ = cls.get_time_until_dry(time_line)

        # check if naive record
        naive = cls.is_naive(record_table)

        # get number of injections total
        number_of_injections = cls.get_seq_number_of_injections(time_line)
        return cls(meta_data = record_table.iloc[0], time_line = time_line, naive = naive, time_until_dry = time_,
                   number_of_injections = number_of_injections, treatment_dict = treatment_dict)

    @classmethod
    def get_treatment_effect(cls, item, time_line):
        log = [time_line[time_point][item] for time_point in time_line.keys() if
               time_line[time_point][item] is not np.nan]
        if not log:
            return np.nan
        else:
            return log[0]

    @classmethod
    def get_seq_number_of_injections(cls, time_line):
        return np.nansum([time_line[time_point]["total_injections"] for time_point in time_line])

    @classmethod
    def is_naive(cls, table):
        first_record = table.iloc[[0]]
        injections = list(map(lambda x: int(x), first_record.injections[0].split(", ")))
        return (not first_record.lens_surgery[0]) & (np.sum(injections) == 0)

    @classmethod
    def get_time_until_dry(cls, time_line):
        months = list(time_line.keys())
        for month in months:
            if time_line[month]["total_fluid"] == 0:
                return month
            else:
                continue
        return -1

    @classmethod
    def set_treatment_effect(cls, time_line, time_dist):
        time_dict = {"three": 2, "six": 5}
        for time_point in time_line.keys():
            injection_point = time_line[time_point]["total_injections"]

            injection_bool = (injection_point > 0) and (injection_point is not np.nan)
            time_bool = (time_point + time_dict[time_dist] <= len(time_line))
            if injection_bool and time_bool:
                current_fluid = time_line[time_point]["total_fluid"]
                effect = time_line[time_point + time_dict[time_dist]]["total_fluid"] / current_fluid
                time_line[time_point][f"{time_dist}_month_effect"] = effect
            else:
                time_line[time_point][f"{time_dist}_month_effect"] = np.nan
        return time_line

    @property
    def time_series(self):
        time_series = []
        # iterate through months and retrieve measurement vector
        months = list(self.time_line.keys())
        for month in months:
            time_series.append(self.time_line[month]["total_fluid"])
        return np.array(time_series)

    @property
    def study_dates(self):
        study_dates = []
        # iterate through months and retrieve measurement vector
        months = list(self.time_line.keys())
        for month in months:
            study_dates.append(self.time_line[month]["study_date"])
        return np.array(study_dates)

    @property
    def segmentation_paths(self):
        laterality_cond = "_" + (self.laterality) + "_"
        patient_cond = str(self.patient_id) + "_"
        segmentation_files = list(
            filter(lambda k: (patient_cond in k) & (laterality_cond in k), MeasureSeqTimeUntilDry.SEG_PATHS))
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
                plot_segmentation_map(map[idx, :, :], show = False, save_path = os.path.join(WORK_SPACE, "dump"),
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

                indices = np.linspace(0, map.shape[0] - 1, MeasureSeqTimeUntilDry.NUM_SEGMENTATIONS, dtype = np.int32)
                self.time_line[month]["segmentation_maps"] = map[indices, :, :]

    def show_time_series(self, show_segmentations=False, show=False, save_fig=False):
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

        # set y axis params
        if np.max(time_series) < 50000:
            y_max = 50000
        else:
            y_max = np.max(time_series)

        fig, ax = plt.subplots(figsize = (int(time_series.shape[0]), 10))
        fig.subplots_adjust(bottom = 0.4)

        xs = np.arange(0, time_series.shape[0], 1)
        ys = time_series

        # plot points
        ax.plot(xs, ys, "bo-")

        x_offset = 0
        xy_box = {0: (x_offset, y_max // 5),
                  1: (x_offset, (y_max // 5) * 2),
                  2: (x_offset, (y_max // 5) * 3)}

        # set image zoom by time series length
        if (time_series.shape[0] < 15) & (time_series.shape[0] > 6):
            zoom = 2 / time_series.shape[0]

        if (time_series.shape[0] > 15) & (time_series.shape[0] <= 25):
            zoom = 3 / time_series.shape[0]

        if (time_series.shape[0] > 25):
            zoom = 4 / time_series.shape[0]

        if (time_series.shape[0] > 40):
            zoom = 6 / time_series.shape[0]

        if (time_series.shape[0] <= 6):
            zoom = 1 / time_series.shape[0]

        # zip joins x and y coordinates in pairs
        for x, y in zip(xs, ys):
            if not self.time_line[x + 1]['cur_va_rounded'] is np.nan:
                label = f"Inj: {self.time_line[x + 1]['total_injections']} \n VA: {self.time_line[x + 1]['cur_va_rounded']}"
            else:
                label = ""
            ax.text(x, y + 1000, label, fontdict = font)

            if show_segmentations & (self.time_line[x + 1]["study_date"] is not np.nan):
                for i in range(MeasureSeqTimeUntilDry.NUM_SEGMENTATIONS):
                    imagebox = OffsetImage(self.time_line[x + 1]["segmentation_maps"][i, :, :],
                                           zoom = zoom, cmap = seg_cmap, norm = seg_norm)
                    imagebox.image.axes = ax
                    ab = AnnotationBbox(imagebox, (x, 0), xybox = (x + xy_box[i][0], 0 - xy_box[i][1]),
                                        frameon = False)
                    ax.add_artist(ab)

        plt.ylim(-100, y_max)
        plt.xlabel("month")
        plt.ylabel("total fluid")
        plt.legend(handles = [darkred_patch])

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
    # load sequences
    seq_pd = pd.read_csv(os.path.join(WORK_SPACE, 'sequences.csv'))

    PATIENT_ID = 1570  # 2005
    LATERALITY = "L"

    filter_ = (seq_pd.patient_id == PATIENT_ID) & (seq_pd.laterality == LATERALITY)
    # seq_pd = seq_pd.loc[filter_]

    unique_records = seq_pd[["patient_id", "laterality"]].drop_duplicates() # .iloc[0:10]

    time_until_dry = []
    for patient, lat in tqdm(unique_records.itertuples(index = False)):
        record_pd = seq_pd[(seq_pd.patient_id == patient) & (seq_pd.laterality == lat)]
        time_until_dry.append(MeasureSeqTimeUntilDry.from_record(record_pd))
        # time_until_dry[-1].show_time_series(show_segmentations = True, show = True, save_fig = True)
        # time_until_dry[-1].dump_segmentation_map(5)

    time_serie_log = {"patient_id": [], "laterality": [], "time_until_dry": [], "number_of_injections": [],
                      "three_month_effect": [], "six_month_effect": [], "number_of_months": []}

    for measurem in time_until_dry:
        for key in time_serie_log.keys():
            time_serie_log[key].append(measurem.__dict__[key])

    time_until_dry_pd = pd.DataFrame(time_serie_log)
    time_until_dry_pd.to_csv(os.path.join(WORK_SPACE, "time_until_dry.csv"))