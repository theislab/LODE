from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pydicom import read_file
from feature_statistics.config import WORK_SPACE, SEG_DIR, OCT_DIR
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from feature_statistics.utils.time_utils import TimeUtils
from feature_statistics.utils.pandas_utils import sum_etdrs_columns, interpolate_numeric_field
from feature_statistics.utils.util_functions import SeqUtils, get_total_number_of_injections, \
    get_first_month_injection, get_treatment_time_line, get_delta_logs
from feature_statistics.utils.plotting_utils import plot_segmentation_map, color_mappings
from tqdm import tqdm
import glob
import matplotlib.patches as mpatches


deltas = ["1-3", "1-6", "1-12", "3-6", "6-12"]


class MeasureSeqTimeUntilDry(SeqUtils):
    NUMBER_OF_MONTHS = 13
    NUM_SEGMENTATIONS = 4
    DAYS = 30
    REGIONS = ["S", "N", "I", "T"]
    FLUIDS = ["3", "4"]
    DATA_POINTS = ["study_date", "total_fluid", "injections", "cur_va_rounded", "next_va"]
    META_DATA = ["patient_id", "laterality", "diagnosis"]
    SEG_PATHS = glob.glob(os.path.join(SEG_DIR, "*"))
    DICOM_PATHS = glob.glob(os.path.join(OCT_DIR, "*/*/*/*.dcm"))
    EXCLUDED_MONTHS = []
    NUMBER_OF_CUT_SERIES = []

    def __init__(self, meta_data, time_line, fluid_time_line, naive):
        self.patient_id = meta_data[MeasureSeqTimeUntilDry.META_DATA[0]]
        self.laterality = meta_data[MeasureSeqTimeUntilDry.META_DATA[1]]
        self.diagnosis = meta_data[MeasureSeqTimeUntilDry.META_DATA[2]]
        self.time_line = time_line
        self.fluid_time_line = fluid_time_line
        self.naive = naive

        if self.fluid_time_line:
            self.abs_, self.inj_ = get_delta_logs(self.fluid_time_line, deltas)

    @classmethod
    def from_record(cls, record_table):
        time_utils = TimeUtils(record_table = record_table)
        time_line = time_utils.time_line
        fluid_time_line = None

        # initalize sequence class
        super().__init__(SeqUtils, time_line = time_line)

        # reset index of table
        record_table.reset_index(inplace = True)

        # add total fluid
        record_table = sum_etdrs_columns(record_table, rings = [1, 2], regions = MeasureSeqTimeUntilDry.REGIONS,
                                         features = [3, 4], foveal_region = ["C0"], new_column_name = "total_fluid")

        # round va
        record_table.insert(loc = 10, column = "cur_va_rounded", value = record_table.cur_va.round(2))

        # add total injections
        total_number_injections = get_total_number_of_injections(table = record_table)

        # assign items to time line
        for data_point in MeasureSeqTimeUntilDry.DATA_POINTS:
            time_line = time_utils.assign_to_timeline(time_line = time_line, item = data_point)

        # interpolate time vector
        for item in ["total_fluid"]:
            time_line = interpolate_numeric_field(time_line, item = item)

        # see soo any injections have been administrered
        if total_number_injections > 0:
            # get first injection date and month
            first_inj_date, first_inj_month = get_first_month_injection(time_line)

            # see so at least one 3 month checkup is available
            if time_utils.set_number_of_months > (first_inj_month + 3):
                fluid_time_line = get_treatment_time_line(time_line, first_month = first_inj_month)

        # check if naive record
        naive = SeqUtils.is_naive(record_table)

        return cls(meta_data = record_table.iloc[0],
                   time_line = time_line,
                   fluid_time_line = fluid_time_line,
                   naive = naive)

    @property
    def time_series(self):
        time_series = []
        # iterate through months and retrieve measurement vector
        months = list(self.fluid_time_line.keys())
        for month in months:
            time_series.append(self.fluid_time_line[month]["total_fluid"])
        return np.array(time_series)

    @property
    def segmentation_paths(self):
        laterality_cond = "_" + (self.laterality) + "_"
        patient_cond = str(self.patient_id) + "_"
        segmentation_files = list(
            filter(lambda k: (patient_cond in k) & (laterality_cond in k), MeasureSeqTimeUntilDry.SEG_PATHS))
        return segmentation_files

    @property
    def dicom_paths(self):
        laterality_cond = self.laterality
        patient_cond = str(self.patient_id)
        dicom_files = list(
            filter(lambda k: (patient_cond in k) & (laterality_cond in k), MeasureSeqTimeUntilDry.DICOM_PATHS))
        return dicom_files

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
                                      save_path = os.path.join(WORK_SPACE, "dump/2d_segmentations"),
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

    def add_oct_to_timeline(self):
        """
        Assigns segmented oct volumes to time line
        @return: None
        @rtype:
        """
        months = list(self.time_line.keys())
        for month in months:
            if self.time_line[month]["study_date"] is np.nan:
                self.time_line[month]["octs"] = np.nan
                continue
            else:
                date = self.time_line[month]["study_date"].replace("-", "")
                path = list(filter(lambda k: (date.replace("-", "") in k), self.dicom_paths))[0]
                oct_ = read_file(path).pixel_array

                indices = np.linspace(0, oct_.shape[0] - 1, MeasureSeqTimeUntilDry.NUM_SEGMENTATIONS, dtype = np.int32)
                self.time_line[month]["octs"] = oct_[indices, :, :]

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
            self.add_oct_to_timeline()

        plt.style.use('ggplot')

        darkred_patch = mpatches.Patch(color = 'darkorange', label = 'injections')
        time_series = self.time_series

        if max(time_series) < 1:
            y_max = 1
        else:
            y_max = 3

        fig, ax = plt.subplots(figsize = (int(time_series.shape[0]), 2))
        fig.subplots_adjust(top = 0.4)

        xs = [1, 3, 6, 12]
        ys = time_series

        # plot points
        ax.plot(xs, ys, "bo-")

        x_offset = 0
        xy_box = {0: (x_offset, y_max // 5),
                  1: (x_offset, (y_max // 5) * 2),
                  2: (x_offset, (y_max // 5) * 3)}

        # zip joins x and y coordinates in pairs
        for x, y in zip(xs, ys):
            if not self.time_line[x + 1]['cur_va_rounded'] is np.nan:
                label = f"Inj: {self.time_line[x + 1]['injections']} \n VA: {self.time_line[x + 1]['cur_va_rounded']}"
            else:
                label = ""
            ax.text(x, y + 0.01, label, fontdict = font)

            if show_segmentations & (self.time_line[x + 1]["study_date"] is not np.nan):
                for i in range(MeasureSeqTimeUntilDry.NUM_SEGMENTATIONS):
                    imagebox = OffsetImage(self.time_line[x + 1]["segmentation_maps"][i, :, :],
                                            cmap = seg_cmap, norm = seg_norm)
                    imagebox.image.axes = ax
                    ab = AnnotationBbox(imagebox, (x, 0), xybox = (x + xy_box[i][0], 0 - xy_box[i][1]),
                                        frameon = False)
                    ax.add_artist(ab)

        plt.ylim(-y_max, y_max)
        plt.xlabel("month")
        plt.ylabel("effect")
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
    """
    questions:
    # distribution of sequence lengths & sequence time span
    # number of sequences with injections
    # number of sequences with diff diagnosises
    # distribution of time until dry
    # distribution of 3 and 6 month treatment effect
    """
    # load sequences
    seq_pd = pd.read_csv(os.path.join(WORK_SPACE, "sequence_data", 'sequences.csv'))

    PATIENT_ID = 1557  # 2005
    LATERALITY = "R"

    filter_ = (seq_pd.patient_id == PATIENT_ID) & (seq_pd.laterality == LATERALITY)
    # seq_pd = seq_pd.loc[filter_]

    unique_records = seq_pd[["patient_id", "laterality"]].drop_duplicates()

    time_until_dry = []
    for patient, lat in tqdm(unique_records.itertuples(index = False)):
        print(patient, lat)
        record_pd = seq_pd[(seq_pd.patient_id == patient) & (seq_pd.laterality == lat)]
        time_until_dry.append(MeasureSeqTimeUntilDry.from_record(record_pd))
        # time_until_dry[-1].show_time_series(show_segmentations = True, show = True, save_fig = True)
        # time_until_dry[-1].dump_segmentation_map(11)

    time_series_log = {"patient_id": [],
                       "laterality": []}

    # add the
    for delta in deltas:
        time_series_log[delta + "abs"] = []
        time_series_log[delta + "inj"] = []

    for measurem in time_until_dry:
        if "abs_" in measurem.__dict__.keys():
            for key in list(time_series_log.keys())[0:2]:
                time_series_log[key].append(measurem.__dict__[key])

            # extract dictionaries
            abs_dict = measurem.__dict__["abs_"]
            inj_dict = measurem.__dict__["inj_"]

            # add delta values
            for key in abs_dict.keys():
                time_series_log[key].append(abs_dict[key])

            for key in inj_dict.keys():
                time_series_log[key].append(inj_dict[key])


    # time_until_dry_pd = pd.DataFrame(time_series_log)
    # time_until_dry_pd.to_csv(os.path.join(WORK_SPACE, "sequence_data/time_until_dry.csv"))