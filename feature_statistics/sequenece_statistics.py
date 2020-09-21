import math
from copy import deepcopy

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
from tqdm import tqdm
import glob
import matplotlib.patches as mpatches
import seaborn as sns

class MeasureSeqGeneral(SeqUtils):
    NUMBER_OF_MONTHS = 13
    NUM_SEGMENTATIONS = 3
    DAYS = 30
    REGIONS = ["S", "N", "I", "T"]
    FLUIDS = ["3", "4"]
    DATA_POINTS = ["study_date", "total_fluid", "avg_atrophy", "total_injections", "cur_va_rounded", "next_va"]
    META_DATA = ["patient_id", "laterality", "diagnosis"]
    SEG_PATHS = glob.glob(os.path.join(SEG_DIR, "*"))

    def __init__(self, meta_data, time_line, naive, number_of_injections):
        self.patient_id = meta_data[MeasureSeqGeneral.META_DATA[0]]
        self.laterality = meta_data[MeasureSeqGeneral.META_DATA[1]]
        self.diagnosis = meta_data[MeasureSeqGeneral.META_DATA[2]]
        self.time_line = time_line
        self.number_of_months = len(time_line)
        self.naive = naive
        self.number_of_visits = self.get_number_of_visits()
        self.number_of_injections = number_of_injections
        self.fluid_deltas = self.month_feature_deltas(self.time_line, feature = "total_fluid",
                                                      after_injections = False)
        self.fluid_deltas_treated = self.month_feature_deltas(self.time_line, feature = "total_fluid",
                                                              after_injections = True)

        self.atrophy_deltas = self.month_feature_deltas(self.time_line, feature = "avg_atrophy",
                                                        after_injections = False)
        self.atrophy_deltas_treated = self.month_feature_deltas(self.time_line, feature = "avg_atrophy",
                                                                after_injections = True)

        self.fluid_std = self.sequence_feature_std(time_line, "total_fluid")
        self.atrophy_std = self.sequence_feature_std(time_line, "avg_atrophy")

    @classmethod
    def from_record(cls, record_table):
        time_utils = TimeUtils(record_table = record_table)
        time_line = time_utils.time_line

        # initalize sequence class
        super().__init__(SeqUtils, time_line)

        # reset index of table
        record_table.reset_index(inplace = True)

        # add total fluid
        record_table = sum_etdrs_columns(record_table, rings = [1, 2], regions = MeasureSeqGeneral.REGIONS,
                                         features = [3, 4], foveal_region = ["C0"], new_column_name = "total_fluid")

        record_table = avg_etdrs_columns(record_table, rings = [1, 2], regions = MeasureSeqGeneral.REGIONS,
                                         features = ["atropy_percentage"], foveal_region = ["C0"],
                                         new_column_name = "avg_atrophy")

        # round va
        record_table.insert(loc = 10, column = "cur_va_rounded", value = record_table.cur_va.round(2))

        # add total injections
        record_table = get_total_number_of_injections(table = record_table)

        # assign items to time line
        for data_point in MeasureSeqGeneral.DATA_POINTS:
            time_line = time_utils.assign_to_timeline(time_line = time_line, item = data_point)

        # interpolate time vector
        for item in ["total_fluid", "avg_atrophy"]:
            time_line = interpolate_numeric_field(time_line, item = item)

        # check if naive record
        naive = SeqUtils.is_naive(record_table)

        # get number of injections total
        number_of_injections = SeqUtils.get_seq_number_of_injections(time_line)
        return cls(meta_data = record_table.iloc[0],
                   time_line = time_line,
                   naive = naive,
                   number_of_injections = number_of_injections)

    def month_feature_deltas(self, time_line, feature, after_injections):
        """
        @param time_line:
        @type time_line:
        @param after_injections:
        @type after_injections:
        @return:
        @rtype:
        """
        diff1 = np.array([time_line[month][feature] for month in time_line.keys()])
        deltas = np.diff(diff1)

        # remove all non treated deltas
        deltas_temp = []
        if after_injections:
            for month in list(time_line.keys())[:-1]:
                if (time_line[month]["total_injections"] != 0) & (time_line[month]["total_injections"] != np.nan):
                    deltas_temp.append(deltas[month - 1])
            return deltas_temp
        return deltas

    def sequence_feature_std(self, time_line, feature):
        """
        @param time_line:
        @type time_line:
        @param after_injections:
        @type after_injections:
        @return:
        @rtype:
        """
        time_series = np.array([time_line[month][feature] for month in time_line.keys()])
        return np.std(time_series)

    @property
    def time_series(self):
        time_series = []
        # iterate through months and retrieve measurement vector
        months = list(self.time_line.keys())
        for month in months:
            time_series.append(self.time_line[month]["total_fluid"])
        return np.array(time_series)

    @property
    def segmentation_paths(self):
        laterality_cond = "_" + (self.laterality) + "_"
        patient_cond = str(self.patient_id) + "_"
        segmentation_files = list(
            filter(lambda k: (patient_cond in k) & (laterality_cond in k), MeasureSeqGeneral.SEG_PATHS))
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


def delta_plot(delta_dict, keys, title):
    for key in keys:
        sns.distplot(delta_dict[key], kde = False, label = key)
    # Plot formatting
    plt.legend(prop = {'size': 12})
    plt.title(title)
    plt.xlabel(keys[0])
    plt.ylabel('number of sequences')
    plt.savefig(os.path.join(SAVE_DIR, f"{title}.png"))
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

    PATIENT_ID = 2005  # 2005
    LATERALITY = "L"

    filter_ = (seq_pd.patient_id == PATIENT_ID) & (seq_pd.laterality == LATERALITY)
    # seq_pd = seq_pd.loc[filter_]

    unique_records = seq_pd[["patient_id", "laterality"]].drop_duplicates()

    time_until_dry = []
    for patient, lat in tqdm(unique_records.itertuples(index = False)):
        try:
            record_pd = seq_pd[(seq_pd.patient_id == patient) & (seq_pd.laterality == lat)]
            time_until_dry.append(MeasureSeqGeneral.from_record(record_pd))
        except:
            print(patient, lat)

    time_serie_log = {
        "patient_id": [],
        "laterality": [],
        "number_of_injections": [],
        "number_of_months": [],
        "diagnosis": [],
        "number_of_visits": [],
        "naive": [],
        "fluid_deltas": [],
        "fluid_deltas_treated": [],
        "atrophy_deltas": [],
        "atrophy_deltas_treated": [],
        "fluid_std": [],
        "atrophy_std": []
    }
    SAVE_DIR = os.path.join(WORK_SPACE, "plots")

    atrophy_deltas = []
    atrophy_deltas_treated = []
    fluid_delta = []
    fluid_delta_treated = []
    fluid_std = []
    atrophy_std = []

    for measurem in time_until_dry:
            atrophy_deltas = atrophy_deltas + measurem.atrophy_deltas.tolist()
            atrophy_deltas_treated = atrophy_deltas_treated + measurem.atrophy_deltas_treated
            fluid_delta = atrophy_deltas + measurem.fluid_deltas.tolist()
            fluid_delta_treated = fluid_delta_treated + measurem.fluid_deltas_treated
            fluid_std.append(measurem.fluid_std)
            atrophy_std.append(measurem.atrophy_std)

    delta_dict = {"atrophy_delta": atrophy_deltas, "atrophy_delta_treated": atrophy_deltas_treated,
                  "fluid_delta": fluid_delta, "fluid_delta_treated": fluid_delta_treated}

    atrophy_keys = ["atrophy_delta", "atrophy_delta_treated"]
    fluid_keys = ["fluid_delta", "fluid_delta_treated"]

    # get fluid distribution
    delta_plot(delta_dict, atrophy_keys, "atrophy_distribution")
    delta_plot(delta_dict, fluid_keys, "fluid_distribution")

    time_until_dry_pd = pd.DataFrame(time_serie_log)
    time_until_dry_pd.to_csv(os.path.join(WORK_SPACE, "sequence_data/statistics.csv"))
