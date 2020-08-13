from copy import copy, deepcopy

from feature_statistics.config import WORK_SPACE
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
from utils.util_functions import nan_helper
import datetime
from tqdm import tqdm
import LSTM.sequences as sequences  # <- this contains the custom code
workspace_dir = WORK_SPACE


class MeasureSeqTimeUntilDry:
    NUMBER_OF_MONTHS = 3
    REGIONS = ["S", "N", "I", "T"]
    FLUIDS = ["3", "4"]
    TIME_LINE = dict((k, {}) for k in range(1, NUMBER_OF_MONTHS + 1))

    def __init__(self, time_line, naive, time_until_dry):
        self.time_line = time_line
        self.naive = naive
        self.time_until_dry = time_until_dry

    @classmethod
    def from_record(cls, record_table):
        """

        """
        # reset index of table
        record_table.reset_index(inplace=True)

        # add total fluid
        record_table = cls.get_total_fluid(table = record_table)

        # assign each recording to a time point
        time_line = deepcopy(MeasureSeqTimeUntilDry.TIME_LINE)

        # sum fluid for each time point
        time_line = cls.assign_to_timeline(time_line, record_table, item = "study_date")
        time_line = cls.assign_to_timeline(time_line, record_table, item = "total_fluid")

        # interpolate time vector
        time_line = cls.interpolate(time_line)

        # calculate time until dry, -1 means not dry
        time_ = cls.get_time_until_dry(time_line)

        # check if naive record
        naive = cls.is_naive(record_table)
        return cls(time_line = time_line, naive = naive, time_until_dry = time_)

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
    def get_total_fluid(cls, table):
        fluid = pd.Series(np.zeros(table.shape[0]))
        for region_header in MeasureSeqTimeUntilDry.REGIONS:
            for i in range(1, 3, 1):
                region_intra_fluid = region_header + f"{i}_{3}"
                region_sub_fluid = region_header + f"{i}_{4}"

                fluid += table[region_intra_fluid]
                fluid += table[region_sub_fluid]
        # add foveal fluid
        fluid += table["C0_3"] + table["C0_4"]
        table.insert(10, "total_fluid", fluid.values.tolist(), True)
        return table

    @classmethod
    def interpolate(cls, time_line):
        interp_vector = []

        # iterate through months and retrieve measurement vector
        months = list(time_line.keys())
        for month in months:
            interp_vector.append(time_line[month]["total_fluid"])

        # interpolate missing values
        interp_array = np.array(interp_vector)
        nans, x = nan_helper(interp_array)
        interp_array[nans] = np.interp(x(nans), x(~nans), interp_array[~nans])

        # assign interp values
        for i, month in enumerate(months):
            time_line[month]["total_fluid"] = interp_array[i]
        return time_line

    @classmethod
    def assign_to_timeline(cls, time_line, table, item):
        """
        time_line, dict: dict of dicts: containing 12 months recording
        table, DataFrame: record data frame
        item, str: column from table to insert into time line
        """
        DAYS = 30
        # assign first & last observation
        time_line[1].update({item: table[item].iloc[0]})
        time_line[MeasureSeqTimeUntilDry.NUMBER_OF_MONTHS].update({item: table[item].iloc[-1]})

        __start__ = datetime.datetime.strptime(table["study_date"].iloc[0], '%Y-%m-%d')
        __end__ = datetime.datetime.strptime(table["study_date"].iloc[-1], '%Y-%m-%d')

        one_month_timestamps = []
        date_x = __start__
        while date_x < __end__:
            date_x += timedelta(days = DAYS)
            if not (__end__ - date_x) < timedelta(days = DAYS):
                one_month_timestamps.append(date_x)

        anchor_dates = np.array(one_month_timestamps)
        candidate_dates = pd.to_datetime(table["study_date"])
        for k, date in enumerate(anchor_dates, 2):
            time_delta = candidate_dates - date
            candidates = [td for td in time_delta if (td.days < 15) & (td.days > -15)]

            if len(candidates) > 1:
                min_diff = np.min(candidates)
                idx_select = np.where(time_delta == min_diff)[0][0]
                time_line[k].update({item: table[item].iloc[idx_select]})
            elif not candidates:
                time_line[k].update({item: np.nan})
                continue
            else:
                idx_select = np.where(time_delta == candidates[0])[0][0]
                time_line[k].update({item: table[item].iloc[idx_select]})
        return time_line

    @classmethod
    def filter_shortvisists(cls, table):
        # filter short visits
        short_visits = table.delta_t < 20
        short_visits[0] = False
        table_filtered = table[~short_visits]

        # reset index after filtering
        table_filtered.reset_index(inplace=True)
        return table_filtered

    def get_time_series(self):
        time_series = []
        # iterate through months and retrieve measurement vector
        months = list(self.time_line.keys())
        for month in months:
            time_series.append(self.time_line[month]["total_fluid"])
        return np.array(time_series)


# load sequences
# sequences_checkup_3 = sequences.load_sequences_from_pickle(os.path.join(workspace_dir, 'sequences_3_test.pickle'))
seq_pd = pd.read_csv(os.path.join(workspace_dir, 'sequences_3.csv'))
unique_records = seq_pd[["patient_id", "laterality"]].drop_duplicates()

time_until_dry = []
for patient, lat in tqdm(unique_records.itertuples(index=False)):
    record_pd = seq_pd[(seq_pd.patient_id == patient) & (seq_pd.laterality == lat)]
    time_until_dry.append(MeasureSeqTimeUntilDry.from_record(record_pd))