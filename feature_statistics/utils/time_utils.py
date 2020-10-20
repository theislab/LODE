import math
from copy import deepcopy

import numpy as np
import datetime
import pandas as pd
from pprint import pprint


class TimeUtils:
    """

    """
    DAYS = 30

    def __init__(self, record_table):
        self.table = record_table
        self.time_steps = [0, 3, 6, 12, 24]
        self.NUMBER_OF_MONTHS = max(self.time_steps) # self.set_number_of_months
        self.time_line = dict((k, {}) for k in range(1, self.NUMBER_OF_MONTHS + 1))
        self.excluded_months = []
        self.cut_time_series = False
        self.DAYS = TimeUtils.DAYS

    @property
    def set_number_of_months(self):
        months = pd.to_datetime(self.table.study_date)
        time_span = months.iloc[-1] - months.iloc[0]

        time_divisor = int(np.round(time_span.days / TimeUtils.DAYS, 0)) + 1  # math.round(time_span.days / 30, 2) + 1
        if time_divisor == 1:
            return 2
        else:
            return time_divisor

    def first_corner_case(self, time_line):
        # edge case, injections given in candidate range of first visit
        candidate_dates = pd.to_datetime(self.table["study_date"].iloc[1:])
        time_delta = candidate_dates - pd.to_datetime(time_line[1]["study_date"])
        candidates = [td for td in time_delta if (td.days < TimeUtils.DAYS / 2) & (td.days >= -TimeUtils.DAYS / 2)]

        if candidates:
            # go through all possible candidates
            for candidate in candidates:
                closest_idx = np.argwhere(time_delta == candidate)[0][0]
                injections_ = self.table.iloc[closest_idx + 1]["injections"]

                # if injections in first neighbour is more than zero, assign to first date
                if injections_ > 0:
                    time_line[1]["injections"] = injections_

        return time_line

    def last_corner_case(self, time_line):
        # edge case, injections given in candidate range of first visit
        candidate_dates = pd.to_datetime(self.table["study_date"].iloc[:-1])
        time_delta = candidate_dates - pd.to_datetime(time_line[max(time_line.keys())]["study_date"])
        candidates = [td for td in time_delta if (td.days < TimeUtils.DAYS / 2) & (td.days >= -TimeUtils.DAYS / 2)]

        if candidates:
            closest_idx = np.argwhere(time_delta == candidates[0])[0][0]
            injections_ = self.table.iloc[closest_idx + 1]["injections"]

            # if injections in first neighbour is more than zero, assign to first date
            if injections_ > 0:
                time_line[1]["injections"] = injections_

        return time_line

    def get_marks(self, first_mark):
        treatment_marks = []
        for mark in self.time_steps[1:]:
            time_mark = first_mark
            time_mark += datetime.timedelta(days = int(TimeUtils.DAYS * mark))
            treatment_marks.append(time_mark)
        return treatment_marks

    def treatment_table(self):
        first_injection_idx = min(np.argwhere(self.table["injections"] > 0))[0]
        treatment_table = self.table.iloc[first_injection_idx:]

        # reset index to start from 1
        treatment_table.index = treatment_table.index - treatment_table.index[0] + 1

        return treatment_table

    def get_first_time_point(self, item, time_line, treatment_table):
        # assign data point to time line
        first_point = treatment_table.index[0]
        first_injection_date = treatment_table.iloc[0].study_date

        # assign item to time line
        time_line[first_point][item] = treatment_table.iloc[0][item]

        # crate treatment marks
        first_mark = pd.to_datetime(first_injection_date)
        return first_mark, time_line

    def assign_to_timeline_str(self, time_line, item, total_number_injections):
        """
        @param time_line: number of months in time period of patients visits
        @type time_line: dict
        @param table: data frame with all meta of patients visit
        @type table: DataFrame
        @param item: name of item to assign to correct month, e.g. total_fluid
        @type item: str
        @return: update timeline with added information from data frame
        @rtype: dict
        """
        if total_number_injections > 0:
            treatment_table = self.treatment_table()
        else:
            treatment_table = deepcopy(self.table)
            # reset index to start from 1
            treatment_table.index = treatment_table.index - treatment_table.index[0] + 1

        first_mark, time_line = self.get_first_time_point(item, time_line, treatment_table)
        treatment_marks = self.get_marks(first_mark)

        for k, mark in enumerate(treatment_marks, 0):
            deltas = mark - pd.to_datetime(treatment_table.study_date)
            candidate_idx = np.argmin(np.array(np.abs(deltas))) + 1

            if deltas[candidate_idx].days <= TimeUtils.DAYS / 2:
                time_line[self.time_steps[k + 1]][item] = treatment_table.loc[candidate_idx][item]
            else:
                # assign None to time line
                time_line[self.time_steps[k + 1]][item] = None
        return time_line

    def assign_date_to_timeline_dec(self, time_line, item, total_number_injections):
        """
        @param time_line: number of months in time period of patients visits
        @type time_line: dict
        @param table: data frame with all meta of patients visit
        @type table: DataFrame
        @param item: name of item to assign to correct month, e.g. total_fluid
        @type item: str
        @return: update timeline with added information from data frame
        @rtype: dict
        """
        if total_number_injections > 0:
            treatment_table = self.treatment_table()
        else:
            treatment_table = deepcopy(self.table)
            # reset index to start from 1
            treatment_table.index = treatment_table.index - treatment_table.index[0] + 1

        first_mark, time_line = self.get_first_time_point(item, time_line, treatment_table)
        treatment_marks = self.get_marks(first_mark)

        for k, mark in enumerate(treatment_marks, 0):
            added_dates = [time_line[key]['study_date'] for key in time_line.keys() if
                           'study_date' in time_line[key].keys()]

            deltas = mark - pd.to_datetime(treatment_table.study_date)
            candidate_idxs = treatment_table.study_date[deltas.values.astype("float") >= 0].tolist()

            for added_date in added_dates:
                if added_date is not None:
                    if added_date in candidate_idxs:
                        candidate_idxs.remove(added_date)

            if candidate_idxs:
                for date in candidate_idxs:
                    item_value = treatment_table.loc[treatment_table.study_date == date][item].values[0]
                    time_line[self.time_steps[k + 1]][item] = item_value
            else:
                # assign None to time line
                time_line[self.time_steps[k + 1]][item] = None
        return time_line

    def assign_features_to_timeline_dec(self, time_line, item, total_number_injections):
        """
        @param time_line: number of months in time period of patients visits
        @type time_line: dict
        @param table: data frame with all meta of patients visit
        @type table: DataFrame
        @param item: name of item to assign to correct month, e.g. total_fluid
        @type item: str
        @return: update timeline with added information from data frame
        @rtype: dict
        """
        if total_number_injections > 0:
            treatment_table = self.treatment_table()
        else:
            treatment_table = deepcopy(self.table)
            # reset index to start from 1
            treatment_table.index = treatment_table.index - treatment_table.index[0] + 1

        added_dates = [time_line[key]['study_date'] for key in time_line.keys() if
                       'study_date' in time_line[key].keys()]

        for date in added_dates:
            item_value = treatment_table.loc[treatment_table.study_date == date][item].values[0]
            dict_key = [key for key in time_line.keys() if time_line[key]["study_date"] == date]
            time_line[dict_key][item] = item_value

    def assign_to_timeline(self, time_line, item):
        """
        @param time_line: number of months in time period of patients visits
        @type time_line: dict
        @param table: data frame with all meta of patients visit
        @type table: DataFrame
        @param item: name of item to assign to correct month, e.g. total_fluid
        @type item: str
        @return: update timeline with added information from data frame
        @rtype: dict
        """
        # assign first & last observation
        time_line[1].update({item: self.table[item].iloc[0]})
        time_line[self.NUMBER_OF_MONTHS].update({item: self.table[item].iloc[-1]})
        time_line[self.NUMBER_OF_MONTHS]["injections"] = self.table.iloc[-1].injections

        # control injection logging for filtered candidate months close to first and last
        time_line = self.first_corner_case(time_line)
        time_line = self.last_corner_case(time_line)

        __start__ = datetime.datetime.strptime(self.table["study_date"].iloc[0], '%Y-%m-%d')
        __end__ = datetime.datetime.strptime(self.table["study_date"].iloc[-1], '%Y-%m-%d')

        one_month_timestamps = []
        date_x = __start__
        while date_x < __end__:
            date_x += datetime.timedelta(days = TimeUtils.DAYS)
            # if not (__end__ - date_x) < timedelta(days = MeasureSeqTimeUntilDry.DAYS):
            one_month_timestamps.append(date_x)

        anchor_dates = np.array(one_month_timestamps)
        candidate_dates = pd.to_datetime(self.table["study_date"])
        for k, date in enumerate(anchor_dates, 2):
            if k <= self.NUMBER_OF_MONTHS:
                add_to_next_injections = 0
                add_to_prev_injections = 0

                time_delta = candidate_dates - date
                candidates = [td for td in time_delta if (td.days < TimeUtils.DAYS / 2) & (td.days >= -TimeUtils.DAYS / 2)]
                if len(candidates) > 1:
                    # assign closest of candidates
                    min_diff = np.min(candidates)
                    idx_select = np.where(time_delta == min_diff)[0][0]
                    time_line[k].update({item: self.table[item].iloc[idx_select]})

                    # if adding injections, assign close by removed studies to neighbouring visits
                    if item == "injections":
                        # remove selected candidate
                        candidates.remove(min_diff)

                        # if candidates are discarded, assign injection event to time line
                        for candidate in candidates:
                            idx_not_select = np.where(time_delta == candidate)[0][0]
                            if candidate.days > 0:
                                add_to_next_injections = self.table["injections"].iloc[idx_not_select]
                            else:
                                add_to_prev_injections = self.table["injections"].iloc[idx_not_select]

                            # update neighbouring time lines

                            # get last added month in time line
                            for month in time_line.keys():
                                if (time_line[month]["study_date"] is not np.nan) & (month < k):
                                    last_added_month = month

                            # add previous and/or next value with the skipped injection data point
                            time_line[last_added_month][item] = time_line[last_added_month][item] \
                                                                + add_to_prev_injections

                            try:
                                # if not last month add recorded injections
                                if (idx_not_select + 1) < self.table.shape[0]:
                                    self.table.loc[idx_not_select + 1, item] = self.table.loc[idx_not_select + 1, item] \
                                                                               + add_to_next_injections
                            except:
                                print("stop")

                elif (not candidates) & (k != list(time_line.keys())[-1]):
                    time_line[k].update({item: np.nan})
                    continue

                elif candidates:
                    idx_select = np.where(time_delta == candidates[0])[0][0]
                    time_line[k].update({item: self.table[item].iloc[idx_select]})
        return time_line
