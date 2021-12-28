import math
from copy import deepcopy

import numpy as np
import datetime
import pandas as pd

from utils.util_functions import nan_helper


class TimeUtils:
    """

    """
    DAYS = 30

    def __init__(self, record_table):
        self.table = record_table
        self.time_steps = [1, 3, 6, 12, 24]
        self.NUMBER_OF_MONTHS = max(self.time_steps)  # self.set_number_of_months
        self.time_line = dict((k, {}) for k in self.time_steps)
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

    def interpolate_numeric_field(self, start_, end_, start_value, end_value):
        """
        Parameters
        ----------
        start_ :
        end_ :
        start_value :
        end_value :

        Returns
        -------

        """
        interp_vector = []

        # iterate through months and retrieve measurement vector
        for i in range(start_, end_ + 1):
            if i == start_:
                interp_vector.append(start_value)
            elif i == end_:
                interp_vector.append(end_value)
            else:
                interp_vector.append(np.nan)

        # interpolate missing values
        interp_array = np.array(interp_vector)
        nans, x = nan_helper(interp_array)
        interp_array[nans] = np.interp(x(nans), x(~nans), interp_array[~nans])
        return interp_array

    def interpolate(self, time_deltas, table, item="total_fluid"):
        """

        Parameters
        ----------
        table :
        item :
        time_deltas :

        Returns
        -------
        interpolated feature value, time range, time span before and after time point
        """

        days = time_deltas.astype('timedelta64[D]').values.astype(int)
        signs = np.sign(days)

        date_idx_before = max(np.argwhere(signs == 1))
        date_idx_after = min(np.argwhere(signs == -1))

        data_before = table.iloc[date_idx_before]
        data_after = table.iloc[date_idx_after]

        date_0 = data_before.study_date.values[0]
        date_1 = data_after.study_date.values[0]

        delta_days = pd.to_datetime(date_1) - pd.to_datetime(date_0)
        start_feature_value = table[item].iloc[date_idx_before].values[0]
        end_feature_value = table[item].iloc[date_idx_after].values[0]

        days_before = abs(days[date_idx_before[0]])
        days_after = abs(days[date_idx_after[0]])

        if np.isnan(start_feature_value) or np.isnan(end_feature_value):
            interpolated_value = np.nan
        else:
            interpolated_vector = self.interpolate_numeric_field(start_ = 0, end_ = delta_days.days,
                                                                 start_value = start_feature_value,
                                                                 end_value = end_feature_value)
            interpolated_value = interpolated_vector[days[date_idx_before[0]]]
        return interpolated_value, delta_days.days, days_before, days_after

    def to_interpolate(self, time_deltas, table):
        """
        Return true if current time stamp has a study date before and after structure time point.
        Parameters
        ----------
        time_deltas : array of time deltas

        Returns
        -------

        """
        signs = np.sign(time_deltas.values.astype(int))

        days = time_deltas.astype('timedelta64[D]').values.astype(int)
        signs = np.sign(days)

        # if no date after exists, then return false to not interpolate
        if -1 not in signs:
            return False

        date_idx_before = max(np.argwhere(signs == 1))
        date_idx_after = min(np.argwhere(signs == -1))

        days_before = abs(days[date_idx_before[0]])
        days_after = abs(days[date_idx_after[0]])

        # if interpolation will not give the right time delta, then return false to consider a carry over
        if (days_after + days_before) > 60 and ((days_after < 60) or (days_before < 60)):
            return False

        if (1 in signs) and (-1 in signs):
            return True
        else:
            return False

    def to_carry_over(self, time_deltas):
        """
        Return true if current time stamp has a study date before and after structure time point.
        Parameters
        ----------
        time_deltas : array of time deltas

        Returns
        -------

        """
        days = time_deltas.astype('timedelta64[D]').values.astype(int)
        signs = np.sign(days)

        if not (-1 in signs):
            return True

        elif -1 in signs:
            date_idx_before = max(np.argwhere(signs == 1))
            date_idx_after = min(np.argwhere(signs == -1))

            days_before = abs(days[date_idx_before[0]])
            days_after = abs(days[date_idx_after[0]])

            # if a future date exist but it is to far away for interpolation, then carry over the closest date if closer
            # then 60 days in time
            if (days_before <= 60) or (days_after <= 60):
                return True
            else:
                return False
        else:
            return False

    def set_exact_match(self, treatment_table, stamp, candidate_idx, time_line, item, k):
        """

        Parameters
        ----------
        time_line : dict; to assign values to
        candidate_idx : int; index of which study matched with structured time point
        treatment_table : DataFrame; table with meta & feature data from first injection
        stamp : TimeStamp; structured time point for which to intepolate
        item : str; name of numerical value to interpolate
        k : int; counter for structured time point

        Returns
        -------

        """

        time_line[self.time_steps[k + 1]]["study_date"] = stamp
        time_line[self.time_steps[k + 1]][item] = treatment_table.loc[candidate_idx][item]
        time_line[self.time_steps[k + 1]]["time_range"] = None
        time_line[self.time_steps[k + 1]]["time_range_before"] = None
        time_line[self.time_steps[k + 1]]["time_range_after"] = None
        time_line[self.time_steps[k + 1]]["injections"] = treatment_table.loc[candidate_idx].cumsum_injections

        if item == "cur_va_rounded":
            time_line[self.time_steps[k + 1]]["insertion_type"] = "match"
        return time_line

    def carry_over_last(self, treatment_table, stamp, deltas, time_line, item, k):
        """
        Parameters
        ----------
        time_line : dict; to assign values to
        deltas : Series; time delta values for structured time point
        treatment_table : DataFrame; table with meta & feature data from first injection
        stamp : TimeStamp; structured time point for which to intepolate
        item : str; name of numerical value to interpolate
        k : int; counter for structured time point

        Returns
        -------

        """
        last_study = treatment_table.iloc[-1]

        time_line[self.time_steps[k + 1]]["study_date"] = stamp
        time_line[self.time_steps[k + 1]]["time_range"] = deltas.iloc[-1].days
        time_line[self.time_steps[k + 1]]["time_range_before"] = deltas.iloc[-1].days
        time_line[self.time_steps[k + 1]]["time_range_after"] = None
        time_line[self.time_steps[k + 1]]["injections"] = last_study.cumsum_injections
        time_line[self.time_steps[k + 1]][item] = last_study[item]
        time_line[self.time_steps[k + 1]]["insertion_type"] = "carry_over"
        return time_line

    def carry_over_closest(self, treatment_table, stamp, deltas, time_line, item, k):
        """
        Parameters
        ----------
        time_line : dict; to assign values to
        deltas : Series; time delta values for structured time point
        treatment_table : DataFrame; table with meta & feature data from first injection
        stamp : TimeStamp; structured time point for which to intepolate
        item : str; name of numerical value to interpolate
        k : int; counter for structured time point

        Returns
        -------

        """

        days = deltas.astype('timedelta64[D]').values.astype(int)
        signs = np.sign(days)

        if -1 in signs:
            date_idx_before = max(np.argwhere(signs == 1))
            date_idx_after = min(np.argwhere(signs == -1))

            days_before = abs(days[date_idx_before[0]])
            days_after = abs(days[date_idx_after[0]])

            start_feature_value = treatment_table[item].iloc[date_idx_before].values[0]
            end_feature_value = treatment_table[item].iloc[date_idx_after].values[0]

            if (days_before <= 60) and (days_after <= 60):
                # get the closest date to the stamp
                start_or_end = np.argmin([days_before, days_after])

                # select the closest dates feature and date values
                feature_value = [start_feature_value, end_feature_value][start_or_end]
                date_idx_select = [date_idx_before, date_idx_after][start_or_end]

                value_to_carry_over = feature_value
                last_study = treatment_table.iloc[date_idx_select]

                time_line[self.time_steps[k + 1]]["study_date"] = stamp

                if start_or_end == 1:
                    time_line[self.time_steps[k + 1]]["time_range"] = days_after
                    time_line[self.time_steps[k + 1]]["time_range_after"] = days_after
                    time_line[self.time_steps[k + 1]]["time_range_before"] = None
                elif start_or_end == 0:
                    time_line[self.time_steps[k + 1]]["time_range"] = days_before
                    time_line[self.time_steps[k + 1]]["time_range_before"] = days_before
                    time_line[self.time_steps[k + 1]]["time_range_after"] = None

                time_line[self.time_steps[k + 1]]["injections"] = last_study.cumsum_injections.values[0]
                time_line[self.time_steps[k + 1]][item] = value_to_carry_over
                time_line[self.time_steps[k + 1]]["insertion_type"] = "carry_over"

            # carry over closest value
            elif days_after < 60:
                value_to_carry_over = end_feature_value
                last_study = treatment_table.iloc[date_idx_after]

                time_line[self.time_steps[k + 1]]["study_date"] = stamp
                time_line[self.time_steps[k + 1]]["time_range"] = days_after
                time_line[self.time_steps[k + 1]]["time_range_after"] = days_after
                time_line[self.time_steps[k + 1]]["time_range_before"] = None
                time_line[self.time_steps[k + 1]]["injections"] = last_study.cumsum_injections.values[0]
                time_line[self.time_steps[k + 1]][item] = value_to_carry_over
                time_line[self.time_steps[k + 1]]["insertion_type"] = "carry_over"
                return time_line

            elif days_before <= 60:
                value_to_carry_over = start_feature_value
                last_study = treatment_table.iloc[date_idx_before]

                time_line[self.time_steps[k + 1]]["study_date"] = stamp
                time_line[self.time_steps[k + 1]]["time_range"] = days_before
                time_line[self.time_steps[k + 1]]["time_range_after"] = None
                time_line[self.time_steps[k + 1]]["time_range_before"] = days_before
                time_line[self.time_steps[k + 1]]["injections"] = last_study.cumsum_injections.values[0]
                time_line[self.time_steps[k + 1]][item] = value_to_carry_over
                time_line[self.time_steps[k + 1]]["insertion_type"] = "carry_over"
                return time_line
        else:
            last_study = treatment_table.iloc[[-1]]

            time_line[self.time_steps[k + 1]]["study_date"] = stamp
            time_line[self.time_steps[k + 1]]["time_range"] = deltas.iloc[-1].days
            time_line[self.time_steps[k + 1]]["time_range_before"] = deltas.iloc[-1].days
            time_line[self.time_steps[k + 1]]["time_range_after"] = None
            time_line[self.time_steps[k + 1]]["injections"] = last_study.cumsum_injections.values[0]
            time_line[self.time_steps[k + 1]][item] = last_study[item].iloc[0]
            time_line[self.time_steps[k + 1]]["insertion_type"] = "carry_over"
            return time_line
        return time_line

    def impute_interpolation(self, time_line, deltas, treatment_table, stamp, item, k):
        """
        Parameters
        ----------
        time_line : dict; to assign values to
        deltas : Series; time delta values for structured time point
        treatment_table : DataFrame; table with meta & feature data from first injection
        stamp : TimeStamp; structured time point for which to intepolate
        item : str; name of numerical value to interpolate
        k : int; counter for structured time point

        Returns
        -------
        time_line with assigned values
        """
        time_point = self.time_steps[k + 1]
        interp_value, interp_time, days_before, days_after = self.interpolate(deltas, treatment_table, item = item)
        time_line[time_point]["study_date"] = stamp
        time_line[time_point]["time_range"] = interp_time
        time_line[time_point]["time_range_before"] = days_before
        time_line[time_point]["time_range_after"] = days_after

        # if surgery, just insert the true false value
        if "surgery" in item:
            time_line[time_point][item] = np.ceil(interp_value)
        else:
            time_line[time_point][item] = interp_value

        # assign num injection from time point before
        time_line[time_point]["injections"] = treatment_table.iloc[np.argmin(abs(deltas)) - 1]["cumsum_injections"]

        time_line[time_point]["insertion_type"] = "interpolation"
        return time_line

    def get_structured_timestamps(self, first_time_stamp):
        """
        Set set 3, 5, 12, 24 time stamp w.r.t. the first injection date.
        Parameters
        ----------
        first_time_stamp :

        Returns
        -------

        """
        treatment_stamps = []
        for mark in self.time_steps[1:]:
            time_mark = first_time_stamp
            time_mark += datetime.timedelta(days = int(TimeUtils.DAYS * mark))
            treatment_stamps.append(time_mark)
        return treatment_stamps

    def treatment_table(self):
        """
        Gets all study dates with injections and filters the first.
        Then filters table from first date.
        resets index and return the table.
        @return: DataFrame table filtered from first treatment date
        @rtype: DataFrame
        """
        first_injection_idx = min(np.argwhere(np.array(self.table["injections"] > 0)))[0]
        treatment_table = self.table.iloc[first_injection_idx:]

        # reset index to start from 1
        treatment_table.index = treatment_table.index - treatment_table.index[0] + 1

        return treatment_table

    def assign_to_timeline_str(self, time_line, item, total_number_injections):
        """

        Parameters
        ----------
        time_line :
        item :
        total_number_injections :

        Returns
        -------

        """

        if total_number_injections > 0:
            treatment_table = self.treatment_table()
        else:
            treatment_table = deepcopy(self.table)

            # reset index to start from 1
            treatment_table.index = treatment_table.index - treatment_table.index[0] + 1

        # if va is zero at time zero, then hand over next value if time delta less than 60
        if item == 'cur_va_rounded':
            if treatment_table.iloc[0][[item]].isna().values[0]:
                for i in range(2, treatment_table.shape[0] - 1):
                    if (treatment_table.delta_t[i-1] < 60) and not treatment_table.iloc[i - 1][[item]].isna().values[0]:
                        treatment_table.loc[i - 1, item] = treatment_table.iloc[i - 1][[item]].values[0]
                        break

        # set up time stamp for structured time series
        first_treatment_date = pd.to_datetime(treatment_table.iloc[0].study_date)
        self.time_line[1]["study_date"] = first_treatment_date
        self.time_line[1][item] = treatment_table.iloc[0][item]
        self.time_line[1]["time_range"] = None
        self.time_line[1]["time_range_before"] = None
        self.time_line[1]["time_range_after"] = None
        self.time_line[1]["injections"] = treatment_table.iloc[0].cumsum_injections
        self.time_line[1]["insertion_type"] = None

        # get 3, 6, 12, 24 time stamps for structured time series
        treatment_stamps = self.get_structured_timestamps(first_treatment_date)

        for k, stamp in enumerate(treatment_stamps, 0):

            # is there an exact match, then match
            deltas = stamp - pd.to_datetime(treatment_table.study_date)
            candidate_idx = np.argmin(np.array(np.abs(deltas))) + 1

            # check for -15 / 15 day match
            if (abs(deltas[candidate_idx].days) <= TimeUtils.DAYS / 2):# & ~treatment_table[[item]].loc[candidate_idx].isna().values[0]:
                time_line = self.set_exact_match(treatment_table, stamp, candidate_idx, time_line, item, k)

            # no match but date before and after, interpolate
            elif self.to_interpolate(deltas, treatment_table):
                time_line = self.impute_interpolation(time_line, deltas, treatment_table, stamp, item, k)

            # no match but date before carry last observation forward
            elif self.to_carry_over(deltas):
                time_line = self.carry_over_closest(treatment_table, stamp, deltas, time_line, item, k)
        return time_line
