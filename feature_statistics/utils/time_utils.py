import math

import numpy as np
import datetime
import pandas as pd


class TimeUtils:
    """

    """
    def __init__(self, record_table):
        self.table = record_table
        self.NUMBER_OF_MONTHS = self.set_number_of_months
        self.time_line = dict((k, {}) for k in range(1, self.NUMBER_OF_MONTHS + 1))
        self.DAYS = 30

    @property
    def set_number_of_months(self):
        months = pd.to_datetime(self.table.study_date)
        time_span = months.iloc[-1] - months.iloc[0]
        return math.ceil(time_span.days / 30) + 1

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

        __start__ = datetime.datetime.strptime(self.table["study_date"].iloc[0], '%Y-%m-%d')
        __end__ = datetime.datetime.strptime(self.table["study_date"].iloc[-1], '%Y-%m-%d')

        one_month_timestamps = []
        date_x = __start__
        while date_x < __end__:
            date_x += datetime.timedelta(days = self.DAYS)
            # if not (__end__ - date_x) < timedelta(days = MeasureSeqTimeUntilDry.DAYS):
            one_month_timestamps.append(date_x)

        anchor_dates = np.array(one_month_timestamps)
        candidate_dates = pd.to_datetime(self.table["study_date"])
        for k, date in enumerate(anchor_dates, 2):
            if k <= self.NUMBER_OF_MONTHS:
                add_to_next_injections = 0
                add_to_prev_injections = 0

                time_delta = candidate_dates - date
                candidates = [td for td in time_delta if (td.days < 15) & (td.days > -15)]
                if len(candidates) > 1:
                    min_diff = np.min(candidates)
                    idx_select = np.where(time_delta == min_diff)[0][0]
                    time_line[k].update({item: self.table[item].iloc[idx_select]})

                    # if adding injections, assign close by removed studies to neighbouring visits
                    if item == "total_injections":
                        # remove selected candidate
                        candidates.remove(min_diff)

                        # if candidates are discarded, assign injection data to time line
                        for candidate in candidates:
                            idx_not_select = np.where(time_delta == candidate)[0][0]
                            if candidate.days > 0:
                                add_to_next_injections = self.table["total_injections"].iloc[idx_not_select]
                            else:
                                add_to_prev_injections = self.table["total_injections"].iloc[idx_not_select]

                            # update neighbouring time lines
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

                elif not candidates:
                    time_line[k].update({item: np.nan})
                    continue
                else:
                    idx_select = np.where(time_delta == candidates[0])[0][0]
                    time_line[k].update({item: self.table[item].iloc[idx_select]})
        return time_line