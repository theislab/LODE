from copy import deepcopy
import numpy as np
import os
import pandas as pd


class SeqUtils():
    def __init__(self, time_line):
        self.time_line = time_line

    def get_number_of_visits(self):
        return len(list(filter(lambda x: x != "nan", self.study_dates)))

    @property
    def study_dates(self):
        study_dates = []
        # iterate through months and retrieve measurement vector
        months = list(self.time_line.keys())
        for month in months:
            study_dates.append(self.time_line[month]["study_date"])
        return np.array(study_dates)

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
        """
        @param time_line: measurement monthly time line dictionary
        @type time_line: dict
        @return: number of injection during patient life cycle
        @rtype: int
        """
        return int(np.nansum([time_line[time_point]["injections"] for time_point in time_line]))

    @classmethod
    def is_naive(cls, table):
        """
        @param table: The record table DataFrame
        @type table: DataFrame
        @return: if record is naive or not
        @rtype: boolean
        """
        first_record = table.iloc[[0]]
        return first_record.lens_surgery[0]

    @classmethod
    def get_recorded_months(cls, time_line):
        return [key for key in time_line.keys() if time_line[key]['study_date'] is not np.nan]

    @classmethod
    def fluid_coefficient(cls, time_line, from_):
        # set copy for independent editing
        time_line_temp = deepcopy(time_line)
        recorded_months = cls.get_recorded_months(time_line)
        first_injection = cls.first_event_time(recorded_months, time_line_temp, "injections")

        if from_ == "first_injection":
            if not first_injection:
                return "no injection"
            else:
                # get first or month before first injection
                injection_month = max(1, recorded_months.index(first_injection))

                final_visit = recorded_months[-1]
                fluid_before_treatment = time_line[injection_month]["total_fluid"]
                fluid_at_final = time_line[final_visit]["total_fluid"]

                x = [injection_month, final_visit]
                y = [fluid_before_treatment, fluid_at_final]
                return np.polyfit(x, y, 1)[0]

        elif from_ == "first_fluid":
            first_fluid = cls.first_event_time(recorded_months, time_line_temp, "total_fluid")
            if not first_fluid:
                return "no fluid"
            else:
                # get first or month before first injection
                fluid_month = max(1, recorded_months.index(first_fluid))

                final_visit = recorded_months[-1]
                fluid_before_treatment = time_line[fluid_month]["total_fluid"]
                fluid_at_final = time_line[final_visit]["total_fluid"]

                x = [fluid_month, final_visit]
                y = [fluid_before_treatment, fluid_at_final]
                return np.polyfit(x, y, 1)[0]

    @classmethod
    def get_time_until_dry(cls, time_line, from_):
        """
        @param time_line: see above
        @type time_line: dict
        @param from_: either 'first_injection' or 'first_fluid', indicating from when to start measuring
        @type str
        @return: first month where no fluid is observed given a prev record with fluid
        @rtype: int
        """
        # set copy for independent editing
        time_line_temp = deepcopy(time_line)
        months = list(time_line_temp.keys())
        recorded_months = cls.get_recorded_months(time_line)

        first_injection = cls.first_event_time(recorded_months, time_line_temp, "injections")
        if from_ == "first_injection":
            if not first_injection:
                return "no injection"
            else:
                # get first or month before first injection
                injection_month = max(1, recorded_months.index(first_injection) - 1)
                for month in months[injection_month:]:
                    if time_line_temp[month]["total_fluid"] == 0:
                        return month
                    else:
                        continue

        elif from_ == "first_fluid":
            first_fluid = cls.first_event_time(recorded_months, time_line, "total_fluid")
            if not first_fluid:
                return "no fluid"
            else:
                for month in months[first_fluid:]:
                    if time_line_temp[month]["total_fluid"] == 0:
                        return month
                    else:
                        continue
        return -1

    @classmethod
    def set_treatment_effect(cls, time_line, time_dist, item):
        time_dict = {"three": 2, "six": 5}
        for time_point in time_line.keys():
            injection_point = time_line[time_point]["injections"]
            injection_bool = (injection_point > 0) and (injection_point is not np.nan)
            time_bool = (time_point + time_dict[time_dist] <= len(time_line))
            if injection_bool and time_bool:
                current_value = time_line[time_point][item]
                effect = time_line[time_point + time_dict[time_dist]][item] / current_value
                time_line[time_point][f"{time_dist}_month_effect"] = effect
            else:
                time_line[time_point][f"{time_dist}_month_effect"] = np.nan
        return time_line

    @classmethod
    def first_event_time(cls, months, time_line, event):
        """
        @param time_line: see above
        @type time_line: dict
        @param event: type of event, fluid, injections etc.
        @type str
        @return: month of event
        @rtype: int
        """
        for month in months:
            if time_line[month][event] > 0:
                return month
        return None


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def get_total_number_of_injections(table):
    """
    @param table: record table with clinical information for sequence
    @type table: DataFrame
    @return: total injections, i.e. number of injections of all types at visit, added as column
    @rtype: int
    """
    return sum(table.injections)


def get_first_month_injection(time_line):
    """
    @param time_line: LODE time line dict with all measurements over time line
    @type table: DataFrame
    @return: total injections, i.e. number of injections of all types at visit, added as column
    @rtype: int
    """
    month_with_injections = [month for month in time_line.keys() if time_line[month]["injections"] > 0]
    if month_with_injections:
        month_ = min(month_with_injections)
        return time_line[month_]["study_date"], month_
    else:
        return None, None


def get_treatment_time_line(time_line, first_month):
    """
    - iterate through time line

    extract fluid, date from month 1, 4, 7, 13

    Warning: function assumes injection is assigned to previous time point.

    for month 1:
    set injection_since_last = 0
    total_injections = 0

    for month 4:
    set injection_since_last = sum(injection) | month = [1, 2, 3]
    total injections = total_injections[month = 1] + injections_since_last

    for month 7:
    set injection_since_last = sum(injection) | month = [4, 5, 6]
    total injections = total_injections[month = 1] + total_injections[month = 4] + injections_since_last

    for month 13:
    set injection_since_last = sum(injection) | month = [7, 8, 8, 10, 12]
    total injections = total_injections[month = 1] + total_injections[month = 4] + total_injections[month = 7] + injections_since_last
    """
    fluid_time_time = {}

    time_points = [first_month, first_month + 3,
                   first_month + 6, first_month + 12]

    deltas = [1, 3, 6, 12]
    # fill fluid time line
    for k, time_point in enumerate(time_points, 1):
        if time_point in time_line.keys():

            # if first time point
            if k == 1:
                num_injection_since_last = 0
                total_injections = 0
            else:
                injections_since_last = [time_line[i]["injections"] for i in range(prev_time_point, time_point)]

                num_injection_since_last = np.nansum(injections_since_last)
                total_injections = fluid_time_time[str(deltas[k - 2])]["total_injections"] + num_injection_since_last

            fluid_time_time[str(deltas[k - 1])] = {"month": time_point,
                                                   "study_date": time_line[time_point]["study_date"],
                                                   "injection_since_last": int(num_injection_since_last),
                                                   "total_injections": int(total_injections),
                                                   "total_fluid": time_line[time_point]["total_fluid"]}

        else:
            fluid_time_time[str(deltas[k - 1])] = {"month": None,
                                                   "study_date": None,
                                                   "injection_since_last": None,
                                                   "total_injections": None,
                                                   "total_fluid": None}
        prev_time_point = time_point
    return fluid_time_time


def get_delta_logs(fluid_time_line, deltas):
    """
    function calculates absolute and relative fluid delta in fluid_time_line dict
    @param fluid_time_line:
    @type fluid_time_line:
    @return: two dicts, with relative and absolute differences
    @rtype: dict
    """
    if fluid_time_line:
        abs_dict = dict([(delta + "abs", None) for delta in deltas])
        injection_dict = dict([(delta + "inj", None) for delta in deltas])

        for abs_delta in abs_dict.keys():
            months = abs_delta.replace("abs", "")

            current_month = months.split("-")[0]
            next_month = months.split("-")[1]

            current_fluid = fluid_time_line[current_month]["total_fluid"]
            next_fluid = fluid_time_line[next_month]["total_fluid"]

            # if time series ended early
            if next_fluid is not None:
                fluid_delta = next_fluid - current_fluid
                next_total_injections = fluid_time_line[next_month]["total_injections"]
            else:
                fluid_delta = None
                next_total_injections = None

            abs_dict[abs_delta] = fluid_delta
            injection_dict[abs_delta.replace("abs", "inj")] = next_total_injections

        return abs_dict, injection_dict
    else:
        return None, None, None


def check_features(workspace_dir, longitudinal_data):
    """
    workspace_dir: str
    longitudinal_data: DataFrame with long. data
    """
    segmentation_feature_path = os.path.join(workspace_dir, "joint_export/segmentation/segmentation_statistics.csv")

    assert os.path.exists(segmentation_feature_path), "Features not available in work space"

    # if feature stat table exists load here
    segmented_data = pd.read_csv(segmentation_feature_path, index_col = 0)

    # get feature names
    feature_names = segmented_data.columns[1:]

    # join with longitudinal data table
    keys = ["patient_id", "laterality", "study_date"]
    segmented_data[keys] = segmented_data.record.str.split("_", expand = True)[[0, 1, 2]]

    # convert keys to same format is in longitudinal_data
    segmented_data["study_date"] = pd.to_datetime(segmented_data["study_date"]).astype(str)

    # convert patient id to int
    segmented_data["patient_id"] = segmented_data["patient_id"].astype(np.int64)

    longitudinal_data = pd.merge(longitudinal_data, segmented_data, left_on = keys, right_on = keys, how = "inner")
    return longitudinal_data, feature_names.tolist()
