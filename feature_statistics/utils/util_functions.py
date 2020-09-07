from copy import deepcopy

import numpy as np


class SeqUtils():
    def __init__(self, time_line):
        self.time_line = time_line
        pass

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
        return int(np.nansum([time_line[time_point]["total_injections"] for time_point in time_line]))

    @classmethod
    def is_naive(cls, table):
        """
        @param table: The record table DataFrame
        @type table: DataFrame
        @return: if record is naive or not
        @rtype: boolean
        """
        first_record = table.iloc[[0]]
        injections = list(map(lambda x: int(x), first_record.injections[0].split(", ")))
        return (not first_record.lens_surgery[0]) & (np.sum(injections) == 0)

    @classmethod
    def get_recorded_months(cls, time_line):
        return [key for key in time_line.keys() if time_line[key]['study_date'] is not np.nan]

    @classmethod
    def fluid_coefficient(cls, time_line, from_):
        # set copy for independent editing
        time_line_temp = deepcopy(time_line)
        recorded_months = cls.get_recorded_months(time_line)
        first_injection = cls.first_event_time(recorded_months, time_line_temp, "total_injections")

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

        first_injection = cls.first_event_time(recorded_months, time_line_temp, "total_injections")
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
            injection_point = time_line[time_point]["total_injections"]
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