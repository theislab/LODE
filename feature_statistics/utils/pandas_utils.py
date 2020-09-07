import pandas as pd
import numpy as np
import itertools

from utils.util_functions import nan_helper


def sum_etdrs_columns(record_table, rings, regions, features, foveal_region, new_column_name):
    fluid = pd.Series(np.zeros(record_table.shape[0]))
    for etdrs_tuple in list(itertools.product(regions, rings, features)):
        region_fluid = f"{etdrs_tuple[0]}{etdrs_tuple[1]}_{etdrs_tuple[2]}"
        fluid += record_table[region_fluid]

    # add foval fluid
    for foveal_tuple in list(itertools.product(foveal_region, features)):
        foveal_fluid = f"{foveal_tuple[0]}_{foveal_tuple[1]}"
        fluid += record_table[foveal_fluid]
    record_table.insert(10, new_column_name, fluid.values.tolist(), True)
    return record_table


def avg_etdrs_columns(record_table, rings, regions, features, foveal_region, new_column_name):
    n_counts = 0
    value_ = pd.Series(np.zeros(record_table.shape[0]))
    for etdrs_tuple in list(itertools.product(regions, rings, features)):
        region_value = f"{etdrs_tuple[0]}{etdrs_tuple[1]}_{etdrs_tuple[2]}"
        value_ += record_table[region_value]
        n_counts += 1

    # add foval fluid
    for foveal_tuple in list(itertools.product(foveal_region, features)):
        foveal_value = f"{foveal_tuple[0]}_{foveal_tuple[1]}"
        value_ += record_table[foveal_value]
        n_counts += 1

    # average the summations
    avg_values = value_ / n_counts
    record_table.insert(10, new_column_name, avg_values.values.tolist(), True)
    return record_table


def get_total_number_of_injections(table):
    """
    @param table: record table with clinical information for sequence
    @type table: DataFrame
    @return: DataFrame with total injections, i.e. number of injections of all types at visit, added as column
    @rtype: DataFrame
    """
    total_injections = [np.sum(list(map(lambda x: int(x), row))) for row in table.injections.str.split(", ")]
    table.insert(loc = 10, column = "total_injections", value = pd.Series(total_injections),
                 allow_duplicates = True)
    return table


def interpolate_numeric_field(time_line, item):
    """
    @param time_line: see prev function
    @type time_line: dict
    @param item: item to be interpolated
    @type item: str
    @return: time_line with interpolated numerical values
    @rtype: dict
    """
    interp_vector = []

    # iterate through months and retrieve measurement vector
    months = list(time_line.keys())
    for month in months:
        interp_vector.append(time_line[month][item])

    # interpolate missing values
    interp_array = np.array(interp_vector)
    nans, x = nan_helper(interp_array)
    interp_array[nans] = np.interp(x(nans), x(~nans), interp_array[~nans])

    # assign interp values
    for i, month in enumerate(months):
        time_line[month][item] = np.round(interp_array[i], 2)
    return time_line
