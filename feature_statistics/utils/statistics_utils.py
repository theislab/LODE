from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
import statsmodels.api as sm

PIXEL_TO_VOLUME = 0.010661 * 0.003872 * 0.112878
THICKNESS_TO_MM = 0.003872


def sum_etdrs(table, time):
    label_mapping = {f"_1_{time}": f"epm_{time}", f"_3_{time}": f"irf_{time}", f"_4_{time}": f"srf_{time}",
                     f"_5_{time}": f"srhm_{time}", f"_6_{time}": f"rpe_{time}", f"_7_{time}": f"fvpde_{time}",
                     f"_8_{time}": f"drusen_{time}", f"_9_{time}": f"phm_{time}", f"_10_{time}": f"choroid_{time}",
                     f"_13_{time}": f"fibrosis_{time}", f"_atropypercentage_{time}": f"atropypercentage_{time}",
                     f"_thicknessmean_{time}": f"thicknessmean_{time}"}

    feature_names = [f"_1_{time}", f"_3_{time}",
                     f"_4_{time}", f"_5_{time}",
                     f"_6_{time}", f"_7_{time}",
                     f"_8_{time}", f"_9_{time}",
                     f"_10_{time}", f"_13_{time}",
                     f"_atropypercentage_{time}",
                     f"thicknessmean_{time}"]

    table_non_spatial_pd = pd.DataFrame([])

    for feature in feature_names:
        col_to_sum = table.columns.str.endswith(feature)
        if ("atropy" in feature) | ("thicknessmean" in feature):
            table_non_spatial_pd[feature] = table.iloc[:, col_to_sum].sum(1) * THICKNESS_TO_MM
        else:
            table_non_spatial_pd[feature] = table.iloc[:, col_to_sum].sum(1) * PIXEL_TO_VOLUME

    table_non_spatial_pd = table_non_spatial_pd.rename(columns = label_mapping)
    return table_non_spatial_pd


def add_etdrs(table, time):
    table_spatial_pd = pd.DataFrame([])

    label_mapping = {f"_1_{time}": f"epm_{time}", f"_3_{time}": f"irf_{time}", f"_4_{time}": f"srf_{time}",
                     f"_5_{time}": f"srhm_{time}", f"_6_{time}": f"rpe_{time}", f"_7_{time}": f"fvpde_{time}",
                     f"_8_{time}": f"drusen_{time}", f"_9_{time}": f"phm_{time}", f"_10_{time}": f"choroid_{time}",
                     f"_13_{time}": f"fibrosis_{time}", f"_atropypercentage_{time}": f"atropypercentage_{time}",
                     f"_thicknessmean_{time}": f"thicknessmean_{time}"}

    etdrs_cells = ["T1", "T2", "S1", "S2", "N1", "N2", "I1", "I2", "C0"]

    spatial_label_mapping = {}
    for ecell in etdrs_cells:
        for label_map in label_mapping.keys():
            spatial_label_mapping[f"{ecell}{label_map}"] = f"{ecell}-{label_mapping[label_map]}"

    feature_names = [f"_1_{time}", f"_3_{time}",
                     f"_4_{time}", f"_5_{time}",
                     f"_6_{time}", f"_7_{time}",
                     f"_8_{time}", f"_9_{time}",
                     f"_10_{time}", f"_13_{time}",
                     f"_atropypercentage_{time}",
                     f"_thicknessmean_{time}"]

    for ecell in etdrs_cells:
        for feature in feature_names:
            spatial_feature = f"{ecell}{feature}"

            if spatial_feature in table.columns.tolist():
                if ("atropy" in spatial_feature) | ("thicknessmean" in spatial_feature):
                    table_spatial_pd.loc[:, spatial_feature] = table.loc[:, spatial_feature] * THICKNESS_TO_MM
                else:
                    table_spatial_pd.loc[:, spatial_feature] = table.loc[:, spatial_feature] * PIXEL_TO_VOLUME
    table_spatial_pd = table_spatial_pd.rename(columns = spatial_label_mapping)
    return table_spatial_pd


def calc_delta_columns(times, feature_dict):
    delta_feature_dict = {}

    # calculate delta columns
    for k, time in enumerate(times):
        if k < len(times) - 1:
            remaining_tps = times[k + 1:]
            for next_time_point in remaining_tps:
                delta_columns = []
                for col in feature_dict[f"feature_{time}"].columns:
                    feat, t = col.split("_")
                    delta_columns.append(f"{feat}_{next_time_point}_delta_{t}")

                delta_pd = pd.DataFrame(np.array(feature_dict[f"feature_{next_time_point}"]) -
                                        np.array(feature_dict[f"feature_{time}"]),
                                        columns = delta_columns)

                delta_feature_dict[f"feature_{next_time_point}_{time}"] = delta_pd
    return delta_feature_dict


def get_feature_dicts(times, table, spatial_sum=False):
    feature_dict = {}

    for time in times:
        if spatial_sum:
            feature_dict[f"feature_{time}"] = sum_etdrs(table, time = time)
        else:
            feature_dict[f"feature_{time}"] = add_etdrs(table, time = time)

    delta_feature_dict = calc_delta_columns(times, feature_dict)
    return {**delta_feature_dict, **feature_dict}


def get_va_dict(times, table):
    column_starter = "cur_va_rounded_"
    va_dict = {}
    delta_va_dict = {}

    for time in times:
        va_dict[f"va_{time}"] = table[f"cur_va_rounded_{time}"]

    # calculate delta columns
    for k, time in enumerate(times):
        if k < len(times) - 1:
            remaining_tps = times[k + 1:]
            for next_time_point in remaining_tps:
                delta_columns = [f"va_{next_time_point}_delta_{time}"]

                delta_pd = pd.DataFrame(np.array(va_dict[f"va_{next_time_point}"]) - np.array(va_dict[f"va_{time}"]),
                                        columns = delta_columns)
                delta_va_dict[f"va_{next_time_point}_{time}"] = delta_pd

    return {**delta_va_dict, **va_dict}


def associate_time_n_factors(table=None, spatial_sum=False, time_filters=None, times=None):
    # filter time points
    for tp in times:
        table = table[time_filters[tp]]

    seq_columns = ['patient_id', 'laterality']
    table["sequence"] = table[seq_columns[0]] + "_" + table[seq_columns[1]]

    # sum etdrs features for seg features across the time points
    features_data_dict = get_feature_dicts(times, table, spatial_sum)
    va_data_dict = get_va_dict(times, table)

    data_dict = {**va_data_dict, **features_data_dict}
    data_frames = list(data_dict.keys())

    df = data_dict[data_frames[0]]
    for data_frame in data_frames[1:]:
        df = pd.concat([df.reset_index(drop = True), data_dict[data_frame].reset_index(drop = True)], axis = 1)

    for time in times[1:]:
        df[f"n_injections_{time}"] = table[f"cumsum_injections_{time}"].values.tolist()
    # filter all columns for the independent ones in list
    df.index = table["sequence"]
    return df


def assert_times(time_point_list):
    allowed_values = [1, 3, 6, 12]

    for value in time_point_list:
        if value not in allowed_values:
            return False
        else:
            continue
    return True


# features independents
def get_seg_independents_str(seg_features, seg_delta, seg_times):
    seg_dependents = []

    for seg_feature in seg_features:
        for time in seg_times:
            seg_dependents.append(f"{seg_feature}_{time}")

        if seg_delta:
            # add all delta columns
            for k, s_time in enumerate(seg_times[:-1]):

                remaining_times = deepcopy(seg_times)
                remaining_times.remove(s_time)

                for r_time in remaining_times[k:]:
                    if f"{r_time}-{s_time}" in seg_delta:
                        seg_dependents.append(f"{seg_feature}_{r_time}_delta_{s_time}")

    return seg_dependents


def get_va_dependents_str(va_delta, va_times):
    va_dependents = []

    for va_time in va_times:
        va_dependents.append(f"cur_va_rounded_{va_time}")

    if va_delta:
        # add all delta columns
        for k, s_time in enumerate(va_times[:-1]):

            remaining_times = deepcopy(va_times)
            remaining_times.remove(s_time)

            for r_time in remaining_times[k:]:
                if f"{r_time}-{s_time}" in va_delta:
                    va_dependents.append(f"va_{r_time}_delta_{s_time}")
                else:
                    continue
    return va_dependents


def filter_time_ranges(data_pd):
    """
    Function looks at 3, 6, and 12 month observation and sees what time difference to the fixed interval time interval
    mark was present. If interval fulfills criteras for inclusion in the study:
    https://docs.google.com/document/d/1bnKlKsCS5NdMMDu6XD5FqIdLow3sQ_64HwDj7UaPRns/edit,

    then the value is set to True, else False.

    Important: The function assumes that data_pd as generator by the structure_time_series.py in the
    feature statistics project: https://github.com/theislab/LODE/tree/master/feature_statistics

    :param data_pd: data frame with the longitudinal poperties table
    :type data_pd: data frame
    :return: boolean vectors denoting for each time interval, which
    :rtype: arrays
    """
    columns = ["time_range_3", "time_range_before_3", "time_range_after_3", "insertion_type_3",
               "time_range_6", "time_range_before_6", "time_range_after_6", "insertion_type_6",
               "time_range_12", "time_range_before_12", "time_range_after_12", "insertion_type_12"]

    filter_base = data_pd[columns]

    # filter for fist month where VA values are available
    filter_1 = ~data_pd.cur_va_rounded_1.isna()

    # 3 month bools
    interp_3 = filter_base.insertion_type_3 == "interpolation"
    carry_over_3 = filter_base.insertion_type_3 == "carry_over"
    match_3 = filter_base.insertion_type_3 == "match"

    # 6 month bools
    interp_6 = filter_base.insertion_type_6 == "interpolation"
    carry_over_6 = filter_base.insertion_type_6 == "carry_over"
    match_6 = filter_base.insertion_type_6 == "match"

    # 12 month bools
    interp_12 = filter_base.insertion_type_12 == "interpolation"
    carry_over_12 = filter_base.insertion_type_12 == "carry_over"
    match_12 = filter_base.insertion_type_12 == "match"

    DAY_FILTER = 60

    # interpolation time filters
    interp_time_before_3 = filter_base.time_range_before_3 < DAY_FILTER
    interp_time_after_3 = filter_base.time_range_after_3 < DAY_FILTER

    interp_time_before_6 = filter_base.time_range_before_6 < DAY_FILTER
    interp_time_after_6 = filter_base.time_range_after_6 < DAY_FILTER

    interp_time_before_12 = filter_base.time_range_before_12 < DAY_FILTER
    interp_time_after_12 = filter_base.time_range_after_12 < DAY_FILTER

    # carry over time filters
    carry_over_time_after_3 = filter_base.time_range_3 < DAY_FILTER
    carry_over_time_after_6 = filter_base.time_range_6 < DAY_FILTER
    carry_over_time_after_12 = filter_base.time_range_12 < DAY_FILTER

    # interpolation 3 months
    interp_bef = np.logical_and(interp_3.values, interp_time_before_3)
    interp_aft = np.logical_and(interp_3.values, interp_time_after_3)

    interp_3 = np.logical_and(interp_bef, interp_aft)

    # carry over 6 months
    carry_over_3 = np.logical_and(carry_over_3.values, carry_over_time_after_3)

    insertion_3_ = np.logical_or(carry_over_3, interp_3)
    filter_3 = np.logical_or(insertion_3_, match_3)

    print("Number of filtered sequences for 3 months are:", sum(filter_3))

    # interpolation 6 months
    interp_bef = np.logical_and(interp_6.values, interp_time_before_6)
    interp_aft = np.logical_and(interp_6.values, interp_time_after_6)

    interp_6 = np.logical_and(interp_bef, interp_aft)

    # carry over 6 months
    carry_over_6 = np.logical_and(carry_over_6.values, carry_over_time_after_6)

    insertion_6_ = np.logical_or(carry_over_6, interp_6)
    filter_6 = np.logical_or(insertion_6_, match_6)

    print("Number of filtered sequences for 6 months are:", sum(filter_6))

    # interpolation 12 months
    interp_bef = np.logical_and(interp_12.values, interp_time_before_12)
    interp_aft = np.logical_and(interp_12.values, interp_time_after_12)

    interp_12 = np.logical_and(interp_bef, interp_aft)

    # carry over 6 months
    carry_over_12 = np.logical_and(carry_over_12.values, carry_over_time_after_12)

    insertion_12_ = np.logical_or(carry_over_12, interp_12)
    filter_12 = np.logical_or(insertion_12_, match_12)

    print("Number of filtered sequences for 12 months are:", sum(filter_12))

    return filter_1, filter_3, filter_6, filter_12


def preprocess_dataframe(data_pd, oct_meta_pd):
    """
    Function renames and merges informtaion used for result analysis.

    Parameters
    ----------
    data_pd : Data frame, table from structure_time_series.py
    oct_meta_pd : meta file to link seg and oct paths on ICB compute server

    Returns
    -------
    dataframe data pd
    """
    #### add patient id and lateraliy
    data_pd["patient_id"] = data_pd.sequence.str.split("_", expand = True)[0]
    data_pd["laterality"] = data_pd.sequence.str.split("_", expand = True)[1]

    # add seg numpy a
    data_pd["seg_record_1"] = data_pd.sequence + "_" + data_pd.study_date_1.str.replace("-", "") + ".npy"
    data_pd["seg_record_3"] = data_pd.sequence + "_" + data_pd.study_date_3.str.replace("-", "") + ".npy"
    data_pd["seg_record_12"] = data_pd.sequence + "_" + data_pd.study_date_12.str.replace("-", "") + ".npy"

    # rename atrophy and thickness columns
    for column in data_pd.columns:
        if "atropy_percentage" in column:
            data_pd.rename(columns = {column: column.replace("atropy_percentage", "atropypercentage")}, inplace = True)

        if "thickness_mean" in column:
            data_pd.rename(columns = {column: column.replace("thickness_mean", "thicknessmean")}, inplace = True)

    study_date_1_dc = data_pd.sequence.str.split("_", expand = True)
    study_date_1_dc = study_date_1_dc.rename(columns = {0: "PATNR", 1: "laterality"})
    study_date_1_dc["study_date"] = data_pd["study_date_1"].str.replace("-", "")

    study_date_1_dc.PATNR = study_date_1_dc.PATNR.astype(np.int64)
    study_date_1_dc.study_date = study_date_1_dc.study_date.astype(np.int64)

    keys = ["PATNR", "laterality", "study_date"]
    result1 = pd.merge(study_date_1_dc, oct_meta_pd[keys + ["oct_path"]], left_on = keys, right_on = keys, how = "left")
    result1 = result1.drop_duplicates(subset = keys)
    result1 = result1.rename(columns = {"oct_path": "study_date_1_dicom_path"})

    result1["sequence"] = result1.PATNR.astype(str) + "_" + result1.laterality
    ##############

    study_date_3_dc = data_pd.sequence.str.split("_", expand = True)
    study_date_3_dc = study_date_3_dc.rename(columns = {0: "PATNR", 1: "laterality"})
    study_date_3_dc["study_date"] = data_pd["study_date_3"].str.replace("-", "")

    study_date_3_dc.PATNR = study_date_3_dc.PATNR.astype(np.int64)
    study_date_3_dc.study_date = study_date_3_dc.study_date.astype(np.int64)

    keys = ["PATNR", "laterality", "study_date"]
    result3 = pd.merge(study_date_3_dc, oct_meta_pd[keys + ["oct_path"]], left_on = keys, right_on = keys, how = "left")
    result3 = result3.drop_duplicates(subset = keys)
    result3 = result3.rename(columns = {"oct_path": "study_date_3_dicom_path"})

    result3["sequence"] = result3.PATNR.astype(str) + "_" + result3.laterality
    #############

    study_date_12_dc = data_pd.sequence.str.split("_", expand = True)
    study_date_12_dc = study_date_12_dc.rename(columns = {0: "PATNR", 1: "laterality"})
    study_date_12_dc["study_date"] = data_pd["study_date_12"].str.replace("-", "")

    study_date_12_dc.PATNR = study_date_12_dc.PATNR.astype(np.int64)
    study_date_12_dc.study_date = study_date_12_dc.study_date.astype(np.int64)

    keys = ["PATNR", "laterality", "study_date"]
    result12 = pd.merge(study_date_12_dc, oct_meta_pd[keys + ["oct_path"]], left_on = keys, right_on = keys,
                        how = "left")
    result12 = result12.drop_duplicates(subset = keys)
    result12 = result12.rename(columns = {"oct_path": "study_date_12_dicom_path"})

    result12["sequence"] = result12.PATNR.astype(str) + "_" + result12.laterality

    data_pd = pd.merge(data_pd, result1[["study_date_1_dicom_path", "sequence"]], left_on = "sequence",
                       right_on = "sequence",
                       how = "left")

    data_pd = pd.merge(data_pd, result3[["study_date_3_dicom_path", "sequence"]], left_on = "sequence",
                       right_on = "sequence",
                       how = "left")

    data_pd = pd.merge(data_pd, result12[["study_date_12_dicom_path", "sequence"]], left_on = "sequence",
                       right_on = "sequence",
                       how = "left")

    return data_pd

if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pydicom import read_file
    import copy
    import sys
    import glob
    from copy import deepcopy
    from scipy import stats
    from matplotlib.gridspec import GridSpec
    import tqdm
    import seaborn as sns

    PROJ_DIR = "/home/olle/PycharmProjects/LODE"

    sys.path.insert(0, os.path.join(PROJ_DIR, 'feature_statistics/utils'))

    import statistics_utils as su

    plt.style.use('seaborn')

    WORK_SPACE = "/home/olle/PycharmProjects/LODE/workspace"

    oct_meta_pd = pd.read_csv(os.path.join(WORK_SPACE, "joint_export/export_tables/oct_meta_information.csv"))

    oct_meta_pd.loc[:, "sequence"] = oct_meta_pd.PATNR.astype(str) + "_" + oct_meta_pd.laterality

    data_pd = pd.read_csv(os.path.join(WORK_SPACE,
                                       "joint_export/longitudinal_properties_naive.csv"))

    filter_1, filter_3, filter_6, filter_12 = su.filter_time_ranges(data_pd)

    data_pd = su.preprocess_dataframe(data_pd, oct_meta_pd)

    seg_features = ["epm", "irf", "srf", "srhm", "rpe", "fvpde", "drusen", "phm", "choroid", "fibrosis",
                    "atropypercentage", "thicknessmean"]

    seg_delta = []
    seg_times = [1, 3, 12]

    assert su.assert_times(seg_times), "Selected time points contains not allowed values"

    seg_independents = su.get_seg_independents_str(seg_features, seg_delta, seg_times)

    va_delta = []
    va_times = [1, 3, 12]

    assert su.assert_times(va_times), "Selected time points contains not allowed values"

    va_independents = su.get_va_dependents_str(va_delta, va_times)

    injection_times = [3]

    assert su.assert_times(injection_times), "Selected time points contains not allowed values"

    injection_independents = []
    for it in injection_times:
        injection_independents.append(f"n_injections_{it}")

    time_filters = {1: filter_1, 3: filter_3, 6: filter_6, 12: filter_12}

    abt = su.associate_time_n_factors(table = data_pd, spatial_sum = True, time_filters = time_filters,
                                      times = [1, 3, 12])