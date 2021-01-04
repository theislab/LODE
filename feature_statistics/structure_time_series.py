from copy import copy

from feature_statistics.config import WORK_SPACE, SEG_DIR, OCT_DIR
import os
import pandas as pd
import numpy as np
from feature_statistics.utils.time_utils import TimeUtils
from feature_statistics.utils.pandas_utils import sum_etdrs_columns
from feature_statistics.utils.util_functions import SeqUtils, get_total_number_of_injections
from tqdm import tqdm
import glob


class MeasureSeqTimeUntilDry(SeqUtils):
    NUM_SEGMENTATIONS = 4
    REGIONS = ["S", "N", "I", "T"]
    FLUIDS = ["3", "4"]

    FEATURE_MAPPING_DICT = {"intra_retinal_fluid": 3,
                            "sub_retinal_fluid": 4,
                            "srhm": 5,
                            "fibrovascular_ped": 7,
                            "choroid": 10,
                            "drusen": 8,
                            "rpe": 6,
                            "epiretinal_membrane": 1,
                            "fibrosis": 13}

    ETDRS_REGIONS = ["T1", "T2", "S1", "S2", "N1", "N2", "C0", "I1", "I2"]

    DATA_POINTS = ["total_fluid",
                   "cur_va_rounded",
                   "next_va",
                   "cumsum_injections",
                   'intra_retinal_fluid',
                   'sub_retinal_fluid',
                   'srhm',
                   'fibrovascular_ped',
                   'choroid',
                   'drusen',
                   'rpe',
                   'epiretinal_membrane',
                   'fibrosis']

    META_DATA = ["patient_id", "laterality", "diagnosis"]
    SEG_PATHS = glob.glob(os.path.join(SEG_DIR, "*"))
    DICOM_PATHS = glob.glob(os.path.join(OCT_DIR, "*/*/*/*.dcm"))
    TIME_POINTS = [1, 3, 6, 12, 24]

    FIELDS = ['study_date',
              'total_fluid',
              'time_range',
              'time_range_before',
              'time_range_after',
              'insertion_type',
              'cur_va_rounded',
              'next_va']

    def __init__(self):
        pass

    def create_log(self, patient_id, laterality, time_line, data_points):
        log = {"sequence": f"{patient_id}_{laterality}"}
        for time_point in MeasureSeqTimeUntilDry.TIME_POINTS:
            for field in MeasureSeqTimeUntilDry.FIELDS + data_points:
                log[f"{field}_{time_point}"] = time_line[time_point][field]
        return log

    def add_feature_aggregates(self, record_table):
        """
        Parameters
        ----------
        record_table :

        Returns
        -------

        """
        record_table = sum_etdrs_columns(record_table,
                                         rings = [1, 2],
                                         regions = MeasureSeqTimeUntilDry.REGIONS,
                                         features = [3, 4],
                                         foveal_region = ["C0"],
                                         new_column_name = "total_fluid")

        for feature in MeasureSeqTimeUntilDry.FEATURE_MAPPING_DICT.keys():
            record_table = sum_etdrs_columns(record_table,
                                             rings = [1, 2],
                                             regions = MeasureSeqTimeUntilDry.REGIONS,
                                             features = [MeasureSeqTimeUntilDry.FEATURE_MAPPING_DICT[feature]],
                                             foveal_region = ["C0"],
                                             new_column_name = feature)

        return record_table

    def assign_region_resolved_metrics(self, data_points):
        """
        Parameters
        ----------
        data_points :

        Returns
        -------

        """
        for feature in MeasureSeqTimeUntilDry.FEATURE_MAPPING_DICT.keys():
            for region in MeasureSeqTimeUntilDry.ETDRS_REGIONS:
                feature_label = MeasureSeqTimeUntilDry.FEATURE_MAPPING_DICT[feature]

                data_points.append(region + "_" + str(feature_label))
        return data_points

    def assign_aggregate_metrics(self, data_points):
        return data_points

    def from_record(self, record_table, region_resolved):

        # reset index of table
        record_table.reset_index(inplace = True)

        record_table = self.add_feature_aggregates(record_table)

        # round va
        record_table.insert(loc = 10, column = "cur_va_rounded", value = record_table.cur_va.round(2))

        # create cumsum injections column
        record_table["cumsum_injections"] = record_table.injections.cumsum()

        total_number_injections = get_total_number_of_injections(table = record_table)

        time_utils = TimeUtils(record_table = record_table)
        time_line = time_utils.time_line

        # initialize sequence class
        # super().__init__(SeqUtils)

        if region_resolved:
            data_points = self.assign_region_resolved_metrics(copy(MeasureSeqTimeUntilDry.DATA_POINTS))
        else:
            data_points = copy(MeasureSeqTimeUntilDry.DATA_POINTS)

        meta_data = record_table.iloc[0]

        patient_id = meta_data[MeasureSeqTimeUntilDry.META_DATA[0]]
        laterality = meta_data[MeasureSeqTimeUntilDry.META_DATA[1]]
        diagnosis = meta_data[MeasureSeqTimeUntilDry.META_DATA[2]]

        # check so patient is treated with no lens surgery
        if (total_number_injections > 0) and (sum(record_table.lens_surgery) == 0):
            for data_point in data_points:
                time_line = time_utils.assign_to_timeline_str(time_line = time_line,
                                                              item = data_point,
                                                              total_number_injections = total_number_injections)

            log = self.create_log(patient_id = patient_id, laterality = laterality, time_line = time_line, data_points=data_points)
        else:
            log = None
        return log


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

    region_resolved = True

    PATIENT_ID = 516  # 2005
    LATERALITY = "L"

    mstd = MeasureSeqTimeUntilDry()
    filter_ = (seq_pd.patient_id == PATIENT_ID) & (seq_pd.laterality == LATERALITY)
    # seq_pd = seq_pd.loc[filter_]
    # seq_pd = seq_pd.iloc[0:10]

    unique_records = seq_pd[["patient_id", "laterality"]].drop_duplicates()

    time_until_dry = []
    for patient, lat in tqdm(unique_records.itertuples(index = False)):
        print(patient, lat)
        record_pd = seq_pd[(seq_pd.patient_id == patient) & (seq_pd.laterality == lat)]
        time_until_dry.append(mstd.from_record(record_pd, region_resolved))

    time_series_log = []

    for measurem in time_until_dry:
        if measurem:
            time_series_log.append(measurem)
        else:
            continue


    time_until_dry_pd = pd.DataFrame(time_series_log)

    # read in naive patient data
    naive_patients = pd.read_csv(os.path.join(WORK_SPACE, "sequence_data/check_naive_patients_clean.csv"),
                                 sep = ",").dropna()
    naive_patients["patient_id"] = naive_patients["patient_id"].astype(int)

    # create sequence id
    naive_patients["sequence"] = naive_patients["patient_id"].astype(str) + "_" + naive_patients["laterality"].astype(
        str)

    time_until_dry_pd_naive = pd.merge(time_until_dry_pd, naive_patients[["sequence", "Naive"]], left_on = "sequence",
                                       right_on = "sequence", how = "left")

    # remove non treated records
    time_until_dry_pd = time_until_dry_pd[time_until_dry_pd['study_date_1'].notna()]
    time_until_dry_pd.to_csv(os.path.join(WORK_SPACE, "sequence_data/longitudinal_properties.csv"))
    time_until_dry_pd_naive.to_csv(os.path.join(WORK_SPACE, "sequence_data/longitudinal_properties_naive.csv"))
