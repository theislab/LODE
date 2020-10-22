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
    DATA_POINTS = ["total_fluid", "cur_va_rounded", "next_va", "cumsum_injection"]
    META_DATA = ["patient_id", "laterality", "diagnosis"]
    SEG_PATHS = glob.glob(os.path.join(SEG_DIR, "*"))
    DICOM_PATHS = glob.glob(os.path.join(OCT_DIR, "*/*/*/*.dcm"))
    TIME_POINTS = [1, 3, 6, 12, 24]
    FIELDS = ['study_date', 'total_fluid', 'time_range', 'time_range_before', 'time_range_after', 'insertion_type',
              'cur_va_rounded', 'next_va']

    def __init__(self, meta_data, time_line):
        self.patient_id = meta_data[MeasureSeqTimeUntilDry.META_DATA[0]]
        self.laterality = meta_data[MeasureSeqTimeUntilDry.META_DATA[1]]
        self.diagnosis = meta_data[MeasureSeqTimeUntilDry.META_DATA[2]]

        if sum([bool(time_line[key]) for key in time_line.keys()]) > 0:
            self.log = MeasureSeqTimeUntilDry.create_log(patient_id = self.patient_id,
                                                         laterality = self.laterality,
                                                         time_line = time_line)

        else:
            for key in time_line.keys():
                for field in MeasureSeqTimeUntilDry.FIELDS:
                    time_line[key][field] = None

            self.log = MeasureSeqTimeUntilDry.create_log(patient_id = self.patient_id,
                                                         laterality = self.laterality,
                                                         time_line = time_line)


    @classmethod
    def create_log(cls, patient_id, laterality, time_line):
        log = {"sequence": f"{patient_id}_{laterality}"}
        for time_point in MeasureSeqTimeUntilDry.TIME_POINTS:
            for field in MeasureSeqTimeUntilDry.FIELDS:
                log[f"{field}_{time_point}"] = time_line[time_point][field]
        return log

    @classmethod
    def from_record(cls, record_table):

        # reset index of table
        record_table.reset_index(inplace = True)

        # add total fluid
        record_table = sum_etdrs_columns(record_table, rings = [1, 2], regions = MeasureSeqTimeUntilDry.REGIONS,
                                         features = [3, 4], foveal_region = ["C0"], new_column_name = "total_fluid")

        # round va
        record_table.insert(loc = 10, column = "cur_va_rounded", value = record_table.cur_va.round(2))

        # create cumsum injections column
        record_table["cumsum_injections"] = record_table.injections.cumsum()

        total_number_injections = get_total_number_of_injections(table = record_table)

        time_utils = TimeUtils(record_table = record_table)
        time_line = time_utils.time_line

        # initalize sequence class
        super().__init__(SeqUtils, time_line = time_line)

        if total_number_injections > 0:
            for data_point in MeasureSeqTimeUntilDry.DATA_POINTS:
                time_line = time_utils.assign_to_timeline_str(time_line = time_line, item = data_point,
                                                              total_number_injections = total_number_injections)

        return cls(meta_data = record_table.iloc[0],
                   time_line = time_line)


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

    PATIENT_ID = 312696  # 2005
    LATERALITY = "L"

    filter_ = (seq_pd.patient_id == PATIENT_ID) & (seq_pd.laterality == LATERALITY)
    # seq_pd = seq_pd.loc[filter_]

    unique_records = seq_pd[["patient_id", "laterality"]].drop_duplicates()

    time_until_dry = []
    for patient, lat in tqdm(unique_records.itertuples(index = False)):
        print(patient, lat)
        record_pd = seq_pd[(seq_pd.patient_id == patient) & (seq_pd.laterality == lat)]
        time_until_dry.append(MeasureSeqTimeUntilDry.from_record(record_pd))

    time_series_log = []

    for measurem in time_until_dry:
        time_series_log.append(measurem.log)

    time_until_dry_pd = pd.DataFrame(time_series_log)

    # remove non treated records
    time_until_dry_pd = time_until_dry_pd[time_until_dry_pd['study_date_1'].notna()]
    time_until_dry_pd.to_csv(os.path.join(WORK_SPACE, "sequence_data/longitudinal_properties.csv"))
