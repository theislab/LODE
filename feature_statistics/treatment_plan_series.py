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
    DATA_POINTS = ["study_date", "total_fluid", "injections", "cur_va_rounded", "next_va", "cumsum_injections"]
    META_DATA = ["patient_id", "laterality", "diagnosis"]
    SEG_PATHS = glob.glob(os.path.join(SEG_DIR, "*"))
    DICOM_PATHS = glob.glob(os.path.join(OCT_DIR, "*/*/*/*.dcm"))
    TIME_POINTS = [1, 3, 6, 12, 24]

    FIELDS = ["study_date", "cumsum_injections", "total_fluid", "cur_va_rounded"]

    def __init__(self, meta_data, time_line, table):
        self.patient_id = meta_data[MeasureSeqTimeUntilDry.META_DATA[0]]
        self.laterality = meta_data[MeasureSeqTimeUntilDry.META_DATA[1]]
        self.diagnosis = meta_data[MeasureSeqTimeUntilDry.META_DATA[2]]

        log = {"sequence": f"{self.patient_id}_{self.laterality}"}
        for time_point in MeasureSeqTimeUntilDry.TIME_POINTS:
            for field in MeasureSeqTimeUntilDry.FIELDS:
                log[f"{field}_{time_point}"] = time_line[time_point][field]

        self.log = log

    @classmethod
    def from_record(cls, record_table):
        time_utils = TimeUtils(record_table = record_table)
        time_line = time_utils.time_line
        fluid_time_line = None

        # initalize sequence class
        super().__init__(SeqUtils, time_line = time_line)

        # reset index of table
        record_table.reset_index(inplace = True)

        # add total fluid
        record_table = sum_etdrs_columns(record_table, rings = [1, 2], regions = MeasureSeqTimeUntilDry.REGIONS,
                                         features = [3, 4], foveal_region = ["C0"], new_column_name = "total_fluid")

        # create cumsum injections column
        record_table["cumsum_injections"] = record_table.injections.cumsum()

        total_number_injections = get_total_number_of_injections(table = record_table)

        # round va
        record_table.insert(loc = 10, column = "cur_va_rounded", value = record_table.cur_va.round(2))

        for data_point in ["study_date"]:
            time_line = time_utils.assign_date_to_timeline_dec(time_line = time_line, item = data_point,
                                                          total_number_injections = total_number_injections)

        for data_point in ["cumsum_injections", "total_fluid", "cur_va_rounded"]:
            time_line = time_utils.assign_features_to_timeline_dec(time_line = time_line, item = data_point,
                                                                   total_number_injections = total_number_injections)

        return cls(meta_data = record_table.iloc[0],
                   time_line = time_line,
                   table = record_table)

    @classmethod
    def n_cut_months(cls, fluid_time_line, record_table):
        ft_study_dates = [fluid_time_line[t_key]["study_date"] for t_key in fluid_time_line.keys()]

        # get final date that is not nan
        for ft_study_date in ft_study_dates:
            if ft_study_date is not np.nan:
                last_study_date = ft_study_date

        r_study_dates = record_table.study_date.tolist()

        cut_dates = []
        ft_table_index = False
        for r_study_date in r_study_dates:
            if ft_table_index:
                cut_dates.append(r_study_date)
            if last_study_date == r_study_date:
                ft_table_index = True

        return len(cut_dates)

    @classmethod
    def n_excluded_months(cls, time_line, record_table):
        t_study_dates = [time_line[t_key]["study_date"] for t_key in time_line.keys()]
        r_study_dates = record_table.study_date.tolist()

        excluded = 0
        for r_study_date in r_study_dates:
            if r_study_date not in t_study_dates:
                excluded += 1

        return excluded


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
    seq_pd = seq_pd.loc[filter_]

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
    time_until_dry_pd.to_csv(os.path.join(WORK_SPACE, "sequence_data/treatment_decision_data.csv"))
