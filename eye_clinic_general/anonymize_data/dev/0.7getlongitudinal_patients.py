import pandas as pd
import os

dwh_tables_dir = "/home/olle/PycharmProjects/LODE/workspace/export2/dwh_tables"
temp_tables_dir = "/home/olle/PycharmProjects/LODE/workspace/export2/temp_tables"

id_mapping = "pseudo_real_id_mapping.csv"

diagnosis_table = "diagnosen.csv"
procedure_table = "prozeduren.csv"
previous_export = "pseudo_id_key.csv"

id_mapping_table = pd.read_csv(os.path.join(temp_tables_dir, id_mapping))
id_mapping_table = id_mapping_table.drop(columns = ["Unnamed: 0.1", "Unnamed: 0"])

diagnosis_table = pd.read_csv(os.path.join(dwh_tables_dir, diagnosis_table))
procedure_table = pd.read_csv(os.path.join(dwh_tables_dir, procedure_table))
prev_export_table = pd.read_csv(os.path.join(dwh_tables_dir, previous_export), sep = ";")

######### filter all patients with visits from at least 3 months
test_pd = id_mapping_table[id_mapping_table.patient_id == 11458881][["patient_id", "study_date"]]
test_pd["study_date_dt"] = pd.to_datetime(test_pd.study_date.astype(int).astype(str))
test_pd = test_pd.sort_values(['patient_id', 'study_date_dt'])[['patient_id', 'study_date', 'study_date_dt']]
test_pd["time_diff"] = test_pd.sort_values(['patient_id', 'study_date_dt']).groupby('patient_id')['study_date_dt'].diff(
    1)
test_pd["time_span"] = test_pd.sort_values(['patient_id', 'study_date_dt']).groupby('patient_id')['study_date_dt'].diff(
    1).sum().days

# convert study date to time stamp
# remove invalid date entries
invalid_date_info = id_mapping_table.study_date == -1
id_mapping_table = id_mapping_table[~invalid_date_info]

id_mapping_table["study_date_dt"] = pd.to_datetime(id_mapping_table.study_date.astype(int).astype(str))

id_mapping_table = id_mapping_table.sort_values(['patient_id', 'study_date_dt'])[['patient_id', 'pseudo_id',
                                                                                  'laterality', 'study_date',
                                                                                  'study_date_dt']]

id_mapping_table["time_diff"] = id_mapping_table.sort_values(['patient_id', 'study_date_dt']).groupby('patient_id')[
    'study_date_dt'].diff(1)

# drop time diff na
id_mapping_table = id_mapping_table[~id_mapping_table["time_diff"].isna()]

# convert days to ints
id_mapping_table["time_diff"] = id_mapping_table["time_diff"].dt.days
id_mapping_table["time_span"] = id_mapping_table.groupby('patient_id')['time_diff'].sum()

# drop na columns
id_mapping_table = id_mapping_table[~id_mapping_table["time_diff"].isna()]
id_mapping_table["pseudo_id"] = id_mapping_table.pseudo_id.astype(int)

all_patients = id_mapping_table[["patient_id"]].drop_duplicates()
all_patients["time_span"] = id_mapping_table.groupby('patient_id')['time_diff'].sum().values

all_longitudinal_patients = all_patients[all_patients.time_span > 90]

id_mapping_longitudinal_patients = pd.merge(all_longitudinal_patients, id_mapping_table, left_on = "patient_id",
                                            right_on = "patient_id", how = "left")

# cast pseudo id to int id_mapping_longitudinal_patients = id_mapping_longitudinal_patients.groupby(
# "patient_id").apply(lambda x: x.sort_values(['study_date_dt'], ascending=True))

longitudinal_patients = id_mapping_longitudinal_patients[["patient_id", "pseudo_id"]].drop_duplicates()

########## filter patients with AMD
# join in diagnosis tabke
long_patients_dkey = pd.merge(longitudinal_patients, diagnosis_table,
                              left_on = "patient_id", right_on = "PATNR", how = "left")

# drop patients without diagnosis
long_patients_dkey_nonna = long_patients_dkey[~long_patients_dkey.DKEY.isna()]

# check if patient has diagnossi H35.3** given
AMD_BOOL = long_patients_dkey_nonna.DKEY.str.contains("H35.3")

# filter for AMD patients only
long_patients_amd = long_patients_dkey_nonna.loc[AMD_BOOL.values]

######### filter patients recieving procedure 5-156.9
# join in diagnosis tabke
long_patients_amd_proc = pd.merge(long_patients_amd, procedure_table,
                                  left_on = "patient_id", right_on = "PATNR", how = "left")

# check which patients have procedure code
long_patients_amd_proc_nonna = long_patients_amd_proc[~long_patients_amd_proc.ICPML.isna()]
PROC_BOOL = long_patients_amd_proc_nonna.ICPML.str.contains("5-156.9")

# filter oput any patients who do not have code
long_patients_amd_inj = long_patients_amd_proc_nonna[PROC_BOOL]

longitudinal_table_current_export = long_patients_amd_inj[
    ['patient_id', 'pseudo_id', 'DKEY', 'ICPML']].drop_duplicates()

# filter old patients
prev_export_filter = longitudinal_table_current_export.pseudo_id.isin(prev_export_table.pseudo_id.values.tolist())

longitudinal_table_only_current_export = longitudinal_table_current_export[~prev_export_filter]
longitudinal_table_only_current_export = longitudinal_table_only_current_export[
    ["patient_id", "pseudo_id"]].drop_duplicates()

final_longitudinal_abt = pd.merge(longitudinal_table_only_current_export,
                                  id_mapping_table[["patient_id", "laterality", "study_date_dt"]],
                                  left_on = "patient_id", right_on = "patient_id", how = "left")

longitudinal_table_only_current_export.to_csv("./lognitudinal_records/longitudinal_patients_export_14-12-2020.csv")
final_longitudinal_abt.to_csv("./lognitudinal_records/longitudinal_time_series_14-12-2020.csv")
